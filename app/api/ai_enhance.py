# app/api/ai_enhance.py
"""
AI-powered CV enhancement API endpoints
REWRITTEN: Accepts CV data as JSON instead of fetching from cloud
"""

import json
import logging
import re
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from openai import AsyncOpenAI

from ..config import get_settings
from ..schemas import (
    CompleteCV,
    AIUsageResponse,
    JobAnalysisRequest,
    JobAnalysisResponse,
)
from ..auth.sessions import get_current_session, record_session_activity
from ..database import get_db
from ..models import AIUsageTracking
import redis

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()

redis_client = redis.Redis(
    host=settings.redis_host or "redis",  # ✅ Use service name from docker-compose
    port=settings.redis_port or 6379,
    db=settings.redis_db or 0,
    decode_responses=True,
)


class AIService:
    """AI service for CV enhancement - works with JSON data directly"""

    def __init__(self):
        self.openai_client = None

        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        logger.info(f"AI Service initialized - OpenAI: {bool(self.openai_client)}")

    async def _track_ai_usage(
        self, session_id: str, service_type: str, tokens_used: int = None
    ):
        """Track AI usage for rate limiting"""
        db = next(get_db())

        try:
            import hashlib

            session_hash = hashlib.sha256(
                f"{session_id}_{settings.secret_key}".encode()
            ).hexdigest()[:64]

            usage = AIUsageTracking(
                session_hash=session_hash,
                service_type=service_type,
                tokens_used=tokens_used,
                cost_estimate=self._estimate_cost(tokens_used) if tokens_used else None,
            )

            db.add(usage)
            db.commit()

        except Exception as e:
            logger.error(f"Failed to track AI usage: {e}")
        finally:
            db.close()

    def _estimate_cost(self, tokens: int, model: str = "gpt-3.5-turbo") -> float:
        """Estimate cost based on tokens"""
        # Rough estimate: $0.002 per 1K tokens for GPT-3.5
        return (tokens / 1000) * 0.002

    async def _check_daily_usage(self, session_id: str) -> Dict[str, int]:
        """Check daily AI usage for rate limiting"""
        db = next(get_db())

        try:
            import hashlib
            from datetime import timedelta

            session_hash = hashlib.sha256(
                f"{session_id}_{settings.secret_key}".encode()
            ).hexdigest()[:64]
            today_start = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            usage_count = (
                db.query(AIUsageTracking)
                .filter(
                    AIUsageTracking.session_hash == session_hash,
                    AIUsageTracking.used_at >= today_start,
                )
                .count()
            )

            return {
                "used_today": usage_count,
                "limit": settings.free_tier_ai_operations,
                "remaining": max(0, settings.free_tier_ai_operations - usage_count),
            }

        except Exception as e:
            logger.error(f"Failed to check daily usage: {e}")
            return {
                "used_today": 0,
                "limit": settings.free_tier_ai_operations,
                "remaining": settings.free_tier_ai_operations,
            }
        finally:
            db.close()

    async def enhance_summary(self, summary: str, cv_context: Dict = None) -> str:
        """Enhance personal summary using AI"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        prompt = f"""
        Improve this professional summary to be more compelling and ATS-friendly:
        
        Original: {summary}
        
        Requirements:
        - Keep it professional and concise (2-3 sentences)
        - Include relevant keywords
        - Highlight key strengths
        - Make it ATS-friendly
        
        Enhanced summary:
        """

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.ai_model_primary,
                messages=[
                    {"role": "system", "content": "You are a professional CV writer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.ai_max_tokens,
                temperature=settings.ai_temperature,
            )

            enhanced_summary = (
                response.choices[0].message.content.strip().strip('"').strip("'")
            )
            return enhanced_summary

        except Exception as e:
            logger.error(f"AI summary enhancement failed: {e}")
            raise HTTPException(status_code=500, detail="AI enhancement failed")

    async def enhance_experience_description(
        self, description: str, company: str, position: str
    ) -> str:
        """Enhance work experience description"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        prompt = f"""
        Improve this work experience description:
        
        Position: {position} at {company}
        Original: {description}
        
        Requirements:
        - Use action verbs
        - Include quantifiable achievements
        - Keep professional tone
        - Make it ATS-friendly
        
        Enhanced description:
        """

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.ai_model_primary,
                messages=[
                    {"role": "system", "content": "You are a professional CV writer."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.ai_max_tokens,
                temperature=settings.ai_temperature,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"AI experience enhancement failed: {e}")
            raise HTTPException(status_code=500, detail="AI enhancement failed")


# Global AI service instance
ai_service = AIService()


@router.get("/usage", response_model=AIUsageResponse)
async def get_ai_usage(session: dict = Depends(get_current_session)):
    """Get current AI usage statistics"""
    usage_stats = await ai_service._check_daily_usage(session["session_id"])
    return AIUsageResponse(**usage_stats)


@router.post("/enhance-summary")
async def enhance_personal_summary(
    request: Dict[str, Any],
    session: dict = Depends(get_current_session),
):
    """Enhance personal summary - receives summary text directly"""

    try:
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "error_code": "AI_LIMIT_REACHED",
                        "used_today": usage_stats["used_today"],
                        "daily_limit": usage_stats["limit"],
                        "hours_until_reset": 24,
                    }
                },
            )

        summary = request.get("current_text", "")
        if not summary:
            raise HTTPException(status_code=400, detail="Summary text is required")

        enhanced_summary = await ai_service.enhance_summary(summary)

        await ai_service._track_ai_usage(session["session_id"], "summary_enhancement")

        return {
            "original_summary": summary,
            "enhanced_summary": enhanced_summary,
            "usage": await ai_service._check_daily_usage(session["session_id"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhance-experience")
async def enhance_experience_description(
    request: Dict[str, Any],
    session: dict = Depends(get_current_session),
):
    """Enhance work experience - receives experience data directly"""

    try:
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "error_code": "AI_LIMIT_REACHED",
                        "used_today": usage_stats["used_today"],
                        "daily_limit": usage_stats["limit"],
                        "hours_until_reset": 24,
                    }
                },
            )

        experiences = request.get("experiences", [])
        if not experiences:
            raise HTTPException(status_code=400, detail="Experience data is required")

        improved_experiences = []

        for exp in experiences:
            improved_desc = await ai_service.enhance_experience_description(
                exp.get("description", ""),
                exp.get("company", ""),
                exp.get("position", ""),
            )
            improved_experiences.append(
                {
                    "company": exp.get("company", ""),
                    "position": exp.get("position", ""),
                    "original": exp.get("description", ""),
                    "improved_description": improved_desc,
                }
            )

        await ai_service._track_ai_usage(
            session["session_id"], "experience_enhancement"
        )

        return {
            "experiences": improved_experiences,
            "usage": await ai_service._check_daily_usage(session["session_id"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Experience enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/improve-section")
async def improve_cv_section(
    request: Dict[str, Any],
    session: dict = Depends(get_current_session),
):
    """Improve a specific CV section - receives section data directly"""

    try:
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "error_code": "AI_LIMIT_REACHED",
                        "used_today": usage_stats["used_today"],
                        "daily_limit": usage_stats["limit"],
                        "hours_until_reset": 24,
                    }
                },
            )

        section = request.get("section")
        if not section:
            raise HTTPException(status_code=400, detail="Section type is required")

        task_id = f"task_{session['session_id']}_{section}_{datetime.now().timestamp()}"

        redis_client.setex(
            f"cv_task:{task_id}",
            3600,
            json.dumps(
                {
                    "status": "processing",
                    "section": section,
                    "created_at": datetime.now().isoformat(),
                }
            ),
        )

        try:
            result = None

            if section == "summary":
                current_text = request.get("current_text", "")
                improved = await ai_service.enhance_summary(current_text)
                result = {"improved_text": improved}

            elif section == "experiences":
                experiences = request.get("experiences", [])
                improved_experiences = []

                for exp in experiences:
                    improved_desc = await ai_service.enhance_experience_description(
                        exp.get("description", ""),
                        exp.get("company", ""),
                        exp.get("position", ""),
                    )
                    improved_experiences.append(
                        {
                            "company": exp.get("company", ""),
                            "position": exp.get("position", ""),
                            "original": exp.get("description", ""),
                            "improved_description": improved_desc,
                        }
                    )

                result = {"experiences": improved_experiences}

            redis_client.setex(
                f"cv_task:{task_id}",
                3600,
                json.dumps(
                    {
                        "status": "completed",
                        "section": section,
                        "result": result,
                        "completed_at": datetime.now().isoformat(),
                    }
                ),
            )

            await ai_service._track_ai_usage(
                session["session_id"], f"section_{section}"
            )

        except Exception as e:
            redis_client.setex(
                f"cv_task:{task_id}",
                3600,
                json.dumps(
                    {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat(),
                    }
                ),
            )

        return {
            "task_id": task_id,
            "status": "processing",
            "check_status_url": f"/cv-ai/task-status/{task_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Section improvement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str, session: dict = Depends(get_current_session)):
    """Get status of an AI enhancement task"""

    task_data = redis_client.get(f"cv_task:{task_id}")

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found or expired")

    task = json.loads(task_data)

    response = {"task_id": task_id, "status": task["status"]}

    if task["status"] == "completed":
        response["result"] = task.get("result", {})
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")

    return response


@router.post("/improve-full-cv")
async def improve_full_cv(
    request: Dict[str, Any],
    session: dict = Depends(get_current_session),
):
    """Improve entire CV - receives CV data as JSON directly"""

    try:
        # Check usage limits
        usage_stats = await ai_service._check_daily_usage(session["session_id"])
        if usage_stats["remaining"] <= 0:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "error_code": "AI_LIMIT_REACHED",
                        "used_today": usage_stats["used_today"],
                        "daily_limit": usage_stats["limit"],
                        "hours_until_reset": 24,
                    }
                },
            )

        # Extract CV data from request body
        cv_data_dict = request.get("cv_data")
        if not cv_data_dict:
            raise HTTPException(
                status_code=400, detail="CV data is required in request body"
            )

        # Parse CV data
        try:
            cv_data = CompleteCV(**cv_data_dict)
        except Exception as e:
            logger.error(f"Failed to parse CV data: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid CV data format: {str(e)}"
            )

        # Optional metadata
        metadata = request.get("metadata", {})
        provider = metadata.get("provider", "unknown")

        logger.info(f"Processing CV enhancement for provider: {provider}")

        # Build CV summary
        cv_summary = {
            "personal_info": {
                "full_name": cv_data.personal_info.full_name
                if cv_data.personal_info
                else "",
                "title": cv_data.personal_info.title if cv_data.personal_info else "",
                "summary": cv_data.personal_info.summary
                if cv_data.personal_info
                else "",
            },
            "experiences": [
                {
                    "company": exp.company,
                    "position": exp.position,
                    "description": exp.description,
                }
                for exp in cv_data.experiences[:3]
            ],
            "skills": [skill.name for skill in cv_data.skills],
        }

        prompt = f"""
        Improve this entire CV with professional, concise suggestions:
        
        CURRENT CV:
        Summary: {cv_summary["personal_info"]["summary"]}
        
        Experience:
        {json.dumps(cv_summary["experiences"], indent=2)}
        
        Skills: {", ".join(cv_summary["skills"])}
        
        Provide improvements in this format:
        
        **Summary:**
        [Improved summary here - 2-3 sentences, professional and impactful]
        
        **Experiences:**
        For each experience, provide:
        **[Company Name]** ([Date Range])
        *[Position]*
        - [Bullet point 1]
        - [Bullet point 2]
        - [Bullet point 3]
        
        **Skills:**
        *Technical Skills:* [comma-separated skills]
        *Soft Skills:* [comma-separated skills]
        
        Keep all text concise, professional, and achievement-focused.
        """

        # Call OpenAI
        response = await ai_service.openai_client.chat.completions.create(
            model=settings.ai_model_premium,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert CV writer. Provide concise, professional improvements.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
        )

        raw_text = response.choices[0].message.content.strip()

        # Parse response
        improvements = {
            "summary": {"main": ""},
            "experiences": {},
            "skills": {"main": {"parsed": {"categories": {}}}},
            "raw": raw_text,
        }

        # Parse summary
        summary_match = re.search(
            r"\*\*Summary:\*\*\s*(.*?)(?=\n\n\*\*|\Z)", raw_text, re.DOTALL
        )
        if summary_match:
            improvements["summary"]["main"] = summary_match.group(1).strip()

        # Parse experiences
        exp_pattern = re.compile(
            r"\*\*([^*\n]+)\*\*\s*\(([^()]+)\)\s*\n\*(.*?)\*\s*\n((?:- .*\n?)+)",
            re.MULTILINE | re.DOTALL,
        )

        for index, match in enumerate(exp_pattern.finditer(raw_text)):
            company = match.group(1).strip()
            position = match.group(3).strip()
            bullets = [
                f"- {bullet.strip()}"
                for bullet in re.findall(r"- (.*?)(?:\n|$)", match.group(4))
                if bullet.strip()
            ]

            improvements["experiences"][f"item_{index}"] = {
                "company": company,
                "position": position,
                "improved": "\n".join(bullets),
                "improved_description": "\n".join(bullets),
            }

        # Parse skills
        skills_match = re.search(
            r"\*\*Skills:\*\*\s*([\s\S]*?)(?=\n\n\*\*|\Z)", raw_text, re.DOTALL
        )
        if skills_match:
            skills_text = skills_match.group(1).strip()
            categories = {}

            category_pattern = re.compile(r"\*(.*?):\*\s*(.*?)(?=\n\*|\Z)", re.DOTALL)
            for category_match in category_pattern.finditer(skills_text):
                category = category_match.group(1).strip()
                skills_list = [
                    skill.strip()
                    for skill in category_match.group(2).split(",")
                    if skill.strip()
                ]
                if skills_list:
                    categories[category] = skills_list

            improvements["skills"]["main"] = {
                "parsed": {"type": "categorized", "categories": categories}
            }

        # Track usage
        tokens_used = response.usage.total_tokens if response.usage else None
        await ai_service._track_ai_usage(
            session["session_id"], "full_cv_enhancement", tokens_used
        )

        await record_session_activity(
            session["session_id"],
            "ai_enhance_full_cv",
            {"provider": provider, "tokens_used": tokens_used},
        )

        return {"improvements": improvements}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full CV improvement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
