# app/job_matching/service.py - SESSION-BASED VERSION
"""
Job Analysis Service adapted for anonymous session-based system
"""

import asyncio
import hashlib
import os
import time
import json
from typing import Dict, Any, Optional, Literal
from datetime import datetime, timedelta
import logging

# Import the knowledge components (these work as-is)
from .knowledge.industry_detector import UniversalIndustryDetector, IndustryType
from .knowledge.skills_matcher import UniversalSkillsMatcher
from .knowledge.recruitment_kb import RecruitmentKnowledgeBase


logger = logging.getLogger(__name__)

# Try to import AI components
try:
    from .agents import create_optimized_job_analysis_agents

    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("CrewAI not available")

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available")

logger = logging.getLogger(__name__)
AnalysisType = Literal["crewai", "claude", "simple"]


class AnalysisError(Exception):
    """Custom exception for analysis errors"""

    pass


class JobAnalysisService:
    """Session-based job analysis service for privacy-first CV platform"""

    def __init__(self):
        # Initialize components
        self.industry_detector = UniversalIndustryDetector()
        self.skills_matcher = UniversalSkillsMatcher()
        self.knowledge_base = RecruitmentKnowledgeBase()

        # Cache for performance
        self._analysis_cache = {}
        self._cache_expiry = timedelta(hours=1)

        # Lazy-loaded AI components
        self._crewai_agents = None
        self._claude_client = None

        logger.info("Job Analysis Service initialized for session-based system")

    def _get_cache_key(
        self, resume_text: str, job_description: str, job_title: str
    ) -> str:
        """Generate cache key for analysis"""
        content = f"{resume_text[:500]}{job_description[:500]}{job_title}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        cached_time = datetime.fromisoformat(cache_entry["timestamp"])
        return datetime.utcnow() - cached_time < self._cache_expiry

    async def analyze_job_match(
        self,
        resume_text: str = None,
        resume_file_path: str = None,
        job_description: str = "",
        job_title: str = "",
        company_name: str = "",
        analysis_type: AnalysisType = "simple",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main job analysis method adapted for session-based system
        """

        # Validate inputs
        if not resume_text and not resume_file_path:
            raise AnalysisError(
                "Either resume_text or resume_file_path must be provided"
            )

        if len(job_description.strip()) < 20:
            raise AnalysisError("Job description must be at least 20 characters")

        if len(job_title.strip()) < 2:
            raise AnalysisError("Job title must be at least 2 characters")

        # If file path provided, read the file
        if resume_file_path and not resume_text:
            try:
                with open(resume_file_path, "r", encoding="utf-8") as f:
                    resume_text = f.read()
            except Exception as e:
                raise AnalysisError(f"Failed to read resume file: {str(e)}")

        # Check cache
        cache_key = self._get_cache_key(resume_text, job_description, job_title)
        if cache_key in self._analysis_cache:
            cache_entry = self._analysis_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(f"Returning cached analysis for session {session_id}")
                result = cache_entry["result"]
                result["cache_used"] = True
                return result

        # Perform analysis based on type
        start_time = time.time()

        try:
            if analysis_type == "crewai" and CREWAI_AVAILABLE:
                result = await self._analyze_with_crewai(
                    resume_text, job_description, job_title, company_name
                )
            elif analysis_type == "claude" and ANTHROPIC_AVAILABLE:
                result = await self._analyze_with_claude(
                    resume_text, job_description, job_title, company_name
                )
            else:
                result = await self._analyze_simple(
                    resume_text, job_description, job_title, company_name
                )

            # Add metadata
            result["analysis_method"] = analysis_type
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            result["session_id"] = session_id
            result["cache_used"] = False

            # Cache the result
            self._analysis_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                f"Analysis completed for session {session_id} in {result['processing_time_ms']}ms"
            )
            return result

        except Exception as e:
            logger.error(f"Analysis failed for session {session_id}: {str(e)}")
            raise AnalysisError(f"Analysis failed: {str(e)}")

    async def _analyze_simple(
        self, resume_text: str, job_description: str, job_title: str, company_name: str
    ) -> Dict[str, Any]:
        """Simple analysis using built-in knowledge base - always works"""

        # Use the universal skills matcher
        match_result = self.skills_matcher.analyze_match(
            resume_text, job_title, job_description
        )

        # Get market insights from knowledge base
        market_insights = self.knowledge_base.query_knowledge(
            f"job market {job_title} {company_name}", k=3
        )

        return {
            "match_score": match_result.overall_match_score,
            "ats_compatibility_score": max(50, match_result.overall_match_score - 10),
            "keywords_present": [
                skill.skill
                for skill in match_result.skill_matches
                if skill.match_score > 70
            ],
            "keywords_missing": [
                skill.skill
                for skill in match_result.skill_matches
                if skill.match_score <= 70
            ],
            "recommendations": match_result.recommendations,
            "strengths": match_result.strengths,
            "improvement_areas": match_result.gaps,
            "should_apply": match_result.should_apply,
            "application_tips": [
                "Tailor your resume to highlight relevant skills",
                "Prepare examples of your experience",
                "Research the company culture",
            ],
            "market_insights": market_insights,
            "industry_detected": match_result.industry,
            "detected_role": match_result.role,
            "confidence_level": match_result.confidence_level,
        }

    async def _analyze_with_crewai(
        self, resume_text: str, job_description: str, job_title: str, company_name: str
    ) -> Dict[str, Any]:
        """Analysis using CrewAI agents"""

        if not CREWAI_AVAILABLE:
            return await self._analyze_simple(
                resume_text, job_description, job_title, company_name
            )

        # Initialize agents if not done
        if not self._crewai_agents:
            self._crewai_agents = create_optimized_job_analysis_agents()

        # Use the agents for analysis
        # This is where you'd implement the CrewAI workflow
        # For now, fall back to simple analysis
        logger.info("CrewAI analysis not fully implemented yet, using simple analysis")
        return await self._analyze_simple(
            resume_text, job_description, job_title, company_name
        )

    async def _analyze_with_claude(
        self, resume_text: str, job_description: str, job_title: str, company_name: str
    ) -> Dict[str, Any]:
        """Analysis using Claude AI"""

        if not ANTHROPIC_AVAILABLE:
            return await self._analyze_simple(
                resume_text, job_description, job_title, company_name
            )

        # Initialize Claude client if not done
        if not self._claude_client:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self._claude_client = Anthropic(api_key=api_key)

        if not self._claude_client:
            return await self._analyze_simple(
                resume_text, job_description, job_title, company_name
            )

        # Use Claude for analysis
        logger.info("Claude analysis not fully implemented yet, using simple analysis")
        return await self._analyze_simple(
            resume_text, job_description, job_title, company_name
        )

    def get_available_methods(self) -> list:
        """Get available analysis methods"""
        methods = ["simple"]
        if CREWAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            methods.append("crewai")
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            methods.append("claude")
        return methods

    def clear_cache(self):
        """Clear the analysis cache"""
        self._analysis_cache.clear()
        logger.info("Analysis cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_entries = sum(
            1 for entry in self._analysis_cache.values() if self._is_cache_valid(entry)
        )
        return {
            "total_entries": len(self._analysis_cache),
            "valid_entries": valid_entries,
            "cache_hit_ratio": "Not tracked yet",
        }

    # Add this method to the JobAnalysisService class in service.py

    def _cleanup_expired_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = []

        for key, entry in self._analysis_cache.items():
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if current_time - cached_time > self._cache_expiry:
                expired_keys.append(key)

        for key in expired_keys:
            del self._analysis_cache[key]

        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
