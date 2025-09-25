# app/job_matching/router.py - SESSION-BASED VERSION
"""
Job Matching Router adapted for anonymous session-based system
"""

import logging
import os
import tempfile
from datetime import datetime
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

import app.celery_app as celery_app

# Import your session management
from ..auth.sessions import get_current_session, record_session_activity
from ..schemas import ErrorResponse

# Import job matching components
from .schema import JobAnalysisRequest, JobAnalysisResponse
from .service import JobAnalysisService, AnalysisError


task_status = {}

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize job analysis service
job_analysis_service = JobAnalysisService()

# File upload constraints
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.get("/available-methods")
async def get_available_analysis_methods():
    """Get available analysis methods"""
    try:
        methods = job_analysis_service.get_available_methods()

        descriptions = {
            "simple": "Built-in Analysis - Uses comprehensive knowledge base",
            "crewai": "CrewAI Multi-Agent System - Advanced AI analysis",
            "claude": "Claude AI - Premium reasoning and insights",
        }

        return {
            "available_methods": methods,
            "descriptions": {
                method: descriptions.get(method, f"{method} analysis")
                for method in methods
            },
            "default_method": "simple",
            "privacy_note": "All analysis is performed without storing your personal data",
        }
    except Exception as e:
        logger.error(f"Error getting analysis methods: {e}")
        raise HTTPException(
            status_code=500, detail="Could not retrieve analysis methods"
        )


@router.post("/analyze", response_model=JobAnalysisResponse)
async def analyze_job_match(
    request: JobAnalysisRequest,
    session: dict = Depends(get_current_session),
    http_request: Request = None,
):
    """
    Analyze job match compatibility - privacy-first approach
    No personal data stored, analysis performed in memory only
    """

    session_id = session.get("session_id")
    logger.info(f"Job analysis request from session {session_id}")

    try:
        # Validate analysis method
        available_methods = job_analysis_service.get_available_methods()
        if request.analysis_method not in available_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis method '{request.analysis_method}' not available. Available: {available_methods}",
            )

        # Perform analysis
        start_time = datetime.utcnow()

        analysis_result = await job_analysis_service.analyze_job_match(
            resume_text=request.resume_text,
            job_description=request.job_description,
            job_title=request.job_title,
            company_name=request.company_name,
            analysis_type=request.analysis_method,
            session_id=session_id,
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Create response
        response = JobAnalysisResponse(
            match_score=analysis_result.get("match_score", 50.0),
            ats_compatibility_score=analysis_result.get(
                "ats_compatibility_score", 50.0
            ),
            keywords_present=analysis_result.get("keywords_present", []),
            keywords_missing=analysis_result.get("keywords_missing", []),
            recommendations=analysis_result.get("recommendations", []),
            strengths=analysis_result.get("strengths", []),
            improvement_areas=analysis_result.get("improvement_areas", []),
            should_apply=analysis_result.get("should_apply", True),
            application_tips=analysis_result.get("application_tips", []),
            analysis_method=analysis_result.get(
                "analysis_method", request.analysis_method
            ),
            market_insights=analysis_result.get("market_insights", []),
        )

        # Record anonymous analytics (no personal data)
        await record_session_activity(
            session_id,
            "job_analysis_completed",
            {
                "analysis_method": request.analysis_method,
                "processing_time_seconds": round(processing_time, 2),
                "match_score_range": f"{int(response.match_score // 10) * 10}-{int(response.match_score // 10) * 10 + 9}",
                "cache_used": analysis_result.get("cache_used", False),
            },
        )

        logger.info(
            f"Job analysis completed for session {session_id} - Score: {response.match_score}%"
        )

        return response

    except AnalysisError as e:
        logger.error(f"Analysis error for session {session_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in job analysis for session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Analysis service temporarily unavailable"
        )


@router.post("/analyze-file", response_model=JobAnalysisResponse)
async def analyze_job_match_with_file(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...),
    job_title: str = Form(...),
    company_name: str = Form(""),
    analysis_method: str = Form("simple"),
    session: dict = Depends(get_current_session),
):
    """
    Analyze job match with uploaded resume file - privacy-first
    File is processed in memory and immediately deleted
    """

    session_id = session.get("session_id")
    logger.info(f"File-based job analysis from session {session_id}")

    temp_file_path = None

    try:
        # Validate file
        if not resume_file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        file_ext = os.path.splitext(resume_file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file_ext}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # Read and validate file content
        file_content = await resume_file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        if len(file_content) < 100:
            raise HTTPException(
                status_code=400, detail="File appears empty or too small"
            )

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_ext, prefix=f"resume_{session_id}_"
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Validate other inputs
        available_methods = job_analysis_service.get_available_methods()
        if analysis_method not in available_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis method '{analysis_method}' not available",
            )

        if len(job_description.strip()) < 20:
            raise HTTPException(
                status_code=400, detail="Job description must be at least 20 characters"
            )

        # Perform analysis
        start_time = datetime.utcnow()

        analysis_result = await job_analysis_service.analyze_job_match(
            resume_file_path=temp_file_path,
            job_description=job_description,
            job_title=job_title,
            company_name=company_name,
            analysis_type=analysis_method,
            session_id=session_id,
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Create response
        response = JobAnalysisResponse(
            match_score=analysis_result.get("match_score", 50.0),
            ats_compatibility_score=analysis_result.get(
                "ats_compatibility_score", 50.0
            ),
            keywords_present=analysis_result.get("keywords_present", []),
            keywords_missing=analysis_result.get("keywords_missing", []),
            recommendations=analysis_result.get("recommendations", []),
            strengths=analysis_result.get("strengths", []),
            improvement_areas=analysis_result.get("improvement_areas", []),
            should_apply=analysis_result.get("should_apply", True),
            application_tips=analysis_result.get("application_tips", []),
            analysis_method=analysis_result.get("analysis_method", analysis_method),
            market_insights=analysis_result.get("market_insights", []),
        )

        # Record analytics
        await record_session_activity(
            session_id,
            "job_analysis_file_completed",
            {
                "analysis_method": analysis_method,
                "file_type": file_ext,
                "file_size_kb": len(file_content) // 1024,
                "processing_time_seconds": round(processing_time, 2),
                "match_score_range": f"{int(response.match_score // 10) * 10}-{int(response.match_score // 10) * 10 + 9}",
            },
        )

        logger.info(f"File analysis completed for session {session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File analysis error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="File analysis failed")
    finally:
        # Always clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")


@router.get("/health")
async def health_check():
    """Health check for job matching service"""
    try:
        available_methods = job_analysis_service.get_available_methods()
        cache_stats = job_analysis_service.get_cache_stats()

        return {
            "status": "healthy",
            "service": "job-matching",
            "version": "1.0.0",
            "privacy_first": True,
            "available_methods": available_methods,
            "cache_stats": cache_stats,
            "supported_file_types": list(ALLOWED_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "timestamp": datetime.utcnow().isoformat(),
            "data_storage": "None - privacy-first design",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.post("/clear-cache")
async def clear_analysis_cache(session: dict = Depends(get_current_session)):
    """Clear analysis cache"""
    try:
        job_analysis_service.clear_cache()

        await record_session_activity(
            session["session_id"], "cache_cleared", {"component": "job_analysis"}
        )

        return {
            "message": "Analysis cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.get("/stats")
async def get_service_stats(session: dict = Depends(get_current_session)):
    """Get anonymous service statistics"""
    try:
        cache_stats = job_analysis_service.get_cache_stats()
        available_methods = job_analysis_service.get_available_methods()

        return {
            "service": "job-matching",
            "privacy_model": "anonymous_sessions",
            "data_retention": "none",
            "available_methods": available_methods,
            "cache_statistics": cache_stats,
            "supported_industries": 8,
            "knowledge_base": "built_in",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.post("/analyze-async")
async def analyze_job_match_async(
    request: JobAnalysisRequest,
    session: dict = Depends(get_current_session),
):
    """
    Analyze job match asynchronously - returns task ID immediately
    Useful for heavy analysis or when immediate response not required
    """

    session_id = session.get("session_id")
    logger.info(f"Async job analysis request from session {session_id}")

    try:
        # Validate analysis method
        available_methods = job_analysis_service.get_available_methods()
        if request.analysis_method not in available_methods:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis method '{request.analysis_method}' not available",
            )

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Dispatch async task
        celery_task = analyze_job_match_async.apply_async(
            args=[
                session_id,
                request.resume_text,
                None,  # resume_file_path
                request.job_description,
                request.job_title,
                request.company_name,
                request.analysis_method,
            ],
            task_id=task_id,
            queue="default",
        )

        # Store task info
        task_status[task_id] = {
            "celery_task_id": celery_task.id,
            "status": "processing",
            "type": "job-match-analysis",
            "analysis_method": request.analysis_method,
            "job_title": request.job_title,
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Record activity
        await record_session_activity(
            session_id,
            "job_analysis_async_started",
            {
                "task_id": task_id,
                "analysis_method": request.analysis_method,
                "job_title": request.job_title,
            },
        )

        logger.info(f"Async job analysis task dispatched: {task_id}")

        return {
            "task_id": task_id,
            "status": "processing",
            "message": f"Job analysis ({request.analysis_method}) in progress",
            "check_status_url": f"/api/job-matching/task-status/{task_id}",
            "estimated_completion": "30-60 seconds",
        }

    except Exception as e:
        logger.error(f"Error starting async job analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to start async analysis")


@router.get("/task-status/{task_id}")
async def check_task_status(
    task_id: str,
    session: dict = Depends(get_current_session),
):
    """Check the status of an async job analysis task"""

    session_id = session.get("session_id")

    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = task_status[task_id]

    # Verify task belongs to current session
    if task_info.get("session_id") != session_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        from celery.result import AsyncResult

        celery_task_id = task_info.get("celery_task_id", task_id)
        task_result = AsyncResult(celery_task_id, app=celery_app)

        if task_result.ready():
            if task_result.successful():
                result = task_result.get()
                task_status[task_id]["status"] = "completed"
                task_status[task_id]["completed_at"] = datetime.utcnow().isoformat()
                task_status[task_id]["result"] = result

                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "analysis_method": task_info.get("analysis_method"),
                    "job_title": task_info.get("job_title"),
                    "completed_at": task_status[task_id]["completed_at"],
                }
            else:
                error = str(task_result.result)
                task_status[task_id]["status"] = "failed"
                task_status[task_id]["error"] = error

                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": error,
                    "analysis_method": task_info.get("analysis_method"),
                    "job_title": task_info.get("job_title"),
                }
        else:
            return {
                "task_id": task_id,
                "status": "processing",
                "message": f"Job analysis ({task_info.get('analysis_method')}) in progress",
                "job_title": task_info.get("job_title"),
                "progress": "Analyzing skills and experience compatibility...",
                "created_at": task_info.get("created_at"),
            }

    except Exception as e:
        logger.error(f"Error checking task status: {e}")
        return {
            "task_id": task_id,
            "status": task_info.get("status", "processing"),
            "error": "Could not check task status",
            "message": "Please try the synchronous endpoint if this persists",
        }
