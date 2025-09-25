# app/job_matching/tasks_async.py - SESSION-BASED VERSION
"""
Async tasks for job matching service using Celery
Adapted for anonymous session-based system
"""

from celery import shared_task
import logging
import time
import asyncio
from datetime import datetime

from ..auth.sessions import get_session_by_id, record_session_activity
from .service import JobAnalysisService, AnalysisError

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="analyze_job_match_async", max_retries=2)
def analyze_job_match_async(
    self,
    session_id,
    resume_text=None,
    resume_file_path=None,
    job_description="",
    job_title="",
    company_name="",
    analysis_type="simple",
):
    """Async version of job match analysis for session-based system"""
    logger.info(f"Starting async job analysis for session {session_id}")

    start_time = time.time()

    try:
        # Initialize job analysis service
        job_service = JobAnalysisService()

        # Perform the analysis using asyncio.run to handle the async method
        analysis_data = asyncio.run(
            job_service.analyze_job_match(
                resume_text=resume_text,
                resume_file_path=resume_file_path,
                job_description=job_description,
                job_title=job_title,
                company_name=company_name,
                analysis_type=analysis_type,
                session_id=session_id,
            )
        )

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Async job analysis completed in {processing_time}ms")

        # Record session activity (async)
        asyncio.run(
            record_session_activity(
                session_id,
                "job_analysis_async_completed",
                {
                    "analysis_method": analysis_type,
                    "processing_time_ms": processing_time,
                    "match_score": analysis_data.get("match_score"),
                    "async_task": True,
                    "success": True,
                },
            )
        )

        # Prepare final result
        final_result = {
            **analysis_data,
            "status": "completed",
            "processing_time_ms": processing_time,
            "session_id": session_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "async_processing": True,
        }

        logger.info(
            f"Async job analysis completed successfully for session {session_id}"
        )
        return final_result

    except AnalysisError as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"Analysis error in async task: {str(e)}")

        # Record failed attempt
        try:
            asyncio.run(
                record_session_activity(
                    session_id,
                    "job_analysis_async_failed",
                    {
                        "analysis_method": analysis_type,
                        "processing_time_ms": processing_time,
                        "error_type": "analysis_error",
                        "error_message": str(e)[:200],
                        "async_task": True,
                        "success": False,
                    },
                )
            )
        except Exception as record_error:
            logger.error(f"Failed to record error activity: {record_error}")

        return {
            "error": str(e),
            "status": "failed",
            "error_type": "analysis_error",
            "processing_time_ms": processing_time,
            "session_id": session_id,
        }

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"System error in async job analysis: {str(e)}")

        # Retry the task if it failed
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying async task (attempt {self.request.retries + 1}/{self.max_retries})"
            )
            self.retry(exc=e, countdown=2**self.request.retries)
        else:
            logger.error("Max retries reached, async task failed permanently")

        return {
            "error": str(e),
            "status": "failed",
            "error_type": "system_error",
            "processing_time_ms": processing_time,
            "session_id": session_id,
            "retry_attempts": self.request.retries,
        }


@shared_task(name="cleanup_old_analysis_cache")
def cleanup_old_analysis_cache():
    """Clean up old cached analysis results"""
    try:
        job_service = JobAnalysisService()

        # Get cache stats before cleanup
        old_stats = job_service.get_cache_stats()

        # Clear expired entries (service handles this internally)
        job_service._cleanup_expired_cache()

        new_stats = job_service.get_cache_stats()

        cleanup_info = {
            "cleaned_entries": old_stats["total_entries"] - new_stats["total_entries"],
            "remaining_entries": new_stats["total_entries"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Cache cleanup completed: {cleanup_info}")
        return cleanup_info

    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
