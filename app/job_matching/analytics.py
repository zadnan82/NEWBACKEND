# app/job_matching/analytics.py
"""
Optional anonymous analytics for job matching service
Tracks usage patterns without storing personal data
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisEvent:
    """Anonymous analytics event for analysis requests"""

    event_type: str = "job_analysis"
    analysis_method: str = ""
    job_category: str = ""  # Categorized job title, not exact
    match_score_range: str = ""  # e.g., "70-80", not exact score
    processing_time_ms: int = 0
    success: bool = True
    error_category: Optional[str] = None
    timestamp: str = ""
    user_hash: str = ""  # Hashed user ID, not actual ID
    session_id: str = ""  # Anonymous session identifier


class AnonymousAnalytics:
    """
    Anonymous analytics collector that respects user privacy
    - No personal information stored
    - No resume content stored
    - No job descriptions stored
    - Only aggregated usage patterns
    """

    def __init__(self):
        self.enabled = os.getenv("ENABLE_ANALYTICS", "false").lower() == "true"
        self.analytics_file = os.getenv("ANALYTICS_FILE", "logs/analytics.jsonl")
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Ensure analytics log directory exists"""
        if self.enabled:
            os.makedirs(os.path.dirname(self.analytics_file), exist_ok=True)

    def _categorize_job_title(self, job_title: str) -> str:
        """Categorize job title to avoid storing exact titles"""
        title_lower = job_title.lower()

        # Define broad categories
        categories = {
            "software_engineer": [
                "software",
                "developer",
                "programmer",
                "engineer",
                "backend",
                "frontend",
                "fullstack",
            ],
            "data_scientist": [
                "data scientist",
                "data analyst",
                "machine learning",
                "ml engineer",
                "ai",
            ],
            "product_manager": ["product manager", "product owner", "pm"],
            "designer": ["designer", "ui", "ux", "graphic", "visual"],
            "marketing": ["marketing", "digital marketing", "content", "social media"],
            "sales": ["sales", "account manager", "business development"],
            "devops": ["devops", "sre", "infrastructure", "cloud", "platform"],
            "qa": ["qa", "quality assurance", "tester", "test"],
            "manager": ["manager", "director", "lead", "head", "supervisor"],
            "consultant": ["consultant", "advisor", "specialist"],
            "analyst": ["analyst", "research", "business analyst"],
            "admin": ["admin", "administrative", "assistant", "coordinator"],
        }

        for category, keywords in categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category

        return "other"

    def _categorize_score(self, score: float) -> str:
        """Categorize match score into ranges"""
        if score >= 90:
            return "90-100"
        elif score >= 80:
            return "80-89"
        elif score >= 70:
            return "70-79"
        elif score >= 60:
            return "60-69"
        elif score >= 50:
            return "50-59"
        elif score >= 40:
            return "40-49"
        elif score >= 30:
            return "30-39"
        elif score >= 20:
            return "20-29"
        elif score >= 10:
            return "10-19"
        else:
            return "0-9"

    def _hash_user_id(self, user_id: str) -> str:
        """Create anonymous hash of user ID"""
        # Add salt to prevent rainbow table attacks
        salt = os.getenv("ANALYTICS_SALT", "default_salt_change_in_production")
        return hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()[:12]

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error types without exposing details"""
        error_lower = error_message.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "rate limit" in error_lower:
            return "rate_limit"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "validation"
        elif "api" in error_lower:
            return "api_error"
        elif "service" in error_lower:
            return "service_error"
        else:
            return "unknown_error"

    def track_analysis_request(
        self,
        analysis_method: str,
        job_title: str,
        match_score: Optional[float] = None,
        processing_time_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Track an analysis request with anonymous data"""

        if not self.enabled:
            return

        try:
            event = AnalysisEvent(
                analysis_method=analysis_method,
                job_category=self._categorize_job_title(job_title),
                match_score_range=self._categorize_score(match_score)
                if match_score is not None
                else "unknown",
                processing_time_ms=processing_time_ms,
                success=success,
                error_category=self._categorize_error(error_message)
                if error_message
                else None,
                timestamp=datetime.utcnow().isoformat(),
                user_hash=self._hash_user_id(user_id) if user_id else "anonymous",
                session_id=session_id or "unknown",
            )

            # Write to JSONL file for analysis
            with open(self.analytics_file, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")

            logger.debug(f"Analytics event tracked: {event.event_type}")

        except Exception as e:
            # Analytics should never break the main application
            logger.error(f"Analytics tracking failed: {e}")

    def get_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get aggregated usage statistics (admin function)"""
        if not self.enabled or not os.path.exists(self.analytics_file):
            return {"error": "Analytics not enabled or no data available"}

        try:
            # Read recent events
            events = []
            cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)

            with open(self.analytics_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event_time = datetime.fromisoformat(
                            event["timestamp"]
                        ).timestamp()
                        if event_time >= cutoff:
                            events.append(event)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

            # Calculate aggregated statistics
            total_requests = len(events)
            successful_requests = sum(1 for e in events if e["success"])

            method_stats = {}
            job_category_stats = {}
            score_distribution = {}
            error_stats = {}

            for event in events:
                # Method usage
                method = event["analysis_method"]
                method_stats[method] = method_stats.get(method, 0) + 1

                # Job categories
                category = event["job_category"]
                job_category_stats[category] = job_category_stats.get(category, 0) + 1

                # Score distribution
                if event["success"] and event["match_score_range"] != "unknown":
                    score_range = event["match_score_range"]
                    score_distribution[score_range] = (
                        score_distribution.get(score_range, 0) + 1
                    )

                # Error categories
                if not event["success"] and event["error_category"]:
                    error_cat = event["error_category"]
                    error_stats[error_cat] = error_stats.get(error_cat, 0) + 1

            # Calculate average processing time for successful requests
            successful_times = [
                e["processing_time_ms"]
                for e in events
                if e["success"] and e["processing_time_ms"] > 0
            ]
            avg_processing_time = (
                sum(successful_times) / len(successful_times) if successful_times else 0
            )

            return {
                "period_days": days,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0,
                "average_processing_time_ms": round(avg_processing_time, 2),
                "method_usage": method_stats,
                "job_categories": job_category_stats,
                "score_distribution": score_distribution,
                "error_categories": error_stats,
                "unique_users": len(
                    set(e["user_hash"] for e in events if e["user_hash"] != "anonymous")
                ),
            }

        except Exception as e:
            logger.error(f"Failed to generate usage stats: {e}")
            return {"error": "Failed to generate statistics"}


# Global analytics instance
analytics = AnonymousAnalytics()


def track_job_analysis(
    analysis_method: str,
    job_title: str,
    match_score: Optional[float] = None,
    processing_time_ms: int = 0,
    success: bool = True,
    error_message: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Convenient function to track job analysis events"""
    analytics.track_analysis_request(
        analysis_method=analysis_method,
        job_title=job_title,
        match_score=match_score,
        processing_time_ms=processing_time_ms,
        success=success,
        error_message=error_message,
        user_id=user_id,
        session_id=session_id,
    )
