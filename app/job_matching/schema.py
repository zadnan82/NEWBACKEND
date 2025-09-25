# app/job_matching/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal, Union
import re


class JobAnalysisRequest(BaseModel):
    resume_text: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Resume text content (50-50,000 characters)",
    )
    job_description: str = Field(
        ...,
        min_length=20,
        max_length=20000,
        description="Job description text (20-20,000 characters)",
    )
    job_title: str = Field(
        ..., min_length=2, max_length=200, description="Job title (2-200 characters)"
    )
    company_name: str = Field(
        "", max_length=200, description="Company name (optional, max 200 characters)"
    )
    analysis_method: Literal["crewai", "claude", "simple"] = Field(
        "simple", description="Analysis method to use"
    )

    @field_validator("resume_text")
    @classmethod
    def validate_resume_text(cls, v):
        v = re.sub(r"\s+", " ", v.strip())
        words = v.split()
        if len(words) < 20:
            raise ValueError(
                "Resume must contain at least 20 words of meaningful content"
            )

        resume_indicators = [
            "experience",
            "education",
            "skills",
            "work",
            "employment",
            "job",
            "position",
            "role",
            "company",
            "university",
            "college",
            "degree",
        ]

        text_lower = v.lower()
        found_indicators = sum(
            1 for indicator in resume_indicators if indicator in text_lower
        )

        if found_indicators < 2:
            raise ValueError(
                "Resume text should contain typical resume content (experience, education, skills, etc.)"
            )

        return v

    @field_validator("job_description")
    @classmethod
    def validate_job_description(cls, v):
        v = re.sub(r"\s+", " ", v.strip())
        words = v.split()
        if len(words) < 10:
            raise ValueError("Job description must contain at least 10 words")
        return v

    @field_validator("job_title")
    @classmethod
    def validate_job_title(cls, v):
        v = v.strip()
        v = re.sub(r"^(job|position|role):\s*", "", v, flags=re.IGNORECASE)
        v = re.sub(r"\s*(job|position|role)$", "", v, flags=re.IGNORECASE)
        if not v:
            raise ValueError("Job title cannot be empty")
        return v


class JobAnalysisResponse(BaseModel):
    match_score: float = Field(..., ge=0, le=100)
    ats_compatibility_score: float = Field(..., ge=0, le=100)
    keywords_present: List[str] = Field(default=[])
    keywords_missing: List[str] = Field(default=[])
    recommendations: List[str] = Field(default=[])
    strengths: List[str] = Field(default=[])
    improvement_areas: List[str] = Field(default=[])
    should_apply: bool = Field(...)
    application_tips: List[str] = Field(default=[])
    analysis_method: Optional[str] = Field(None)
    market_insights: Union[List[str], str, None] = Field(default=[])

    @field_validator("market_insights")
    @classmethod
    def normalize_market_insights(cls, v):
        if v is None:
            return []
        elif isinstance(v, str):
            return [v] if v.strip() else []
        elif isinstance(v, list):
            return [str(item) for item in v if item]
        else:
            return [str(v)] if v else []


class AnalysisMethodsResponse(BaseModel):
    available_methods: List[str] = Field(
        ..., description="List of available analysis methods"
    )
    descriptions: Dict[str, str] = Field(
        ..., description="Descriptions of each analysis method"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "available_methods": ["crewai", "claude", "both"],
                "descriptions": {
                    "crewai": "CrewAI Agents - Multi-agent system with specialized roles",
                    "claude": "Claude AI - Advanced reasoning and natural language understanding",
                    "both": "Combined Analysis - Insights from both systems with comparison",
                },
            }
        }


# Optional analytics schema (if you decide to add lightweight logging)
class AnalysisMetrics(BaseModel):
    """Lightweight analytics without storing personal data"""

    analysis_method: str
    job_title_category: str  # Categorized, not exact title
    match_score_range: str  # e.g., "70-80", not exact score
    processing_time_ms: int
    success: bool
    error_type: Optional[str] = None
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_method": "claude",
                "job_title_category": "software_engineer",
                "match_score_range": "70-80",
                "processing_time_ms": 3500,
                "success": True,
                "error_type": None,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


# Keep existing schemas for resume optimization if needed
class ResumeOptimizationRequest(BaseModel):
    current_resume_text: str = Field(
        ...,
        min_length=50,
        max_length=50000,
        description="Current resume text to optimize",
    )
    target_job_description: str = Field(
        ...,
        min_length=20,
        max_length=20000,
        description="Target job description to optimize for",
    )
    target_industry: str = Field(
        "general", max_length=100, description="Target industry (default: general)"
    )


class ResumeOptimizationResponse(BaseModel):
    improved_summary: str = Field(..., description="Improved professional summary")
    improved_experience: str = Field(
        ..., description="Improved experience descriptions"
    )
    added_keywords: List[str] = Field(
        default=[], description="Keywords that should be added"
    )
    ats_improvements: List[str] = Field(
        default=[], description="ATS optimization suggestions"
    )
    content_suggestions: List[str] = Field(
        default=[], description="Content improvement suggestions"
    )


# Error response schemas
class ValidationErrorResponse(BaseModel):
    detail: str
    field: Optional[str] = None
    input_value: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Resume text must contain at least 20 words of meaningful content",
                "field": "resume_text",
                "input_value": "Short text",
            }
        }


class ServiceErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    retry_after: Optional[int] = None  # Seconds to wait before retry

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Analysis service temporarily unavailable",
                "error_code": "SERVICE_UNAVAILABLE",
                "retry_after": 300,
            }
        }
