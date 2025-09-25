# app/schemas.py
"""
Pydantic schemas for privacy-first CV platform.
Updated to accept incomplete/draft data gracefully.
"""

import re
from pydantic import BaseModel, EmailStr, field_validator, Field
from typing import Any, Dict, Literal, Optional, List, Union
from datetime import date, datetime
from enum import Enum


# ================== CORE CV COMPONENT SCHEMAS ==================


class PersonalInfoBase(BaseModel):
    full_name: str = ""
    title: Optional[str] = None
    email: str = ""
    mobile: str = ""
    city: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    driving_license: Optional[str] = None
    nationality: Optional[str] = None
    place_of_birth: Optional[str] = None
    date_of_birth: Optional[date] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None
    summary: Optional[str] = None

    @field_validator("date_of_birth", mode="before")
    def parse_date_of_birth(cls, v):
        if v == "" or v is None:
            return None
        return v

    @field_validator("email", mode="before")
    def validate_email(cls, v):
        if not v or v == "":
            return ""
        return v

    @field_validator("mobile", mode="before")
    def validate_mobile(cls, v):
        if not v or v == "" or (isinstance(v, str) and len(v.strip()) < 5):
            return "+0000000000"
        return v

    @field_validator(
        "title",
        "city",
        "address",
        "postal_code",
        "driving_license",
        "nationality",
        "place_of_birth",
        "linkedin",
        "website",
        "summary",
        mode="before",
    )
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class EducationBase(BaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    location: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    current: Optional[bool] = False
    gpa: Optional[str] = None
    description: Optional[str] = None

    @field_validator("start_date", "end_date", mode="before")
    def parse_dates(cls, v):
        if not v or v == "":
            return None

        if isinstance(v, str):
            # Handle YYYY-MM format
            if re.match(r"^\d{4}-\d{2}$", v):
                return f"{v}-01"
            # Handle YYYY format
            elif re.match(r"^\d{4}$", v):
                return f"{v}-01-01"

        return v

    @field_validator("location", "gpa", "description", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class ExperienceBase(BaseModel):
    company: str = ""
    position: str = ""
    location: Optional[str] = None
    city: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    current: Optional[bool] = False
    description: Optional[str] = None

    @field_validator("start_date", "end_date", mode="before")
    def parse_dates(cls, v):
        if not v or v == "":
            return None

        if isinstance(v, str):
            if re.match(r"^\d{4}-\d{2}$", v):
                return f"{v}-01"
            elif re.match(r"^\d{4}$", v):
                return f"{v}-01-01"

        return v

    @field_validator("location", "city", "description", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class SkillBase(BaseModel):
    name: str = ""
    level: Optional[str] = None

    @field_validator("level", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class LanguageBase(BaseModel):
    language: str = ""
    proficiency: Optional[str] = None

    @field_validator("proficiency", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class ReferralBase(BaseModel):
    name: str = ""
    relation: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

    @field_validator("relation", "phone", "email", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class CustomSectionBase(BaseModel):
    title: str = ""
    content: Optional[str] = None

    @field_validator("content", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class ExtracurricularActivityBase(BaseModel):
    name: str = ""
    description: Optional[str] = None

    @field_validator("description", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class HobbyBase(BaseModel):
    name: str = ""


class CourseBase(BaseModel):
    name: str = ""
    institution: Optional[str] = None
    completion_date: Optional[str] = None
    description: Optional[str] = None

    @field_validator("institution", "completion_date", "description", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class InternshipBase(BaseModel):
    company: str = ""
    position: str = ""
    location: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    current: Optional[bool] = False
    description: Optional[str] = None

    @field_validator("start_date", "end_date", mode="before")
    def parse_dates(cls, v):
        if not v or v == "":
            return None

        if isinstance(v, str):
            if re.match(r"^\d{4}-\d{2}$", v):
                return f"{v}-01"
            elif re.match(r"^\d{4}$", v):
                return f"{v}-01-01"

        return v

    @field_validator("location", "description", mode="before")
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v


class PhotoBase(BaseModel):
    photolink: Optional[str] = None

    @field_validator("photolink", mode="before")
    def validate_photo_link(cls, v):
        if v is None or v == "":
            return None

        # Allow Base64 images
        if v.startswith("data:image/"):
            if not re.match(r"^data:image/(jpeg|jpg|png|gif|webp);base64,", v):
                raise ValueError("Invalid Base64 image format")

            try:
                base64_part = v.split(",")[1]
                if len(base64_part) < 100:
                    raise ValueError("Base64 image data too small")

                # 2MB limit
                estimated_size = len(base64_part) * 0.75
                if estimated_size > 2 * 1024 * 1024:
                    raise ValueError("Image too large (max 2MB)")

            except (IndexError, ValueError):
                raise ValueError("Invalid Base64 image data")

            return v

        # Allow regular URLs
        if v.startswith(("http://", "https://")):
            if len(v) > 500:
                raise ValueError("Photo URL too long")
            return v

        raise ValueError("Photo must be a valid Base64 image or URL")


class CustomizationBase(BaseModel):
    template: str = "stockholm"
    accent_color: str = "#1a5276"
    font_family: str = "Helvetica, Arial, sans-serif"
    line_spacing: float = 1.5
    headings_uppercase: bool = False
    hide_skill_level: bool = False
    language: str = "en"

    class Config:
        from_attributes = True


# ================== COMPLETE CV STRUCTURE ==================


class CompleteCV(BaseModel):
    """Complete CV structure - accepts incomplete/draft data"""

    title: str = "My Resume"
    is_public: bool = False
    customization: CustomizationBase = CustomizationBase()
    personal_info: Optional[PersonalInfoBase] = None

    educations: List[EducationBase] = []
    experiences: List[ExperienceBase] = []
    skills: List[SkillBase] = []
    languages: List[LanguageBase] = []
    referrals: List[ReferralBase] = []
    custom_sections: List[CustomSectionBase] = []
    extracurriculars: List[ExtracurricularActivityBase] = []
    hobbies: List[HobbyBase] = []
    courses: List[CourseBase] = []
    internships: List[InternshipBase] = []
    photo: Optional[PhotoBase] = None


# ================== RESPONSE SCHEMAS ==================


class PersonalInfoResponse(PersonalInfoBase):
    id: str

    class Config:
        from_attributes = True


class EducationResponse(EducationBase):
    id: str

    class Config:
        from_attributes = True


class ExperienceResponse(ExperienceBase):
    id: str

    class Config:
        from_attributes = True


class SkillResponse(SkillBase):
    id: str

    class Config:
        from_attributes = True


class LanguageResponse(LanguageBase):
    id: str

    class Config:
        from_attributes = True


class ReferralResponse(ReferralBase):
    id: str

    class Config:
        from_attributes = True


class CustomSectionResponse(CustomSectionBase):
    id: str

    class Config:
        from_attributes = True


class ExtracurricularActivityResponse(ExtracurricularActivityBase):
    id: str

    class Config:
        from_attributes = True


class HobbyResponse(HobbyBase):
    id: str

    class Config:
        from_attributes = True


class CourseResponse(CourseBase):
    id: str

    class Config:
        from_attributes = True


class InternshipResponse(InternshipBase):
    id: str

    class Config:
        from_attributes = True


class PhotoResponse(PhotoBase):
    id: str

    class Config:
        from_attributes = True


class CustomizationResponse(CustomizationBase):
    id: str

    class Config:
        from_attributes = True


class ResumeResponse(BaseModel):
    """Frontend-compatible resume response"""

    id: str
    title: str
    is_public: bool = False
    customization: Optional[CustomizationResponse] = None
    personal_info: Optional[PersonalInfoResponse] = None
    educations: List[EducationResponse] = []
    experiences: List[ExperienceResponse] = []
    skills: List[SkillResponse] = []
    languages: List[LanguageResponse] = []
    referrals: List[ReferralResponse] = []
    custom_sections: List[CustomSectionResponse] = []
    extracurriculars: List[ExtracurricularActivityResponse] = []
    hobbies: List[HobbyResponse] = []
    courses: List[CourseResponse] = []
    internships: List[InternshipResponse] = []
    photos: Optional[PhotoResponse] = None

    class Config:
        from_attributes = True


# ================== CLOUD & SESSION SCHEMAS ==================


class CloudProvider(str, Enum):
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    DROPBOX = "dropbox"
    BOX = "box"


class CloudFileMetadata(BaseModel):
    file_id: str
    name: str
    provider: CloudProvider
    last_modified: datetime
    size_bytes: Optional[int] = None
    created_at: datetime


class CloudSession(BaseModel):
    session_id: str
    connected_providers: List[CloudProvider]
    expires_at: datetime
    created_at: datetime


class CloudAuthRequest(BaseModel):
    provider: CloudProvider
    redirect_uri: str


class CloudAuthResponse(BaseModel):
    auth_url: str
    state: str


class CloudConnectionStatus(BaseModel):
    provider: CloudProvider
    connected: bool
    email: Optional[str] = None
    storage_quota: Optional[Dict[str, Any]] = None


# ================== AI ENHANCEMENT SCHEMAS ==================


class PersonalInfoSummaryRequest(BaseModel):
    summary: str


class ExperienceDescriptionRequest(BaseModel):
    description: str


class SkillRequest(BaseModel):
    name: str = ""
    level: Optional[str] = None

    @field_validator("name")
    def process_skill_name(cls, v):
        v = " ".join(v.split())
        return v


class SkillsUpdateRequest(BaseModel):
    skills: Union[List[SkillRequest], SkillRequest]

    @field_validator("skills")
    def ensure_list(cls, v):
        if isinstance(v, SkillRequest):
            return [v]
        return v


class AIUsageResponse(BaseModel):
    used_today: int
    limit: int
    remaining: int

    class Config:
        from_attributes = True


# ================== COVER LETTER SCHEMAS ==================


class CoverLetterBase(BaseModel):
    title: str = ""
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    job_description: Optional[str] = None
    recipient_name: Optional[str] = None
    recipient_title: Optional[str] = None
    cover_letter_content: str = ""
    is_favorite: Optional[bool] = False


class CoverLetterGenerationRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    recipient_name: Optional[str] = None
    recipient_title: Optional[str] = None
    cv_file_id: str


class CoverLetterResponse(CoverLetterBase):
    id: str
    cv_file_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ================== JOB MATCHING SCHEMAS ==================


class JobAnalysisRequest(BaseModel):
    cv_file_id: str
    job_description: str
    job_title: str
    company_name: str = ""


class JobAnalysisResponse(BaseModel):
    match_score: float
    ats_compatibility_score: float
    keywords_present: List[str] = []
    keywords_missing: List[str] = []
    recommendations: List[str] = []
    strengths: List[str] = []
    improvement_areas: List[str] = []
    should_apply: bool
    application_tips: List[str] = []


# ================== SUBSCRIPTION & PRICING ==================


class PricingTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    BUSINESS = "business"


class SubscriptionFeatures(BaseModel):
    ai_operations_daily: int
    cloud_providers: List[CloudProvider]
    advanced_templates: bool
    priority_support: bool
    api_access: bool
    bulk_operations: bool


class SubscriptionStatus(BaseModel):
    tier: PricingTier
    features: SubscriptionFeatures
    usage_today: Dict[str, int]
    expires_at: Optional[datetime] = None


# ================== TEMPORARY SHARING ==================


class TemporaryShareRequest(BaseModel):
    cv_file_id: str
    max_views: int = 50
    expires_hours: int = 24
    password: Optional[str] = None


class TemporaryShareResponse(BaseModel):
    share_id: str
    share_url: str
    qr_code_url: str
    expires_at: datetime
    max_views: int
    view_count: int


# ================== ERROR RESPONSES ==================


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    detail: str
    field_errors: Dict[str, List[str]]


# ================== CLOUD STORAGE METADATA ==================


class CVFileMetadata(BaseModel):
    version: str = "1.0"
    created_at: datetime
    last_modified: datetime
    created_with: str = "cv-privacy-platform"
    analytics_consent: bool = False
