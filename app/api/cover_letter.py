# app/api/cover_letter.py - FIXED VERSION with Multi-Provider Support
"""
FIXED AI-powered cover letter generation API endpoints that work with any provider
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Form, File, UploadFile
from datetime import datetime
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# FIXED: Use modern OpenAI client
from openai import AsyncOpenAI

from app.cloud.google_drive import GoogleDriveProvider
from app.cloud.onedrive import OneDriveProvider

from ..config import settings
from ..schemas import CompleteCV, CloudProvider
from ..auth.sessions import (
    get_current_session,
    record_session_activity,
    session_manager,
)
from ..cloud.google_drive_service import google_drive_service, GoogleDriveError

logger = logging.getLogger(__name__)
router = APIRouter()

# Import OneDrive service for multi-provider support
try:
    from ..cloud.onedrive_service import onedrive_service, OneDriveError

    ONEDRIVE_AVAILABLE = True
    logger.info("✅ OneDrive service imported successfully")
except ImportError:
    ONEDRIVE_AVAILABLE = False
    logger.warning("⚠️ OneDrive service not available")


class CoverLetterRequest(BaseModel):
    job_description: str
    job_title: str
    company_name: str
    recipient_name: Optional[str] = None
    recipient_title: Optional[str] = None
    resume_id: Optional[str] = None  # Google Drive file ID
    save_to_database: bool = False
    title: Optional[str] = None


class CoverLetterService:
    """Fixed AI service for cover letter generation with multi-provider support"""

    def __init__(self):
        # FIXED: Use modern OpenAI client
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("✅ OpenAI client initialized successfully")
        else:
            self.openai_client = None
            logger.warning("⚠️ No OpenAI API key found")

    def _convert_cv_dict_to_complete_cv(self, cv_dict: Dict[str, Any]) -> CompleteCV:
        """Convert CV dictionary to CompleteCV object"""
        try:
            # Handle different photo field formats
            photo_data = cv_dict.get("photo", cv_dict.get("photos", {}))
            if not isinstance(photo_data, dict):
                photo_data = {"photolink": None}

            # Ensure all required fields exist with defaults
            cv_data = {
                "title": cv_dict.get("title", "My Resume"),
                "is_public": cv_dict.get("is_public", False),
                "customization": cv_dict.get(
                    "customization",
                    {
                        "template": "stockholm",
                        "accent_color": "#1a5276",
                        "font_family": "Helvetica, Arial, sans-serif",
                        "line_spacing": 1.5,
                        "headings_uppercase": False,
                        "hide_skill_level": False,
                        "language": "en",
                    },
                ),
                "personal_info": cv_dict.get("personal_info", {}),
                "experiences": cv_dict.get("experiences", []),
                "educations": cv_dict.get("educations", []),
                "skills": cv_dict.get("skills", []),
                "languages": cv_dict.get("languages", []),
                "referrals": cv_dict.get("referrals", []),
                "custom_sections": cv_dict.get("custom_sections", []),
                "extracurriculars": cv_dict.get("extracurriculars", []),
                "hobbies": cv_dict.get("hobbies", []),
                "courses": cv_dict.get("courses", []),
                "internships": cv_dict.get("internships", []),
                "photo": photo_data,
            }

            return CompleteCV.parse_obj(cv_data)
        except Exception as e:
            logger.error(f"❌ Failed to convert CV dict to CompleteCV: {e}")
            raise ValueError(f"Invalid CV data format: {e}")

    async def generate_cover_letter_from_cv(
        self,
        cv_data: CompleteCV,
        job_description: str,
        job_title: str,
        company_name: str,
        recipient_name: str = "",
        recipient_title: str = "",
    ) -> Dict[str, Any]:
        """Generate a cover letter based on CV data and job requirements"""

        if not self.openai_client:
            raise HTTPException(status_code=503, detail="AI service not configured")

        try:
            # Extract relevant information from CV
            personal_info = cv_data.personal_info
            experiences = cv_data.experiences[:3]  # Limit to recent experiences
            skills = [skill.name for skill in cv_data.skills[:10]]  # Limit skills
            educations = cv_data.educations[:2]  # Limit education entries

            # Build comprehensive CV summary
            applicant_name = personal_info.full_name if personal_info else "Applicant"
            applicant_title = personal_info.title if personal_info else ""
            applicant_summary = personal_info.summary if personal_info else ""

            # Format experiences for the prompt
            experience_text = ""
            for exp in experiences:
                experience_text += f"- {exp.position} at {exp.company}"
                if exp.description:
                    experience_text += f": {exp.description[:200]}"
                experience_text += "\n"

            # Format education
            education_text = ""
            for edu in educations:
                education_text += (
                    f"- {edu.degree} in {edu.field_of_study} from {edu.institution}\n"
                )

            # Format recipient information
            greeting = "Dear Hiring Manager,"
            if recipient_name and recipient_title:
                greeting = f"Dear {recipient_name}, {recipient_title},"
            elif recipient_name:
                greeting = f"Dear {recipient_name},"

            # Create detailed prompt for cover letter generation
            prompt = f"""
            Generate a professional cover letter in JSON format for the following job application:

            JOB DETAILS:
            Position: {job_title}
            Company: {company_name}
            Job Description: {job_description[:1500]}  # Truncate to avoid token limits

            APPLICANT PROFILE:
            Name: {applicant_name}
            Current Title: {applicant_title}
            Professional Summary: {applicant_summary}
            
            Recent Work Experience:
            {experience_text}
            
            Key Skills: {", ".join(skills)}
            
            Education:
            {education_text}

            REQUIREMENTS:
            1. Create a compelling 3-4 paragraph cover letter
            2. Highlight relevant experience and skills from the CV that match the job requirements
            3. Show enthusiasm for the specific role and company
            4. Demonstrate value proposition and cultural fit
            5. Use professional but engaging tone
            6. Make it ATS-friendly with relevant keywords from job description
            7. Keep it concise but impactful (under 400 words total)

            Return the response in this EXACT JSON format:
            {{
                "greeting": "{greeting}",
                "introduction": "Opening paragraph introducing yourself and the position",
                "body_paragraphs": [
                    "Second paragraph highlighting relevant experience and achievements",
                    "Third paragraph showing knowledge of company and cultural fit"
                ],
                "closing": "Final paragraph with call to action and enthusiasm",
                "signature": "Sincerely,\\n{applicant_name}"
            }}
            """

            # Generate cover letter using OpenAI
            logger.info(
                f"🤖 Generating cover letter for {applicant_name} applying to {job_title} at {company_name}"
            )

            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use reliable model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert career coach and cover letter writer. Generate professional, personalized cover letters that help candidates stand out. Always respond with valid JSON in the exact format requested.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.7,
            )

            content = response.choices[0].message.content.strip()
            logger.info(
                f"✅ Generated cover letter content ({len(content)} characters)"
            )

            # Parse the JSON response
            try:
                cover_letter_data = json.loads(content)
                logger.info("✅ Successfully parsed JSON response from OpenAI")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse JSON from OpenAI: {e}")
                logger.error(f"Raw response: {content[:200]}...")
                # Fallback: create structured response
                cover_letter_data = {
                    "greeting": greeting,
                    "introduction": f"I am writing to express my strong interest in the {job_title} position at {company_name}.",
                    "body_paragraphs": [
                        f"With my experience in {experiences[0].position if experiences else 'relevant fields'}, I am well-positioned to contribute to your team.",
                        f"I am particularly drawn to {company_name} and excited about the opportunity to apply my skills in {', '.join(skills[:3])}.",
                    ],
                    "closing": "I would welcome the opportunity to discuss how my background and enthusiasm can contribute to your team's success.",
                    "signature": f"Sincerely,\n{applicant_name}",
                }

            return {
                "success": True,
                "cover_letter": cover_letter_data,
                "applicant_info": {
                    "name": applicant_name,
                    "email": personal_info.email if personal_info else "",
                    "phone": personal_info.mobile if personal_info else "",
                },
                "job_info": {
                    "title": job_title,
                    "company": company_name,
                    "recipient_name": recipient_name,
                    "recipient_title": recipient_title,
                },
            }

        except Exception as e:
            logger.error(f"❌ Cover letter generation failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Cover letter generation failed: {str(e)}"
            )


# Global service instance
cover_letter_service = CoverLetterService()


@router.post("/generate")
async def generate_cover_letter(
    # Handle both multipart form data and JSON
    job_description: str = Form(...),
    job_title: str = Form(...),
    company_name: str = Form(...),
    recipient_name: str = Form(""),
    recipient_title: str = Form(""),
    resume_id: Optional[str] = Form(None),
    resume_data: Optional[str] = Form(None),  # NEW: Accept CV data directly
    provider: Optional[str] = Form(None),
    data_source: Optional[str] = Form(None),  # NEW: Track data source
    skip_cloud_loading: Optional[str] = Form(None),  # NEW: Skip cloud loading flag
    save_to_database: bool = Form(False),
    title: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None),
    session: dict = Depends(get_current_session),
):
    """
    FIXED: Generate cover letter with multi-provider support
    Priority: resume_data > resume_file > cloud loading
    """
    try:
        logger.info(
            f"🔄 Starting cover letter generation for session: {session.get('session_id')}"
        )
        logger.info(f"📋 Job: {job_title} at {company_name}")
        logger.info(f"📄 Resume ID: {resume_id}")
        logger.info(f"📁 Has uploaded file: {resume_file is not None}")
        logger.info(f"📊 Has resume data: {resume_data is not None}")
        logger.info(f"🏷️ Data source: {data_source}")
        logger.info(f"⏭️ Skip cloud loading: {skip_cloud_loading}")

        cloud_tokens = session.get("cloud_tokens", {})
        available_providers = list(cloud_tokens.keys())
        logger.info(f"🔍 Available cloud providers: {available_providers}")

        cv_data = None

        # PRIORITY 1: Use provided resume_data (from frontend)
        if resume_data:
            logger.info("📥 Using provided resume data from frontend")
            try:
                cv_dict = json.loads(resume_data)
                cv_data = cover_letter_service._convert_cv_dict_to_complete_cv(cv_dict)
                logger.info(f"✅ Successfully parsed provided CV data: {cv_data.title}")
            except Exception as e:
                logger.error(f"❌ Failed to parse provided resume data: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid resume data format: {str(e)}"
                )

        # PRIORITY 2: Use uploaded file if no resume_data
        elif resume_file:
            logger.info(f"📤 Processing uploaded resume file: {resume_file.filename}")
            try:
                # Read uploaded file content
                file_content = await resume_file.read()
                file_text = file_content.decode("utf-8")

                # For demo purposes, create a basic CV structure from uploaded text
                # In a real implementation, you'd parse the uploaded CV more thoroughly
                cv_data = CompleteCV(
                    title=f"Resume - {resume_file.filename}",
                    customization={
                        "template": "stockholm",
                        "accent_color": "#1a5276",
                        "font_family": "Helvetica, Arial, sans-serif",
                        "line_spacing": 1.5,
                        "headings_uppercase": False,
                        "hide_skill_level": False,
                        "language": "en",
                    },
                    personal_info={
                        "full_name": "Resume Holder",
                        "email": "user@example.com",
                        "mobile": "+1234567890",
                        "summary": file_text[:500],  # Use first part as summary
                    },
                    experiences=[],
                    skills=[],
                    educations=[],
                )
                logger.info("✅ Created basic CV structure from uploaded file")
            except Exception as e:
                logger.error(f"❌ Failed to process uploaded file: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Failed to process uploaded file: {str(e)}"
                )

        # PRIORITY 3: Load from cloud providers (only if no data provided and not skipped)
        elif resume_id and skip_cloud_loading != "true":
            logger.info(f"📥 Attempting to load CV from cloud providers: {resume_id}")

            # Determine which provider to use based on file ID format or explicit provider
            target_provider = provider

            if not target_provider:
                # Auto-detect provider based on file ID format
                if "!" in resume_id:  # OneDrive format
                    target_provider = "onedrive"
                else:  # Assume Google Drive
                    target_provider = "google_drive"

                logger.info(f"🔍 Auto-detected provider: {target_provider}")

            # Try to load from the determined provider
            if target_provider == "google_drive" and "google_drive" in cloud_tokens:
                logger.info(f"📥 Loading CV from Google Drive: {resume_id}")
                try:
                    google_drive_tokens = cloud_tokens["google_drive"]
                    cv_data = await google_drive_service.load_cv(
                        google_drive_tokens, resume_id
                    )
                    logger.info(
                        f"✅ Successfully loaded CV from Google Drive: {cv_data.title}"
                    )
                except GoogleDriveError as gd_error:
                    logger.error(f"❌ Google Drive error: {gd_error}")
                    raise HTTPException(
                        status_code=502, detail=f"Google Drive error: {str(gd_error)}"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to load CV from Google Drive: {e}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to load CV from Google Drive: {str(e)}",
                    )

            elif (
                target_provider == "onedrive"
                and "onedrive" in cloud_tokens
                and ONEDRIVE_AVAILABLE
            ):
                logger.info(f"📥 Loading CV from OneDrive: {resume_id}")
                try:
                    onedrive_tokens = cloud_tokens["onedrive"]
                    cv_data = await onedrive_service.load_cv(onedrive_tokens, resume_id)
                    logger.info(
                        f"✅ Successfully loaded CV from OneDrive: {cv_data.title}"
                    )
                except OneDriveError as od_error:
                    logger.error(f"❌ OneDrive error: {od_error}")
                    raise HTTPException(
                        status_code=502, detail=f"OneDrive error: {str(od_error)}"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to load CV from OneDrive: {e}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Failed to load CV from OneDrive: {str(e)}",
                    )

            else:
                available_msg = f"Available providers: {available_providers}"
                if not ONEDRIVE_AVAILABLE:
                    available_msg += " (OneDrive service not available)"

                logger.error(
                    f"❌ No suitable provider found for resume_id: {resume_id}"
                )
                logger.error(f"   Target provider: {target_provider}")
                logger.error(f"   {available_msg}")

                raise HTTPException(
                    status_code=403,
                    detail=f"No {target_provider} connection found or provider not available. {available_msg}",
                )

        # Check if we have any CV data
        if not cv_data:
            logger.error("❌ No CV data available for cover letter generation")
            raise HTTPException(
                status_code=400,
                detail="No CV data available for cover letter generation. Please provide resume_data, upload a file, or connect a cloud provider.",
            )

        # Generate cover letter using AI
        logger.info(f"🤖 Generating cover letter with AI for: {cv_data.title}")
        cover_letter_result = await cover_letter_service.generate_cover_letter_from_cv(
            cv_data=cv_data,
            job_description=job_description,
            job_title=job_title,
            company_name=company_name,
            recipient_name=recipient_name,
            recipient_title=recipient_title,
        )

        logger.info(f"✅ Cover letter generated successfully")

        # Record activity
        try:
            activity_data = {
                "job_title": job_title,
                "company": company_name,
                "data_source": data_source or "unknown",
            }

            if resume_data:
                activity_data["source"] = "provided_data"
            elif resume_file:
                activity_data["source"] = "upload"
            elif resume_id:
                activity_data["source"] = (
                    f"cloud_{target_provider if 'target_provider' in locals() else 'unknown'}"
                )

            await record_session_activity(
                session["session_id"],
                "cover_letter_generated",
                activity_data,
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to record activity (non-critical): {e}")

        # Return the generated cover letter
        return {
            "success": True,
            "cover_letter": cover_letter_result["cover_letter"],
            "applicant_info": cover_letter_result.get("applicant_info", {}),
            "job_info": cover_letter_result.get("job_info", {}),
            "message": "Cover letter generated successfully",
            "job_title": job_title,
            "company_name": company_name,
            "generated_at": datetime.utcnow().isoformat(),
            "data_source": data_source or "unknown",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in cover letter generation: {str(e)}")
        import traceback

        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Cover letter generation failed: {str(e)}"
        )


# Keep all other endpoints unchanged...
@router.post("/save")
async def save_cover_letter_to_drive(
    cover_letter_data: dict,
    session: dict = Depends(get_current_session),
):
    """
    Save a generated cover letter to Google Drive or OneDrive based on user preference
    """
    try:
        logger.info(
            f"💾 Saving cover letter to cloud storage for session: {session.get('session_id')}"
        )

        # Get user's preferred cloud provider from session or data
        cloud_provider = cover_letter_data.get("cloud_provider", "google_drive")

        # Get cloud tokens from session
        cloud_tokens = session.get("cloud_tokens", {})

        if cloud_provider == "google_drive":
            provider_tokens = cloud_tokens.get("google_drive")
            provider_name = "Google Drive"
        elif cloud_provider == "onedrive":
            provider_tokens = cloud_tokens.get("onedrive")
            provider_name = "OneDrive"
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid cloud provider. Supported providers: google_drive, onedrive",
            )

        if not provider_tokens:
            raise HTTPException(
                status_code=403,
                detail=f"No {provider_name} connection found. Please connect {provider_name} first.",
            )

        logger.info(
            f"📄 Cover letter title: {cover_letter_data.get('title', 'Untitled')} to {provider_name}"
        )

        # Prepare cover letter data for storage
        cover_letter_storage_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "type": "cover_letter",
                "cloud_provider": cloud_provider,
            },
            "cover_letter_data": {
                "title": cover_letter_data.get("title", "Untitled Cover Letter"),
                "company_name": cover_letter_data.get("company_name", ""),
                "job_title": cover_letter_data.get("job_title", ""),
                "job_description": cover_letter_data.get("job_description", ""),
                "recipient_name": cover_letter_data.get("recipient_name", ""),
                "recipient_title": cover_letter_data.get("recipient_title", ""),
                "cover_letter_content": cover_letter_data.get(
                    "cover_letter_content", {}
                ),
                "applicant_info": cover_letter_data.get("applicant_info", {}),
                "job_info": cover_letter_data.get("job_info", {}),
                "is_favorite": cover_letter_data.get("is_favorite", False),
                "resume_id": cover_letter_data.get("resume_id"),
                "created_at": cover_letter_data.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
                "updated_at": datetime.utcnow().isoformat(),
            },
        }

        # Generate filename for the cover letter
        safe_title = "".join(
            c
            for c in cover_letter_data.get("title", "cover_letter")
            if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:50]
        filename = f"cover_letter_{safe_title}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        # Save to cloud storage
        content = json.dumps(cover_letter_storage_data, indent=2, default=str)

        file_id = None
        if cloud_provider == "google_drive":
            access_token = provider_tokens["access_token"]
            async with GoogleDriveProvider(access_token) as provider:
                file_id = await provider.upload_file(
                    filename, content, folder_name="Cover_Letters"
                )

        elif cloud_provider == "onedrive":
            access_token = provider_tokens["access_token"]
            async with OneDriveProvider(access_token) as provider:
                file_id = await provider.upload_file(
                    filename, content, folder_name="Cover_Letters"
                )

        logger.info(f"✅ Cover letter saved to {provider_name}: {file_id}")

        # Record activity
        try:
            await record_session_activity(
                session["session_id"],
                "cover_letter_saved",
                {
                    "file_id": file_id,
                    "title": cover_letter_data.get("title"),
                    "company": cover_letter_data.get("company_name"),
                    "cloud_provider": cloud_provider,
                },
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to record activity (non-critical): {e}")

        # Return success response
        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "cloud_provider": cloud_provider,
            "message": f"Cover letter saved to {provider_name} successfully",
            "cover_letter_data": {
                **cover_letter_storage_data["cover_letter_data"],
                "id": file_id,
            },
        }

    except (GoogleDriveError, OneDriveError) as e:
        logger.error(f"❌ Cloud storage error: {e}")
        raise HTTPException(status_code=502, detail=f"Cloud storage error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error saving cover letter: {str(e)}")
        import traceback

        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save cover letter: {str(e)}"
        )


@router.get("/test")
async def test_cover_letter_endpoint():
    """Test endpoint to verify cover letter API is working"""
    return {
        "message": "Cover letter API is working",
        "timestamp": datetime.utcnow().isoformat(),
        "openai_configured": cover_letter_service.openai_client is not None,
        "onedrive_available": ONEDRIVE_AVAILABLE,
    }


@router.get("/list-cover-letters")
async def list_cover_letters_provider(
    provider: str = Query(..., description="Cloud provider (google_drive or onedrive)"),
    session: dict = Depends(get_current_session),
):
    """
    Temporary endpoint for frontend compatibility
    """
    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if provider not in cloud_tokens:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Provider '{provider}' not connected",
                    "cover_letters": [],
                },
            )

        # For now, return empty array with success message
        return {
            "success": True,
            "provider": provider,
            "cover_letters": [],
            "message": f"Provider '{provider}' connected but cover letters listing not fully implemented yet",
        }

    except Exception as e:
        logger.error(f"❌ List cover letters failed for {provider}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to list cover letters: {str(e)}",
                "cover_letters": [],
            },
        )


@router.get("/list")
async def list_cover_letters_multi_provider(
    provider: Optional[str] = Query(None),
    session: dict = Depends(get_current_session),
):
    """
    List cover letters from all connected providers or a specific provider
    """
    try:
        logger.info(
            f"📋 Listing cover letters from providers for session: {session.get('session_id')}"
        )

        cloud_tokens = session.get("cloud_tokens", {})
        available_providers = list(cloud_tokens.keys())

        if provider and provider not in available_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{provider}' not connected. Available: {available_providers}",
            )

        target_providers = [provider] if provider else available_providers
        all_cover_letters = []

        for target_provider in target_providers:
            try:
                if target_provider == "google_drive":
                    # Use Google Drive service
                    google_drive_tokens = cloud_tokens["google_drive"]
                    valid_tokens = await google_drive_service.ensure_valid_token(
                        google_drive_tokens
                    )

                    if valid_tokens != google_drive_tokens:
                        cloud_tokens["google_drive"] = valid_tokens
                        await session_manager.update_session_cloud_tokens(
                            session["session_id"], cloud_tokens
                        )

                    access_token = valid_tokens["access_token"]

                    async with GoogleDriveProvider(access_token) as provider:
                        files = await provider.list_files(folder_name="Cover_Letters")

                    # Process Google Drive files
                    for file in files:
                        if "cover_letter" in file.name.lower() and file.name.endswith(
                            ".json"
                        ):
                            all_cover_letters.append(
                                {
                                    "id": file.file_id,
                                    "title": file.name.replace(".json", "")
                                    .replace("cover_letter_", "")
                                    .replace("_", " ")
                                    .title(),
                                    "name": file.name,
                                    "created_at": file.created_at.isoformat(),
                                    "updated_at": file.last_modified.isoformat(),
                                    "size": file.size_bytes,
                                    "provider": "google_drive",
                                    "storageType": "google_drive",
                                }
                            )

                elif target_provider == "onedrive" and ONEDRIVE_AVAILABLE:
                    # Use OneDrive service
                    onedrive_tokens = cloud_tokens["onedrive"]
                    # Add OneDrive implementation here similar to Google Drive
                    # You'll need to implement the OneDrive equivalent
                    pass

            except Exception as provider_error:
                logger.warning(
                    f"⚠️ Failed to list cover letters from {target_provider}: {provider_error}"
                )
                continue

        logger.info(
            f"✅ Found {len(all_cover_letters)} cover letters from {len(target_providers)} providers"
        )

        return {
            "success": True,
            "cover_letters": all_cover_letters,
            "count": len(all_cover_letters),
            "providers": target_providers,
        }

    except Exception as e:
        logger.error(f"❌ Multi-provider cover letter listing failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list cover letters: {str(e)}"
        )


@router.put("/update/{file_id}")
async def update_cover_letter(
    file_id: str,
    cover_letter_data: dict,
    session: dict = Depends(get_current_session),
):
    """Update an existing cover letter in Google Drive"""
    try:
        logger.info(f"🔄 Updating cover letter: {file_id}")

        # Get Google Drive tokens from session
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            raise HTTPException(
                status_code=403,
                detail="No Google Drive connection found",
            )

        # Ensure token is valid
        valid_tokens = await google_drive_service.ensure_valid_token(
            google_drive_tokens
        )

        # Update session if tokens were refreshed
        if valid_tokens != google_drive_tokens:
            cloud_tokens["google_drive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        access_token = valid_tokens["access_token"]

        # Prepare updated cover letter data
        updated_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "type": "cover_letter",
            },
            "cover_letter_data": cover_letter_data,
        }

        # Convert to JSON
        content = json.dumps(updated_data, indent=2, default=str)

        # Update file in Google Drive
        async with GoogleDriveProvider(access_token) as provider:
            success = await provider.update_coverletter(file_id, content)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update cover letter")

        logger.info(f"✅ Cover letter updated successfully: {file_id}")

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cover_letter_updated",
            {
                "file_id": file_id,
                "title": cover_letter_data.get("title"),
                "company": cover_letter_data.get("company_name"),
            },
        )

        return {
            "success": True,
            "file_id": file_id,
            "message": "Cover letter updated successfully",
        }

    except Exception as e:
        logger.error(f"❌ Cover letter update failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update cover letter: {str(e)}"
        )
