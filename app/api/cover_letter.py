# app/api/cover_letter.py - FIXED VERSION
"""
Fixed AI-powered cover letter generation API endpoints that work with Google Drive CVs
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Form, File, UploadFile
from datetime import datetime
from pydantic import BaseModel

# FIXED: Use modern OpenAI client
from openai import AsyncOpenAI

from app.cloud.google_drive import GoogleDriveProvider

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
    """Fixed AI service for cover letter generation with Google Drive integration"""

    def __init__(self):
        # FIXED: Use modern OpenAI client
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("✅ OpenAI client initialized successfully")
        else:
            self.openai_client = None
            logger.warning("⚠️ No OpenAI API key found")

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


@router.post("/save")
async def save_cover_letter_to_drive(
    cover_letter_data: dict,
    session: dict = Depends(get_current_session),
):
    """
    Save a generated cover letter to Google Drive
    """
    try:
        logger.info(
            f"💾 Saving cover letter to Google Drive for session: {session.get('session_id')}"
        )

        # Get Google Drive tokens from session
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            raise HTTPException(
                status_code=403,
                detail="No Google Drive connection found. Please connect Google Drive first.",
            )

        logger.info(
            f"📄 Cover letter title: {cover_letter_data.get('title', 'Untitled')}"
        )

        # Prepare cover letter data for storage
        cover_letter_storage_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "type": "cover_letter",
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

        # Save to Google Drive using the service
        content = json.dumps(cover_letter_storage_data, indent=2, default=str)

        access_token = google_drive_tokens["access_token"]

        # Use Google Drive service to save
        async with GoogleDriveProvider(access_token) as provider:
            file_id = await provider.upload_file(
                filename, content, folder_name="Cover_Letters"
            )

        logger.info(f"✅ Cover letter saved to Google Drive: {file_id}")

        # Record activity
        try:
            await record_session_activity(
                session["session_id"],
                "cover_letter_saved",
                {
                    "file_id": file_id,
                    "title": cover_letter_data.get("title"),
                    "company": cover_letter_data.get("company_name"),
                },
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to record activity (non-critical): {e}")

        # Return success response
        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "message": f"Cover letter saved to Google Drive successfully",
            "cover_letter_data": {
                **cover_letter_storage_data["cover_letter_data"],
                "id": file_id,
            },
        }

    except GoogleDriveError as e:
        logger.error(f"❌ Google Drive error: {e}")
        raise HTTPException(status_code=502, detail=f"Google Drive error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error saving cover letter: {str(e)}")
        import traceback

        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save cover letter: {str(e)}"
        )


@router.post("/generate")
async def generate_cover_letter(
    # Handle both multipart form data and JSON
    job_description: str = Form(...),
    job_title: str = Form(...),
    company_name: str = Form(...),
    recipient_name: str = Form(""),
    recipient_title: str = Form(""),
    resume_id: Optional[str] = Form(None),
    save_to_database: bool = Form(False),
    title: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None),
    session: dict = Depends(get_current_session),
):
    """
    Generate cover letter from Google Drive CV or uploaded file
    FIXED: Now properly loads CV from Google Drive using session authentication
    """
    try:
        logger.info(
            f"🔄 Starting cover letter generation for session: {session.get('session_id')}"
        )
        logger.info(f"📋 Job: {job_title} at {company_name}")
        logger.info(f"📄 Resume ID: {resume_id}")
        logger.info(f"📁 Has uploaded file: {resume_file is not None}")

        # FIXED: Get Google Drive tokens from session (not cloud_tokens)
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        logger.info(f"🔍 Session has Google Drive tokens: {bool(google_drive_tokens)}")
        logger.info(f"🔍 Available cloud providers: {list(cloud_tokens.keys())}")

        if not google_drive_tokens and not resume_file:
            raise HTTPException(
                status_code=403,
                detail="No Google Drive connection found and no resume file uploaded. Please connect Google Drive or upload a resume file.",
            )

        cv_data = None

        # Load CV data - prioritize resume_id over uploaded file
        if resume_id and google_drive_tokens:
            logger.info(f"📥 Loading CV from Google Drive: {resume_id}")
            try:
                # FIXED: Use the google_drive_service directly with proper tokens
                cv_data = await google_drive_service.load_cv(
                    google_drive_tokens, resume_id
                )
                logger.info(
                    f"✅ Successfully loaded CV from Google Drive: {cv_data.title}"
                )
            except GoogleDriveError as gd_error:
                logger.error(f"❌ Google Drive error: {gd_error}")
                if not resume_file:
                    raise HTTPException(
                        status_code=502, detail=f"Google Drive error: {str(gd_error)}"
                    )
            except Exception as e:
                logger.error(f"❌ Failed to load CV from Google Drive: {e}")
                if not resume_file:
                    raise HTTPException(
                        status_code=404, detail=f"Failed to load CV: {str(e)}"
                    )

        # Fallback to uploaded file if Google Drive loading failed or no resume_id
        if not cv_data and resume_file:
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

        if not cv_data:
            raise HTTPException(
                status_code=400,
                detail="No CV data available for cover letter generation",
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
            await record_session_activity(
                session["session_id"],
                "cover_letter_generated",
                {
                    "job_title": job_title,
                    "company": company_name,
                    "source": "google_drive" if resume_id else "upload",
                },
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


@router.get("/test")
async def test_cover_letter_endpoint():
    """Test endpoint to verify cover letter API is working"""
    return {
        "message": "Cover letter API is working",
        "timestamp": datetime.utcnow().isoformat(),
        "openai_configured": cover_letter_service.openai_client is not None,
    }


@router.get("/list-cover-letters")
async def list_cover_letters_from_google_drive(
    session: dict = Depends(get_current_session),
):
    """List all cover letters from Google Drive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        google_drive_tokens = cloud_tokens.get("google_drive")

        if not google_drive_tokens:
            raise HTTPException(
                status_code=403, detail="No Google Drive connection found"
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

        async with GoogleDriveProvider(access_token) as provider:
            # Look for cover letter files in Cover_Letters folder
            files = await provider.list_files(folder_name="Cover_Letters")

        # Filter for cover letter files
        cover_letter_files = []
        for file in files:
            if "cover_letter" in file.name.lower() and file.name.endswith(".json"):
                cover_letter_files.append(
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
                    }
                )

        logger.info(f"📋 Found {len(cover_letter_files)} cover letters")

        return {
            "success": True,
            "provider": "google_drive",
            "cover_letters": cover_letter_files,
            "count": len(cover_letter_files),
        }

    except Exception as e:
        logger.error(f"❌ List cover letters failed: {str(e)}")
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
