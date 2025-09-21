# app/api/onedrive_api.py
"""
OneDrive API endpoints - Separated from Google Drive for better maintainability
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from datetime import datetime
import time
import secrets
import json
from pydantic import ValidationError

from app.cloud.onedrive import OneDriveProvider

from ..auth.sessions import (
    get_current_session,
    get_optional_session,
    session_manager,
    record_session_activity,
)
from ..cloud.onedrive_service import onedrive_service, OneDriveError
from ..schemas import CompleteCV, CloudConnectionStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/providers")
async def list_onedrive_info():
    """Get OneDrive provider information"""
    return {
        "providers": [
            {
                "id": "onedrive",
                "name": "Microsoft OneDrive",
                "description": "Store your CVs in OneDrive",
                "logo_url": "/static/logos/onedrive.png",
                "supported_features": ["read", "write", "delete", "folders"],
                "status": "available",
            }
        ]
    }


@router.post("/connect")
async def initiate_onedrive_connection(
    request: Request, session: dict = Depends(get_optional_session)
):
    """Initiate OneDrive OAuth connection"""
    try:
        # Create session if doesn't exist
        if not session:
            from ..auth.sessions import create_anonymous_session

            session_data = await create_anonymous_session(request)
            session_id = session_data["session_id"]
        else:
            session_id = session["session_id"]

        # Generate OAuth state parameter
        state = f"{session_id}:{secrets.token_urlsafe(16)}"

        # Get OAuth URL from OneDrive service
        auth_url = onedrive_service.get_oauth_url(state)

        logger.info(f"🔗 Generated OneDrive OAuth URL for session: {session_id}")

        return {"auth_url": auth_url, "state": state, "provider": "onedrive"}

    except Exception as e:
        logger.error(f"❌ OneDrive connection initiation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate OneDrive connection: {str(e)}",
        )


@router.get("/callback")
async def handle_onedrive_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Microsoft"),
    state: str = Query(..., description="State parameter from Microsoft"),
    scope: Optional[str] = Query(None, description="Granted scopes"),
    error: Optional[str] = Query(None, description="Error from Microsoft"),
    error_description: Optional[str] = Query(None, description="Error description"),
):
    """Handle OneDrive OAuth callback"""

    logger.info("🔗 OneDrive OAuth callback received")
    logger.info(f"🔗 Code: {code[:20] if code else 'None'}...")
    logger.info(f"🔗 State: {state}")
    logger.info(f"🔗 Error: {error}")

    # Check for OAuth errors
    if error:
        logger.error(f"❌ Microsoft OAuth error: {error}")
        error_msg = f"Microsoft authorization failed: {error}"
        if error_description:
            error_msg += f" - {error_description}"

        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/onedrive?error={error}&error_description={error_description or ''}"
        )

    if not code:
        logger.error("❌ No authorization code received")
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/onedrive?error=no_code"
        )

    if not state:
        logger.error("❌ No state parameter received")
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/onedrive?error=no_state"
        )

    try:
        # Extract session ID from state
        session_id = state.split(":")[0] if ":" in state else state
        logger.info(f"🔄 Processing callback for session: {session_id}")

        # Verify session exists
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            logger.error(f"❌ Session not found: {session_id}")
            return RedirectResponse(
                url="http://localhost:5173/cloud/callback/onedrive?error=invalid_session"
            )

        # Exchange code for tokens
        logger.info("🔄 Exchanging authorization code for tokens...")
        tokens = await onedrive_service.exchange_code_for_tokens(code)

        logger.info("✅ Successfully exchanged code for tokens")

        # Test the connection to get user info
        connection_test = await onedrive_service.test_connection(tokens)
        if not connection_test["success"]:
            raise OneDriveError(
                f"Connection test failed: {connection_test.get('error')}"
            )

        # Store tokens in session
        cloud_tokens = session_data.get("cloud_tokens", {})
        cloud_tokens["onedrive"] = {
            **tokens,
            "email": connection_test["user"]["email"],
            "name": connection_test["user"]["name"],
        }

        # Update session with tokens
        await session_manager.update_session_cloud_tokens(session_id, cloud_tokens)
        logger.info("✅ Successfully stored OneDrive tokens")

        # Record activity
        await record_session_activity(
            session_id,
            "onedrive_connected",
            {"email": connection_test["user"]["email"]},
        )

        # Redirect to frontend with success
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/onedrive?success=true&provider=onedrive"
        )

    except Exception as e:
        logger.error(f"❌ OneDrive callback processing failed: {str(e)}")
        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/onedrive?error=processing_failed&error_description={str(e)}"
        )


@router.get("/status")
async def get_onedrive_status(session: dict = Depends(get_current_session)):
    """Get OneDrive connection status"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            return {
                "provider": "onedrive",
                "connected": False,
                "email": None,
                "storage_quota": None,
            }

        # Check connection status using the service
        status = await onedrive_service.get_connection_status(onedrive_tokens)
        return status.dict()

    except Exception as e:
        logger.error(f"❌ OneDrive status check failed: {str(e)}")
        return {
            "provider": "onedrive",
            "connected": False,
            "email": None,
            "storage_quota": None,
            "error": str(e),
        }


@router.post("/test-save")
async def test_onedrive_save(session: dict = Depends(get_current_session)):
    """Test OneDrive save functionality with simple data"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            return {
                "success": False,
                "error": "No OneDrive connection found",
                "provider": "onedrive",
            }

        # Create simple test data that matches the expected schema
        test_data = {
            "title": "Test CV",
            "is_public": False,
            "customization": {
                "template": "stockholm",
                "accent_color": "#1a5276",
                "font_family": "Helvetica, Arial, sans-serif",
                "line_spacing": 1.5,
                "headings_uppercase": False,
                "hide_skill_level": False,
                "language": "en",
            },
            "personal_info": {
                "full_name": "Test User",
                "email": "test@example.com",
                "mobile": "+1234567890",
                "city": "",
                "address": "",
                "postal_code": "",
                "driving_license": "",
                "nationality": "",
                "place_of_birth": "",
                "date_of_birth": "",
                "linkedin": "",
                "website": "",
                "summary": "",
                "title": "",
            },
            "educations": [],
            "experiences": [],
            "skills": [],
            "languages": [],
            "referrals": [],
            "custom_sections": [],
            "extracurriculars": [],
            "hobbies": [],
            "courses": [],
            "internships": [],
            "photo": {"photolink": None},
        }

        # Save test data
        file_id = await onedrive_service.save_cv(onedrive_tokens, test_data)

        return {"success": True, "file_id": file_id, "message": "Test save successful"}

    except Exception as e:
        logger.error(f"❌ Test save failed: {str(e)}")
        return {"success": False, "error": str(e), "provider": "onedrive"}


@router.post("/save")
async def save_cv_to_onedrive(
    cv_data: dict,
    session: dict = Depends(get_current_session),
):
    """Save a CV to OneDrive"""

    start_time = time.time()

    try:
        logger.info("🐛 STEP 1: Starting save_cv_to_onedrive function")

        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        logger.info(
            f"🐛 STEP 2: Session data retrieved, has onedrive_tokens: {bool(onedrive_tokens)}"
        )

        if not onedrive_tokens:
            logger.error("❌ No OneDrive connection found")
            return {
                "success": False,
                "error": "No OneDrive connection found. Please connect your OneDrive account first.",
                "provider": "onedrive",
            }

        logger.info("🐛 STEP 3: CV data received")
        logger.info("🐛 STEP 4: CV data parsed successfully")
        logger.info(f"   - Title: {cv_data.get('title', 'No title')}")
        logger.info(f"   - Has personal_info: {bool(cv_data.get('personal_info'))}")
        logger.info(
            f"   - Photo field: {cv_data.get('photo', {}).get('photolink') is not None}"
        )
        logger.info(f"   - Data keys: {list(cv_data.keys())}")

        # Validate that we have minimum required data
        if not cv_data.get("title"):
            logger.error("❌ STEP 5 FAILED: No title provided in CV data")
            return {
                "success": False,
                "error": "CV title is required",
                "provider": "onedrive",
            }

        logger.info("🐛 STEP 6: About to ensure valid token")

        # Ensure token is valid (refresh if needed)
        try:
            valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)
            logger.info("🐛 STEP 7: Token validation successful")
        except Exception as token_error:
            logger.error(f"❌ STEP 6 FAILED: Token validation failed: {token_error}")
            return {
                "success": False,
                "error": "OneDrive connection expired. Please reconnect.",
                "provider": "onedrive",
            }

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )
            logger.info("🐛 STEP 9: Session updated with new tokens")
        else:
            logger.info("🐛 STEP 9: No session update needed")

        logger.info("🐛 STEP 10: About to call onedrive_service.save_cv")

        # Save CV to OneDrive
        try:
            logger.info(f"💾 Starting OneDrive save for: {cv_data.get('title')}")

            file_id = await onedrive_service.save_cv(valid_tokens, cv_data)

            processing_time = time.time() - start_time
            logger.info(
                f"✅ STEP 11: OneDrive save completed successfully: {file_id} (took {processing_time:.2f}s)"
            )

            # Record activity in background
            try:
                await record_session_activity(
                    session["session_id"],
                    "cv_saved",
                    {"provider": "onedrive", "file_id": file_id},
                )
                logger.info("🐛 STEP 12: Activity recorded successfully")
            except Exception as activity_error:
                logger.warning(
                    f"⚠️ Failed to record activity (non-critical): {activity_error}"
                )

            # Return success response
            response_data = {
                "success": True,
                "provider": "onedrive",
                "file_id": file_id,
                "message": f"CV '{cv_data.get('title')}' saved to OneDrive successfully",
            }
            logger.info("🐛 STEP 13: Returning success response")
            return response_data

        except ValidationError as ve:
            logger.error(f"❌ STEP 10 FAILED - CV validation error during save: {ve}")
            validation_details = []
            for error in ve.errors():
                validation_details.append(
                    f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                )

            return {
                "success": False,
                "error": f"CV data validation failed: {'; '.join(validation_details[:3])}{'...' if len(validation_details) > 3 else ''}",
                "provider": "onedrive",
            }

        except OneDriveError as od_error:
            logger.error(f"❌ STEP 10 FAILED - OneDrive service error: {od_error}")
            return {
                "success": False,
                "error": f"OneDrive error: {str(od_error)}",
                "provider": "onedrive",
            }
        except Exception as save_error:
            logger.error(f"❌ STEP 10 FAILED - Unexpected save error: {save_error}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Unexpected error during save: {str(save_error)}",
                "provider": "onedrive",
            }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ OVERALL FAILURE after {processing_time:.2f}s: {str(e)}")
        import traceback

        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Failed to save CV: {str(e)}",
            "provider": "onedrive",
        }


@router.get("/debug")
async def debug_onedrive_session(session: dict = Depends(get_current_session)):
    """Debug endpoint for OneDrive session info"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive", {})

        return {
            "session_id": session.get("session_id"),
            "has_onedrive_tokens": "onedrive" in cloud_tokens,
            "token_keys": list(onedrive_tokens.keys()) if onedrive_tokens else [],
            "has_access_token": bool(onedrive_tokens.get("access_token")),
            "has_refresh_token": bool(onedrive_tokens.get("refresh_token")),
            "expires_at": onedrive_tokens.get("expires_at"),
            "email": onedrive_tokens.get("email"),
            "provider_count": len(cloud_tokens),
        }

    except Exception as e:
        logger.error(f"❌ Debug info failed: {str(e)}")
        return {"error": str(e), "session_id": session.get("session_id", "unknown")}


# Cover letter endpoints for OneDrive
@router.post("/cover-letter/save-cover-letter")
async def save_cover_letter_to_onedrive(
    cover_letter_data: dict,
    session: dict = Depends(get_current_session),
):
    """Save a cover letter to OneDrive"""
    try:
        logger.info("💾 Starting cover letter save to OneDrive...")

        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            return {
                "success": False,
                "error": "No OneDrive connection found",
                "provider": "onedrive",
            }

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Prepare cover letter for storage
        storage_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "type": "cover_letter",
            },
            "cover_letter_data": cover_letter_data,
        }

        # Generate filename
        safe_title = "".join(
            c
            for c in cover_letter_data.get("title", "cover_letter")
            if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:50]
        filename = f"cover_letter_{safe_title}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        # Save to OneDrive
        content = json.dumps(storage_data, indent=2, default=str)
        access_token = valid_tokens["access_token"]

        async with OneDriveProvider(access_token) as provider:
            file_id = await provider.upload_file(
                filename, content, folder_name="Cover_Letters"
            )

        logger.info(f"✅ Cover letter saved successfully: {file_id}")

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cover_letter_saved",
            {
                "provider": "onedrive",
                "file_id": file_id,
                "title": cover_letter_data.get("title"),
            },
        )

        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "message": "Cover letter saved to OneDrive successfully",
            "provider": "onedrive",
            "cover_letter_data": {**cover_letter_data, "id": file_id},
        }

    except Exception as e:
        logger.error(f"❌ Cover letter save failed: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to save cover letter: {str(e)}",
            "provider": "onedrive",
        }


@router.get("/cover-letters")
async def list_cover_letters_from_onedrive(
    session: dict = Depends(get_current_session),
):
    """List all cover letters from OneDrive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            raise HTTPException(status_code=403, detail="No OneDrive connection found")

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        access_token = valid_tokens["access_token"]

        async with OneDriveProvider(access_token) as provider:
            # Look for cover letter files in Cover_Letters folder
            files = await provider.list_files(folder_name="Cover_Letters")

        # Process each cover letter file to extract metadata
        cover_letter_files = []
        for file in files:
            if "cover_letter" in file.name.lower() and file.name.endswith(".json"):
                try:
                    # Load the file content to extract company and job title
                    async with OneDriveProvider(access_token) as provider:
                        content = await provider.download_file(file.file_id)

                    # Parse the content to get metadata
                    company_name = "Not specified"
                    job_title = "Not specified"
                    title = (
                        file.name.replace(".json", "")
                        .replace("cover_letter_", "")
                        .replace("_", " ")
                        .title()
                    )

                    try:
                        data = json.loads(content)
                        cover_letter_data = data.get("cover_letter_data", {})

                        # Extract the actual data
                        company_name = (
                            cover_letter_data.get("company_name") or "Not specified"
                        )
                        job_title = (
                            cover_letter_data.get("job_title") or "Not specified"
                        )
                        title = cover_letter_data.get("title") or title

                        logger.info(
                            f"📄 Extracted metadata - Title: {title}, Company: {company_name}, Job: {job_title}"
                        )

                    except Exception as parse_error:
                        logger.warning(
                            f"⚠️ Failed to parse cover letter content for {file.file_id}: {parse_error}"
                        )

                    cover_letter_files.append(
                        {
                            "id": file.file_id,
                            "title": title,
                            "company_name": company_name,
                            "job_title": job_title,
                            "name": file.name,
                            "created_at": file.created_at.isoformat(),
                            "updated_at": file.last_modified.isoformat(),
                            "size": file.size_bytes,
                        }
                    )

                except Exception as file_error:
                    logger.warning(
                        f"⚠️ Failed to process cover letter file {file.file_id}: {file_error}"
                    )
                    # Add basic metadata if we can't parse the content
                    cover_letter_files.append(
                        {
                            "id": file.file_id,
                            "title": file.name.replace(".json", "")
                            .replace("cover_letter_", "")
                            .replace("_", " ")
                            .title(),
                            "company_name": "Not specified",
                            "job_title": "Not specified",
                            "name": file.name,
                            "created_at": file.created_at.isoformat(),
                            "updated_at": file.last_modified.isoformat(),
                            "size": file.size_bytes,
                        }
                    )

        logger.info(f"📋 Found {len(cover_letter_files)} cover letters")

        return {
            "success": True,
            "provider": "onedrive",
            "cover_letters": cover_letter_files,
            "count": len(cover_letter_files),
        }

    except Exception as e:
        logger.error(f"❌ List cover letters failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list cover letters: {str(e)}"
        )


@router.get("/cover-letters/{file_id}")
async def load_cover_letter_from_onedrive(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Load a specific cover letter from OneDrive"""
    try:
        logger.info(f"📥 Loading cover letter: {file_id}")

        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            raise HTTPException(status_code=403, detail="No OneDrive connection found")

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        access_token = valid_tokens["access_token"]

        # Load cover letter from OneDrive
        async with OneDriveProvider(access_token) as provider:
            content = await provider.download_file(file_id)

        logger.info(f"📄 Raw content loaded, length: {len(content)} chars")

        # Parse JSON content without schema validation
        try:
            data = json.loads(content)
            logger.info("📋 JSON parsed successfully")

            # Extract cover letter data
            cover_letter_data = data.get("cover_letter_data", {})

            if not cover_letter_data:
                # Maybe it's direct cover letter data
                cover_letter_data = data

            # Structure the response for frontend
            structured_data = {
                "id": file_id,
                "title": cover_letter_data.get("title", "Untitled Cover Letter"),
                "company_name": cover_letter_data.get("company_name", ""),
                "job_title": cover_letter_data.get("job_title", ""),
                "recipient_name": cover_letter_data.get("recipient_name", ""),
                "recipient_title": cover_letter_data.get("recipient_title", ""),
                "job_description": cover_letter_data.get("job_description", ""),
                "cover_letter_content": cover_letter_data.get(
                    "cover_letter_content", ""
                ),
                "applicant_info": cover_letter_data.get("applicant_info", {}),
                "job_info": cover_letter_data.get("job_info", {}),
                "is_favorite": cover_letter_data.get("is_favorite", False),
                "resume_id": cover_letter_data.get("resume_id"),
                "created_at": cover_letter_data.get(
                    "created_at", datetime.utcnow().isoformat()
                ),
                "updated_at": cover_letter_data.get(
                    "updated_at", datetime.utcnow().isoformat()
                ),
                # Add author info for display
                "author_name": cover_letter_data.get("applicant_info", {}).get(
                    "name", ""
                ),
                "author_email": cover_letter_data.get("applicant_info", {}).get(
                    "email", ""
                ),
                "author_phone": cover_letter_data.get("applicant_info", {}).get(
                    "phone", ""
                ),
            }

            logger.info(
                f"✅ Cover letter structured successfully: {structured_data['title']}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse cover letter JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid cover letter format")

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cover_letter_loaded",
            {"provider": "onedrive", "file_id": file_id},
        )

        return {
            "success": True,
            "provider": "onedrive",
            "cover_letter_data": structured_data,
        }

    except OneDriveError as e:
        logger.error(f"❌ OneDrive load failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"OneDrive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Cover letter load failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load cover letter: {str(e)}"
        )


@router.delete("/cover-letters/{file_id}")
async def delete_cover_letter_from_onedrive(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Delete a cover letter from OneDrive"""
    try:
        logger.info(f"🗑️ Deleting cover letter: {file_id}")

        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            raise HTTPException(status_code=403, detail="No OneDrive connection found")

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        access_token = valid_tokens["access_token"]

        # Delete cover letter from OneDrive
        async with OneDriveProvider(access_token) as provider:
            success = await provider.delete_file(file_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="Cover letter not found or could not be deleted"
            )

        logger.info(f"✅ Cover letter deleted successfully: {file_id}")

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cover_letter_deleted",
            {"provider": "onedrive", "file_id": file_id},
        )

        return {
            "success": True,
            "message": "Cover letter deleted successfully",
            "provider": "onedrive",
            "file_id": file_id,
        }

    except OneDriveError as e:
        logger.error(f"❌ OneDrive delete failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"OneDrive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Cover letter delete failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete cover letter: {str(e)}"
        )


@router.put("/cover-letters/{file_id}")
async def update_cover_letter_in_onedrive(
    file_id: str,
    cover_letter_data: dict,
    session: dict = Depends(get_current_session),
):
    """Update an existing cover letter in OneDrive"""
    try:
        logger.info(f"🔄 Updating cover letter in OneDrive: {file_id}")

        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            return {
                "success": False,
                "error": "No OneDrive connection found",
                "provider": "onedrive",
            }

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Prepare updated cover letter data
        updated_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "type": "cover_letter",
            },
            "cover_letter_data": {
                **cover_letter_data,
                "updated_at": datetime.utcnow().isoformat(),
            },
        }

        # Convert to JSON
        content = json.dumps(updated_data, indent=2, default=str)
        access_token = valid_tokens["access_token"]

        # Update file in OneDrive
        async with OneDriveProvider(access_token) as provider:
            success = await provider.update_file(file_id, content)

        if success:
            logger.info(f"✅ Cover letter updated successfully: {file_id}")

            # Record activity
            await record_session_activity(
                session["session_id"],
                "cover_letter_updated",
                {
                    "provider": "onedrive",
                    "file_id": file_id,
                    "title": cover_letter_data.get("title"),
                },
            )

            return {
                "success": True,
                "file_id": file_id,
                "message": "Cover letter updated successfully",
                "provider": "onedrive",
            }
        else:
            return {
                "success": False,
                "error": "Failed to update cover letter",
                "provider": "onedrive",
            }

    except Exception as e:
        logger.error(f"❌ Update cover letter failed: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to update cover letter: {str(e)}",
            "provider": "onedrive",
        }  # List CVs
        files = await onedrive_service.list_cvs(valid_tokens)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_list",
            {"provider": "onedrive", "count": len(files)},
        )

        return {
            "provider": "onedrive",
            "files": [file.dict() for file in files],
            "count": len(files),
        }

    except OneDriveError as e:
        logger.error(f"❌ OneDrive list failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"OneDrive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV list failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list CVs: {str(e)}")


@router.get("/load/{file_id}")
async def load_cv_from_onedrive(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Load a specific CV from OneDrive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            raise HTTPException(status_code=403, detail="No OneDrive connection found")

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Load CV from OneDrive
        cv_data = await onedrive_service.load_cv(valid_tokens, file_id)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_loaded",
            {"provider": "onedrive", "file_id": file_id},
        )

        # Convert to response format - ensure it matches frontend expectations
        response_data = cv_data.dict()
        response_data["id"] = file_id

        return {"success": True, "provider": "onedrive", "cv_data": response_data}

    except OneDriveError as e:
        logger.error(f"❌ OneDrive load failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"OneDrive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV load failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load CV: {str(e)}")


@router.delete("/delete/{file_id}")
async def delete_cv_from_onedrive(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Delete a CV from OneDrive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            raise HTTPException(status_code=403, detail="No OneDrive connection found")

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Delete CV from OneDrive
        success = await onedrive_service.delete_cv(valid_tokens, file_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="CV not found or could not be deleted"
            )

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_deleted",
            {"provider": "onedrive", "file_id": file_id},
        )

        return {
            "success": True,
            "message": "CV deleted successfully",
            "provider": "onedrive",
            "file_id": file_id,
        }

    except OneDriveError as e:
        logger.error(f"❌ OneDrive delete failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"OneDrive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete CV: {str(e)}")


@router.post("/disconnect")
async def disconnect_onedrive(session: dict = Depends(get_current_session)):
    """Disconnect from OneDrive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if "onedrive" not in cloud_tokens:
            raise HTTPException(status_code=404, detail="No OneDrive connection found")

        # Remove OneDrive tokens
        del cloud_tokens["onedrive"]

        # Update session
        await session_manager.update_session_cloud_tokens(
            session["session_id"], cloud_tokens
        )

        # Record activity
        await record_session_activity(
            session["session_id"], "onedrive_disconnected", {}
        )

        return {
            "success": True,
            "message": "Disconnected from OneDrive",
            "provider": "onedrive",
        }

    except Exception as e:
        logger.error(f"❌ OneDrive disconnection failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to disconnect from OneDrive: {str(e)}"
        )


@router.put("/update-file/{file_id}")
async def update_cv_in_onedrive(
    file_id: str,
    cv_data: dict,
    session: dict = Depends(get_current_session),
):
    """Update an existing CV in OneDrive"""
    try:
        logger.info(f"🔄 Updating CV in OneDrive: {file_id}")

        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            return {
                "success": False,
                "error": "No OneDrive connection found",
                "provider": "onedrive",
            }

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Update CV in OneDrive
        success = await onedrive_service.update_cv(valid_tokens, file_id, cv_data)

        if success:
            # Record activity
            await record_session_activity(
                session["session_id"],
                "cv_updated",
                {"provider": "onedrive", "file_id": file_id},
            )

            return {
                "success": True,
                "file_id": file_id,
                "message": "CV updated successfully",
                "provider": "onedrive",
            }
        else:
            return {
                "success": False,
                "error": "Failed to update CV",
                "provider": "onedrive",
            }

    except Exception as e:
        logger.error(f"❌ Update CV failed: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to update CV: {str(e)}",
            "provider": "onedrive",
        }


@router.get("/list")
async def list_onedrive_cvs(session: dict = Depends(get_current_session)):
    """List all CVs from OneDrive"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        onedrive_tokens = cloud_tokens.get("onedrive")

        if not onedrive_tokens:
            raise HTTPException(status_code=403, detail="No OneDrive connection found")

        # Ensure token is valid
        valid_tokens = await onedrive_service.ensure_valid_token(onedrive_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != onedrive_tokens:
            cloud_tokens["onedrive"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # List CVs
        files = await onedrive_service.list_cvs(valid_tokens)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_list",
            {"provider": "onedrive", "count": len(files)},
        )

        return {
            "provider": "onedrive",
            "files": [file.dict() for file in files],
            "count": len(files),
        }

    except OneDriveError as e:
        logger.error(f"❌ OneDrive list failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"OneDrive error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV list failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list CVs: {str(e)}")


@router.post("/exchange-token")
async def exchange_onedrive_token(
    request: dict, session: dict = Depends(get_current_session)
):
    """Exchange authorization code for OneDrive tokens (frontend-first flow)"""
    try:
        code = request.get("code")
        state = request.get("state")

        if not code:
            return {"success": False, "error": "No authorization code provided"}

        # Exchange code for tokens
        tokens = await onedrive_service.exchange_code_for_tokens(code)

        # Test connection
        connection_test = await onedrive_service.test_connection(tokens)
        if not connection_test["success"]:
            return {"success": False, "error": "Connection test failed"}

        # Store tokens in session
        cloud_tokens = session.get("cloud_tokens", {})
        cloud_tokens["onedrive"] = {
            **tokens,
            "email": connection_test["user"]["email"],
            "name": connection_test["user"]["name"],
        }

        await session_manager.update_session_cloud_tokens(
            session["session_id"], cloud_tokens
        )

        return {
            "success": True,
            "provider": "onedrive",
            "user": connection_test["user"],
        }

    except Exception as e:
        logger.error(f"OneDrive token exchange failed: {e}")
        return {"success": False, "error": str(e)}
