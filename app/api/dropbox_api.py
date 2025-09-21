# app/api/dropbox_api.py
"""
Dropbox API endpoints - Following Google Drive API pattern
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

import time
from datetime import datetime
import secrets
import json
from pydantic import BaseModel, ValidationError

from app.cloud.dropbox import DropboxProvider

from ..auth.sessions import (
    get_current_session,
    get_optional_session,
    session_manager,
    record_session_activity,
)
from ..cloud.dropbox_service import dropbox_service, DropboxError
from ..schemas import CompleteCV, CloudConnectionStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/providers")
async def list_dropbox_info():
    """Get Dropbox provider information"""
    return {
        "providers": [
            {
                "id": "dropbox",
                "name": "Dropbox",
                "description": "Store your CVs in Dropbox",
                "logo_url": "/static/logos/dropbox.png",
                "supported_features": ["read", "write", "delete", "folders"],
                "status": "available",
            }
        ]
    }


@router.post("/connect")
async def initiate_dropbox_connection(
    request: Request, session: dict = Depends(get_optional_session)
):
    """Initiate Dropbox OAuth connection"""
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

        # Get OAuth URL from Dropbox service
        auth_url = dropbox_service.get_oauth_url(state)

        logger.info(f"🔗 Generated Dropbox OAuth URL for session: {session_id}")

        return {"auth_url": auth_url, "state": state, "provider": "dropbox"}

    except Exception as e:
        logger.error(f"❌ Dropbox connection initiation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate Dropbox connection: {str(e)}",
        )


@router.get("/callback")
async def handle_dropbox_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Dropbox"),
    state: str = Query(..., description="State parameter from Dropbox"),
    error: Optional[str] = Query(None, description="Error from Dropbox"),
    error_description: Optional[str] = Query(None, description="Error description"),
):
    """Handle Dropbox OAuth callback"""

    logger.info(f"🔗 Dropbox OAuth callback received")
    logger.info(f"🔗 Code: {code[:20] if code else 'None'}...")
    logger.info(f"🔗 State: {state}")
    logger.info(f"🔗 Error: {error}")

    # Check for OAuth errors
    if error:
        logger.error(f"❌ Dropbox OAuth error: {error}")
        error_msg = f"Dropbox authorization failed: {error}"
        if error_description:
            error_msg += f" - {error_description}"

        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/dropbox?error={error}&error_description={error_description or ''}"
        )

    if not code:
        logger.error("❌ No authorization code received")
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/dropbox?error=no_code"
        )

    if not state:
        logger.error("❌ No state parameter received")
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/dropbox?error=no_state"
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
                url="http://localhost:5173/cloud/callback/dropbox?error=invalid_session"
            )

        # Exchange code for tokens
        logger.info("🔄 Exchanging authorization code for tokens...")
        tokens = await dropbox_service.exchange_code_for_tokens(code)

        logger.info("✅ Successfully exchanged code for tokens")

        # Test the connection to get user info
        connection_test = await dropbox_service.test_connection(tokens)
        if not connection_test["success"]:
            raise DropboxError(
                f"Connection test failed: {connection_test.get('error')}"
            )

        # Store tokens in session
        cloud_tokens = session_data.get("cloud_tokens", {})
        cloud_tokens["dropbox"] = {
            **tokens,
            "email": connection_test["user"]["email"],
            "name": connection_test["user"]["name"],
        }

        # Update session with tokens
        await session_manager.update_session_cloud_tokens(session_id, cloud_tokens)
        logger.info("✅ Successfully stored Dropbox tokens")

        # Record activity
        await record_session_activity(
            session_id,
            "dropbox_connected",
            {"email": connection_test["user"]["email"]},
        )

        # Redirect to frontend with success
        return RedirectResponse(
            url="http://localhost:5173/cloud/callback/dropbox?success=true&provider=dropbox"
        )

    except Exception as e:
        logger.error(f"❌ Dropbox callback processing failed: {str(e)}")
        return RedirectResponse(
            url=f"http://localhost:5173/cloud/callback/dropbox?error=processing_failed&error_description={str(e)}"
        )


@router.get("/status")
async def get_dropbox_status(session: dict = Depends(get_current_session)):
    """Get Dropbox connection status"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox")

        if not dropbox_tokens:
            return {
                "provider": "dropbox",
                "connected": False,
                "email": None,
                "storage_quota": None,
            }

        # Check connection status using the service
        status = await dropbox_service.get_connection_status(dropbox_tokens)
        return status.dict()

    except Exception as e:
        logger.error(f"❌ Dropbox status check failed: {str(e)}")
        return {
            "provider": "dropbox",
            "connected": False,
            "email": None,
            "storage_quota": None,
            "error": str(e),
        }


@router.post("/test-save")
async def test_dropbox_save(session: dict = Depends(get_current_session)):
    """Test Dropbox save functionality with simple data"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox")

        if not dropbox_tokens:
            return {
                "success": False,
                "error": "No Dropbox connection found",
                "provider": "dropbox",
            }

        # Create simple test data
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
        file_id = await dropbox_service.save_cv(dropbox_tokens, test_data)

        return {"success": True, "file_id": file_id, "message": "Test save successful"}

    except Exception as e:
        logger.error(f"❌ Test save failed: {str(e)}")
        return {"success": False, "error": str(e), "provider": "dropbox"}


@router.post("/save")
async def save_cv_to_dropbox(
    cv_data: dict,
    session: dict = Depends(get_current_session),
):
    """Save a CV to Dropbox"""
    start_time = time.time()

    try:
        logger.info("💾 Starting save_cv_to_dropbox function")

        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox")

        if not dropbox_tokens:
            logger.error("❌ No Dropbox connection found")
            return {
                "success": False,
                "error": "No Dropbox connection found. Please connect your Dropbox account first.",
                "provider": "dropbox",
            }

        logger.info(f"📄 CV data received: {cv_data.get('title', 'No title')}")

        # Validate that we have minimum required data
        if not cv_data.get("title"):
            logger.error("❌ No title provided in CV data")
            return {
                "success": False,
                "error": "CV title is required",
                "provider": "dropbox",
            }

        # Ensure token is valid
        try:
            valid_tokens = await dropbox_service.ensure_valid_token(dropbox_tokens)
            logger.info("✅ Token validation successful")
        except Exception as token_error:
            logger.error(f"❌ Token validation failed: {token_error}")
            return {
                "success": False,
                "error": "Dropbox connection expired. Please reconnect.",
                "provider": "dropbox",
            }

        # Update session if tokens were refreshed
        if valid_tokens != dropbox_tokens:
            cloud_tokens["dropbox"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )
            logger.info("✅ Session updated with new tokens")

        # Save CV to Dropbox
        try:
            logger.info(f"💾 Starting Dropbox save for: {cv_data.get('title')}")

            file_id = await dropbox_service.save_cv(valid_tokens, cv_data)

            processing_time = time.time() - start_time
            logger.info(
                f"✅ Dropbox save completed successfully: {file_id} (took {processing_time:.2f}s)"
            )

            # Record activity in background
            try:
                await record_session_activity(
                    session["session_id"],
                    "cv_saved",
                    {"provider": "dropbox", "file_id": file_id},
                )
                logger.info("✅ Activity recorded successfully")
            except Exception as activity_error:
                logger.warning(
                    f"⚠️ Failed to record activity (non-critical): {activity_error}"
                )

            # Return success response
            response_data = {
                "success": True,
                "provider": "dropbox",
                "file_id": file_id,
                "message": f"CV '{cv_data.get('title')}' saved to Dropbox successfully",
            }
            logger.info("✅ Returning success response")
            return response_data

        except ValidationError as ve:
            logger.error(f"❌ CV validation error during save: {ve}")
            validation_details = []
            for error in ve.errors():
                validation_details.append(
                    f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                )

            return {
                "success": False,
                "error": f"CV data validation failed: {'; '.join(validation_details[:3])}{'...' if len(validation_details) > 3 else ''}",
                "provider": "dropbox",
            }

        except DropboxError as db_error:
            logger.error(f"❌ Dropbox service error: {db_error}")
            return {
                "success": False,
                "error": f"Dropbox error: {str(db_error)}",
                "provider": "dropbox",
            }
        except Exception as save_error:
            logger.error(f"❌ Unexpected save error: {save_error}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Unexpected error during save: {str(save_error)}",
                "provider": "dropbox",
            }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Overall failure after {processing_time:.2f}s: {str(e)}")
        import traceback

        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Failed to save CV: {str(e)}",
            "provider": "dropbox",
        }


@router.get("/list")
async def list_dropbox_cvs(session: dict = Depends(get_current_session)):
    """List all CVs from Dropbox"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox")

        if not dropbox_tokens:
            raise HTTPException(status_code=403, detail="No Dropbox connection found")

        # Ensure token is valid
        valid_tokens = await dropbox_service.ensure_valid_token(dropbox_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != dropbox_tokens:
            cloud_tokens["dropbox"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # List CVs
        files = await dropbox_service.list_cvs(valid_tokens)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_list",
            {"provider": "dropbox", "count": len(files)},
        )

        return {
            "provider": "dropbox",
            "files": [file.dict() for file in files],
            "count": len(files),
        }

    except DropboxError as e:
        logger.error(f"❌ Dropbox list failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Dropbox error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV list failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list CVs: {str(e)}")


@router.get("/load/{file_id}")
async def load_cv_from_dropbox(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Load a specific CV from Dropbox"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox")

        if not dropbox_tokens:
            raise HTTPException(status_code=403, detail="No Dropbox connection found")

        # Ensure token is valid
        valid_tokens = await dropbox_service.ensure_valid_token(dropbox_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != dropbox_tokens:
            cloud_tokens["dropbox"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Load CV from Dropbox
        cv_data = await dropbox_service.load_cv(valid_tokens, file_id)

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_loaded",
            {"provider": "dropbox", "file_id": file_id},
        )

        # Convert to response format
        response_data = cv_data.dict()
        response_data["id"] = file_id

        return {"success": True, "provider": "dropbox", "cv_data": response_data}

    except DropboxError as e:
        logger.error(f"❌ Dropbox load failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Dropbox error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV load failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load CV: {str(e)}")


@router.delete("/delete/{file_id}")
async def delete_cv_from_dropbox(
    file_id: str,
    session: dict = Depends(get_current_session),
):
    """Delete a CV from Dropbox"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox")

        if not dropbox_tokens:
            raise HTTPException(status_code=403, detail="No Dropbox connection found")

        # Ensure token is valid
        valid_tokens = await dropbox_service.ensure_valid_token(dropbox_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != dropbox_tokens:
            cloud_tokens["dropbox"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Delete CV from Dropbox
        success = await dropbox_service.delete_cv(valid_tokens, file_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="CV not found or could not be deleted"
            )

        # Record activity
        await record_session_activity(
            session["session_id"],
            "cv_deleted",
            {"provider": "dropbox", "file_id": file_id},
        )

        return {
            "success": True,
            "message": "CV deleted successfully",
            "provider": "dropbox",
            "file_id": file_id,
        }

    except DropboxError as e:
        logger.error(f"❌ Dropbox delete failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Dropbox error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ CV delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete CV: {str(e)}")


@router.put("/update-file/{file_id}")
async def update_cv_in_dropbox(
    file_id: str,
    cv_data: dict,
    session: dict = Depends(get_current_session),
):
    """Update an existing CV in Dropbox"""
    try:
        logger.info(f"🔄 Updating CV in Dropbox: {file_id}")

        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox")

        if not dropbox_tokens:
            return {
                "success": False,
                "error": "No Dropbox connection found",
                "provider": "dropbox",
            }

        # Ensure token is valid
        valid_tokens = await dropbox_service.ensure_valid_token(dropbox_tokens)

        # Update session if tokens were refreshed
        if valid_tokens != dropbox_tokens:
            cloud_tokens["dropbox"] = valid_tokens
            await session_manager.update_session_cloud_tokens(
                session["session_id"], cloud_tokens
            )

        # Update CV in Dropbox
        success = await dropbox_service.update_cv(valid_tokens, file_id, cv_data)

        if success:
            # Record activity
            await record_session_activity(
                session["session_id"],
                "cv_updated",
                {"provider": "dropbox", "file_id": file_id},
            )

            return {
                "success": True,
                "file_id": file_id,
                "message": f"CV updated successfully",
                "provider": "dropbox",
            }
        else:
            return {
                "success": False,
                "error": "Failed to update CV",
                "provider": "dropbox",
            }

    except Exception as e:
        logger.error(f"❌ Update CV failed: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to update CV: {str(e)}",
            "provider": "dropbox",
        }


@router.post("/disconnect")
async def disconnect_dropbox(session: dict = Depends(get_current_session)):
    """Disconnect from Dropbox"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})

        if "dropbox" not in cloud_tokens:
            raise HTTPException(status_code=404, detail="No Dropbox connection found")

        # Remove Dropbox tokens
        del cloud_tokens["dropbox"]

        # Update session
        await session_manager.update_session_cloud_tokens(
            session["session_id"], cloud_tokens
        )

        # Record activity
        await record_session_activity(session["session_id"], "dropbox_disconnected", {})

        return {
            "success": True,
            "message": "Disconnected from Dropbox",
            "provider": "dropbox",
        }

    except Exception as e:
        logger.error(f"❌ Dropbox disconnection failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to disconnect from Dropbox: {str(e)}"
        )


@router.get("/debug")
async def debug_dropbox_session(session: dict = Depends(get_current_session)):
    """Debug endpoint for Dropbox session info"""
    try:
        cloud_tokens = session.get("cloud_tokens", {})
        dropbox_tokens = cloud_tokens.get("dropbox", {})

        return {
            "session_id": session.get("session_id"),
            "has_dropbox_tokens": "dropbox" in cloud_tokens,
            "token_keys": list(dropbox_tokens.keys()) if dropbox_tokens else [],
            "has_access_token": bool(dropbox_tokens.get("access_token")),
            "has_refresh_token": bool(dropbox_tokens.get("refresh_token")),
            "expires_at": dropbox_tokens.get("expires_at"),
            "email": dropbox_tokens.get("email"),
            "provider_count": len(cloud_tokens),
        }

    except Exception as e:
        logger.error(f"❌ Debug info failed: {str(e)}")
        return {"error": str(e), "session_id": session.get("session_id", "unknown")}
