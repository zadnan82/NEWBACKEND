# app/cloud/onedrive_service.py
"""
OneDrive focused service - Matches Google Drive service functionality
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from pydantic import ValidationError

from ..config import get_settings
from ..schemas import (
    CloudProvider,
    CloudFileMetadata,
    CompleteCV,
    CVFileMetadata,
    CloudConnectionStatus,
)
from .onedrive import (
    OneDriveProvider,
    OneDriveError,
    onedrive_oauth,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class OneDriveService:
    """Service for OneDrive operations - matches Google Drive service exactly"""

    def __init__(self):
        # Use persistent encryption key from settings
        self.encryption_key = settings.get_encryption_key_bytes()
        self.cipher = Fernet(self.encryption_key)
        logger.info("OneDriveService initialized")

    def _encrypt_token_data(self, token_data: Dict[str, Any]) -> str:
        """Encrypt OneDrive token data"""
        try:
            token_json = json.dumps(token_data, default=str)
            encrypted = self.cipher.encrypt(token_json.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise OneDriveError(f"Failed to encrypt tokens: {str(e)}")

    def _decrypt_token_data(self, encrypted_tokens: str) -> Dict[str, Any]:
        """Decrypt OneDrive token data"""
        try:
            decrypted = self.cipher.decrypt(encrypted_tokens.encode())
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise OneDriveError(f"Failed to decrypt tokens: {str(e)}")

    def _format_cv_filename(self, title: str, timestamp: datetime = None) -> str:
        """Generate standardized CV filename for OneDrive"""
        timestamp = timestamp or datetime.utcnow()
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:50]
        return f"cv_{safe_title}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

    def _parse_cv_from_storage(self, content: str) -> CompleteCV:
        """Parse CV data from OneDrive storage"""
        try:
            data = json.loads(content)
            cv_data = data.get("cv_data", data)

            # Convert the data to match frontend schema exactly
            frontend_cv = self._convert_backend_to_frontend_schema(cv_data)

            return CompleteCV.parse_obj(frontend_cv)
        except Exception as e:
            logger.error(f"CV parsing failed: {e}")
            raise ValueError(f"Invalid CV file format: {e}")

    def _convert_frontend_to_backend_schema(self, cv_data: Dict) -> Dict:
        """Convert frontend schema to backend schema"""
        logger.info(
            f"Converting frontend schema to backend: {cv_data.get('title', 'No title')}"
        )

        # Handle photo field - frontend sends 'photo', backend expects 'photo'
        photo_data = cv_data.get("photo", cv_data.get("photos", {}))
        if not isinstance(photo_data, dict):
            photo_data = {"photolink": None}

        # Log photo information for debugging
        if photo_data.get("photolink"):
            photolink = photo_data["photolink"]
            if photolink.startswith("data:image"):
                logger.info(
                    f"📷 Base64 image detected, size: ~{len(photolink) // 1024}KB"
                )
            elif photolink.startswith("http"):
                logger.info(f"📷 URL image detected: {photolink[:50]}...")
            else:
                logger.warning(f"📷 Unknown photo format: {photolink[:50]}...")

        converted = {
            "title": cv_data.get("title", "My Resume"),
            "is_public": cv_data.get("is_public", False),
            "customization": cv_data.get(
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
            "personal_info": cv_data.get("personal_info", {}),
            "educations": cv_data.get("educations", []),
            "experiences": cv_data.get("experiences", []),
            "skills": cv_data.get("skills", []),
            "languages": cv_data.get("languages", []),
            "referrals": cv_data.get("referrals", []),
            "custom_sections": cv_data.get("custom_sections", []),
            "extracurriculars": cv_data.get("extracurriculars", []),
            "hobbies": cv_data.get("hobbies", []),
            "courses": cv_data.get("courses", []),
            "internships": cv_data.get("internships", []),
            "photo": photo_data,
        }

        logger.info(
            f"✅ Schema conversion completed - photo field: {bool(converted['photo'].get('photolink'))}"
        )
        return converted

    def _prepare_cv_for_storage(self, cv_data: CompleteCV) -> str:
        """Prepare CV data for OneDrive storage"""
        cv_dict = cv_data.dict()

        # Log storage preparation info
        photo_info = "No photo"
        if cv_dict.get("photo", {}).get("photolink"):
            photolink = cv_dict["photo"]["photolink"]
            if photolink.startswith("data:image"):
                size_kb = len(photolink) // 1024
                photo_info = f"Base64 image (~{size_kb}KB)"
            elif photolink.startswith("http"):
                photo_info = f"URL image"
            else:
                photo_info = f"Unknown format"

        logger.info(
            f"📦 Preparing CV for OneDrive storage: {cv_data.title}, Photo: {photo_info}"
        )

        storage_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "provider": "onedrive",
                "photo_type": "base64"
                if cv_dict.get("photo", {})
                .get("photolink", "")
                .startswith("data:image")
                else "url"
                if cv_dict.get("photo", {}).get("photolink", "").startswith("http")
                else "none",
            },
            "cv_data": cv_dict,
        }

        # Calculate approximate storage size
        json_str = json.dumps(storage_data, indent=2, default=str)
        size_mb = len(json_str.encode("utf-8")) / (1024 * 1024)
        logger.info(f"📦 Storage data size: ~{size_mb:.2f}MB")

        if size_mb > 10:  # Warn if file is getting large
            logger.warning(f"⚠️ Large file size detected: {size_mb:.2f}MB")

        return json_str

    def _convert_backend_to_frontend_schema(self, cv_data: Dict) -> Dict:
        """Convert backend schema to frontend schema"""
        logger.info(
            f"Converting backend schema to frontend: {cv_data.get('title', 'No title')}"
        )

        # Handle photo field - backend uses 'photo', frontend expects 'photos' in some contexts
        photo_data = cv_data.get("photo", cv_data.get("photos", {}))
        if not isinstance(photo_data, dict):
            photo_data = {"photolink": None}

        converted = {
            "title": cv_data.get("title", "My Resume"),
            "is_public": cv_data.get("is_public", False),
            "customization": cv_data.get(
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
            "personal_info": cv_data.get("personal_info", {}),
            "educations": cv_data.get("educations", []),
            "experiences": cv_data.get("experiences", []),
            "skills": cv_data.get("skills", []),
            "languages": cv_data.get("languages", []),
            "referrals": cv_data.get("referrals", []),
            "custom_sections": cv_data.get("custom_sections", []),
            "extracurriculars": cv_data.get("extracurriculars", []),
            "hobbies": cv_data.get("hobbies", []),
            "courses": cv_data.get("courses", []),
            "internships": cv_data.get("internships", []),
            "photo": photo_data,
        }

        return converted

    async def get_connection_status(
        self, token_data: Dict[str, Any]
    ) -> CloudConnectionStatus:
        """Get OneDrive connection status"""
        if not token_data or not token_data.get("access_token"):
            return CloudConnectionStatus(
                provider=CloudProvider.ONEDRIVE,
                connected=False,
                email=None,
                storage_quota=None,
            )

        try:
            access_token = token_data["access_token"]

            async with OneDriveProvider(access_token) as provider:
                # Test connection and get user info
                connection_test = await provider.test_connection()

                if not connection_test["success"]:
                    return CloudConnectionStatus(
                        provider=CloudProvider.ONEDRIVE,
                        connected=False,
                        email=None,
                        storage_quota=None,
                    )

                # Get storage quota
                try:
                    storage_quota = await provider.get_storage_quota()
                except Exception as e:
                    logger.warning(f"Failed to get storage quota: {e}")
                    storage_quota = None

                return CloudConnectionStatus(
                    provider=CloudProvider.ONEDRIVE,
                    connected=True,
                    email=connection_test["user"]["email"],
                    storage_quota=storage_quota,
                )

        except Exception as e:
            logger.error(f"OneDrive connection status check failed: {e}")
            return CloudConnectionStatus(
                provider=CloudProvider.ONEDRIVE,
                connected=False,
                email=None,
                storage_quota=None,
                error=str(e),
            )

    async def save_cv(self, tokens: dict, cv_data: Dict) -> str:
        """Save CV to OneDrive"""
        try:
            logger.info(
                f"💾 Starting OneDrive save for CV: {cv_data.get('title', 'Unknown')}"
            )

            # Convert frontend schema to what CompleteCV expects
            converted_data = self._convert_frontend_to_backend_schema(cv_data)

            # Validate with CompleteCV schema
            try:
                complete_cv = CompleteCV(**converted_data)
                logger.info("✅ CV data validated successfully")
            except ValidationError as ve:
                logger.error(f"❌ CV validation failed: {ve}")
                for error in ve.errors():
                    logger.error(f"   - {error['loc']}: {error['msg']}")
                raise OneDriveError(f"Invalid CV data: {ve}")

            # Prepare for storage
            file_name = self._format_cv_filename(complete_cv.title)
            content = self._prepare_cv_for_storage(complete_cv)

            # Upload to OneDrive
            access_token = tokens["access_token"]

            async with OneDriveProvider(access_token) as provider:
                file_id = await provider.upload_file(file_name, content)

            logger.info(f"✅ CV saved to OneDrive successfully: {file_id}")
            return file_id

        except ValidationError as e:
            logger.error(f"❌ CV validation failed: {e}")
            raise OneDriveError(f"Invalid CV data: {e}")
        except Exception as e:
            logger.error(f"❌ OneDrive save failed: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise OneDriveError(f"Failed to save CV: {str(e)}")

    async def update_cv(self, tokens: dict, file_id: str, cv_data: Dict) -> bool:
        """Update an existing CV in OneDrive"""
        try:
            logger.info(f"🔄 Updating CV in OneDrive: {file_id}")

            # Convert and validate data (same as save)
            converted_data = self._convert_frontend_to_backend_schema(cv_data)
            complete_cv = CompleteCV(**converted_data)

            # Prepare content for storage
            content = self._prepare_cv_for_storage(complete_cv)

            access_token = tokens["access_token"]

            async with OneDriveProvider(access_token) as provider:
                success = await provider.update_file(file_id, content)

            if success:
                logger.info(f"✅ CV updated successfully: {file_id}")
            else:
                logger.error(f"❌ Failed to update CV: {file_id}")

            return success

        except Exception as e:
            logger.error(f"❌ CV update failed: {e}")
            raise OneDriveError(f"Failed to update CV: {str(e)}")

    async def load_cv(self, token_data: Dict[str, Any], file_id: str) -> CompleteCV:
        """Load CV from OneDrive"""
        if not token_data or not token_data.get("access_token"):
            raise OneDriveError("No OneDrive access token available")

        access_token = token_data["access_token"]

        try:
            async with OneDriveProvider(access_token) as provider:
                content = await provider.download_file(file_id)
                cv_data = self._parse_cv_from_storage(content)
                logger.info(f"CV loaded from OneDrive: {file_id}")
                return cv_data
        except Exception as e:
            logger.error(f"Failed to load CV from OneDrive: {e}")
            raise OneDriveError(f"Failed to load CV: {str(e)}")

    async def list_cvs(self, token_data: Dict[str, Any]) -> List[CloudFileMetadata]:
        """List all CVs from OneDrive"""
        if not token_data or not token_data.get("access_token"):
            raise OneDriveError("No OneDrive access token available")

        access_token = token_data["access_token"]

        try:
            async with OneDriveProvider(access_token) as provider:
                files = await provider.list_files("CVs")
                logger.info(f"Listed {len(files)} CVs from OneDrive")
                return files
        except Exception as e:
            logger.error(f"Failed to list CVs from OneDrive: {e}")
            raise OneDriveError(f"Failed to list CVs: {str(e)}")

    async def delete_cv(self, token_data: Dict[str, Any], file_id: str) -> bool:
        """Delete CV from OneDrive"""
        if not token_data or not token_data.get("access_token"):
            raise OneDriveError("No OneDrive access token available")

        access_token = token_data["access_token"]

        try:
            async with OneDriveProvider(access_token) as provider:
                success = await provider.delete_file(file_id)

                if success:
                    logger.info(f"CV deleted from OneDrive: {file_id}")
                else:
                    logger.warning(f"Failed to delete CV from OneDrive: {file_id}")

                return success
        except Exception as e:
            logger.error(f"Failed to delete CV from OneDrive: {e}")
            raise OneDriveError(f"Failed to delete CV: {str(e)}")

    async def test_connection(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test OneDrive connection"""
        if not token_data or not token_data.get("access_token"):
            return {
                "success": False,
                "error": "No access token available",
                "provider": "onedrive",
            }

        access_token = token_data["access_token"]

        try:
            async with OneDriveProvider(access_token) as provider:
                result = await provider.test_connection()
                result["provider"] = "onedrive"
                return result
        except Exception as e:
            logger.error(f"OneDrive connection test failed: {e}")
            return {"success": False, "error": str(e), "provider": "onedrive"}

    def get_oauth_url(self, state: str) -> str:
        """Get OneDrive OAuth authorization URL"""
        return onedrive_oauth.get_auth_url(state)

    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        return await onedrive_oauth.exchange_code_for_tokens(code)

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh OneDrive access token"""
        if not refresh_token:
            raise OneDriveError("Refresh token is required")
        return await onedrive_oauth.refresh_token(refresh_token)

    async def ensure_valid_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure token is valid, refresh if necessary"""
        if not token_data.get("expires_at"):
            return token_data  # No expiration info, assume valid

        try:
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            now = datetime.utcnow()

            # Check if token expires in the next 5 minutes
            if (expires_at - now).total_seconds() < 300:
                logger.info("OneDrive token expired/expiring, refreshing...")

                if not token_data.get("refresh_token"):
                    raise OneDriveError("Token expired and no refresh token available")

                # Refresh the token
                refreshed_tokens = await self.refresh_access_token(
                    token_data["refresh_token"]
                )

                # Merge old data with new tokens
                updated_token_data = {
                    **token_data,
                    "access_token": refreshed_tokens["access_token"],
                    "expires_at": refreshed_tokens["expires_at"],
                    "refresh_token": refreshed_tokens.get(
                        "refresh_token", token_data.get("refresh_token")
                    ),
                }

                logger.info("Successfully refreshed OneDrive token")
                return updated_token_data

        except Exception as e:
            logger.error(f"Token validation/refresh failed: {e}")
            # Return original token data, let the API call fail if it's actually invalid

        return token_data


# Global OneDrive service instance
onedrive_service = OneDriveService()
