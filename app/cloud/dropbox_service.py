# app/cloud/dropbox_service.py
"""
Dropbox focused service - Following Google Drive service pattern
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
    CloudConnectionStatus,
)
from .dropbox import (
    DropboxProvider,
    DropboxError,
    dropbox_oauth,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class DropboxService:
    """Service for Dropbox operations - matches Google Drive service pattern"""

    def __init__(self):
        # Use persistent encryption key from settings
        self.encryption_key = settings.get_encryption_key_bytes()
        self.cipher = Fernet(self.encryption_key)
        logger.info("DropboxService initialized")

    def _encrypt_token_data(self, token_data: Dict[str, Any]) -> str:
        """Encrypt Dropbox token data"""
        try:
            token_json = json.dumps(token_data, default=str)
            encrypted = self.cipher.encrypt(token_json.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise DropboxError(f"Failed to encrypt tokens: {str(e)}")

    def _decrypt_token_data(self, encrypted_tokens: str) -> Dict[str, Any]:
        """Decrypt Dropbox token data"""
        try:
            decrypted = self.cipher.decrypt(encrypted_tokens.encode())
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise DropboxError(f"Failed to decrypt tokens: {str(e)}")

    def _format_cv_filename(self, title: str, timestamp: datetime = None) -> str:
        """Generate standardized CV filename for Dropbox"""
        timestamp = timestamp or datetime.utcnow()
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_title = safe_title.replace(" ", "_")[:50]
        return f"cv_{safe_title}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

    def _parse_cv_from_storage(self, content: str) -> CompleteCV:
        """Parse CV data from Dropbox storage"""
        try:
            data = json.loads(content)
            cv_data = data.get("cv_data", data)

            # Convert the data to match frontend schema
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

        # Handle photo field
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

    def _convert_backend_to_frontend_schema(self, cv_data: Dict) -> Dict:
        """Convert backend schema to frontend schema - FIXED for Dropbox"""
        logger.info(
            f"Converting backend schema to frontend: {cv_data.get('title', 'No title')}"
        )

        # Handle different field naming patterns that might come from Dropbox
        personal_info = cv_data.get("personal_info", {})

        # Ensure full_name field exists and is properly formatted
        if "full_name" not in personal_info:
            # Try alternative field names that might be used
            if "name" in personal_info:
                personal_info["full_name"] = personal_info["name"]
            elif "displayName" in personal_info:
                personal_info["full_name"] = personal_info["displayName"]
            elif "firstName" in personal_info and "lastName" in personal_info:
                personal_info["full_name"] = (
                    f"{personal_info['firstName']} {personal_info['lastName']}"
                )
            else:
                personal_info["full_name"] = ""  # Ensure the field exists

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
            "personal_info": personal_info,  # Use the fixed personal_info
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
            "photo": cv_data.get("photo", cv_data.get("photos", {})),
        }

        logger.info(
            f"✅ Schema conversion completed - full_name: {converted['personal_info'].get('full_name', 'MISSING')}"
        )
        return converted

    def _parse_cv_from_storage(self, content: str) -> CompleteCV:
        """Parse CV data from Dropbox storage - WITH DEBUG"""
        try:
            data = json.loads(content)
            cv_data = data.get("cv_data", data)

            # DEBUG: Log the actual structure
            logger.info(f"📋 RAW DROPBOX CV DATA STRUCTURE:")
            logger.info(f"   - Has personal_info: {'personal_info' in cv_data}")
            if "personal_info" in cv_data:
                logger.info(
                    f"   - personal_info keys: {list(cv_data['personal_info'].keys())}"
                )
                logger.info(
                    f"   - full_name value: {cv_data['personal_info'].get('full_name', 'MISSING')}"
                )
                logger.info(
                    f"   - name value: {cv_data['personal_info'].get('name', 'MISSING')}"
                )

            # Convert the data to match frontend schema
            frontend_cv = self._convert_backend_to_frontend_schema(cv_data)

            # DEBUG: Log the converted structure
            logger.info(f"📋 CONVERTED CV DATA:")
            logger.info(
                f"   - full_name: {frontend_cv.get('personal_info', {}).get('full_name', 'STILL MISSING')}"
            )

            return CompleteCV.parse_obj(frontend_cv)
        except Exception as e:
            logger.error(f"CV parsing failed: {e}")
            raise ValueError(f"Invalid CV file format: {e}")

    def _prepare_cv_for_storage(self, cv_data: CompleteCV) -> str:
        """Prepare CV data for Dropbox storage"""
        cv_dict = cv_data.dict()

        storage_data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "last_modified": datetime.utcnow().isoformat(),
                "created_with": "cv-privacy-platform",
                "provider": "dropbox",
            },
            "cv_data": cv_dict,
        }

        return json.dumps(storage_data, indent=2, default=str)

    async def get_connection_status(
        self, token_data: Dict[str, Any]
    ) -> CloudConnectionStatus:
        """Get Dropbox connection status"""
        if not token_data or not token_data.get("access_token"):
            return CloudConnectionStatus(
                provider=CloudProvider.DROPBOX,
                connected=False,
                email=None,
                storage_quota=None,
            )

        try:
            access_token = token_data["access_token"]

            async with DropboxProvider(access_token) as provider:
                # Test connection and get user info
                connection_test = await provider.test_connection()

                if not connection_test["success"]:
                    return CloudConnectionStatus(
                        provider=CloudProvider.DROPBOX,
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
                    provider=CloudProvider.DROPBOX,
                    connected=True,
                    email=connection_test["user"]["email"],
                    storage_quota=storage_quota,
                )

        except Exception as e:
            logger.error(f"Dropbox connection status check failed: {e}")
            return CloudConnectionStatus(
                provider=CloudProvider.DROPBOX,
                connected=False,
                email=None,
                storage_quota=None,
                error=str(e),
            )

    async def test_connection(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Dropbox connection"""
        if not token_data or not token_data.get("access_token"):
            return {
                "success": False,
                "error": "No access token available",
                "provider": "dropbox",
            }

        access_token = token_data["access_token"]

        try:
            async with DropboxProvider(access_token) as provider:
                result = await provider.test_connection()
                result["provider"] = "dropbox"
                return result
        except Exception as e:
            logger.error(f"Dropbox connection test failed: {e}")
            return {"success": False, "error": str(e), "provider": "dropbox"}

    def get_oauth_url(self, state: str) -> str:
        """Get Dropbox OAuth authorization URL"""
        return dropbox_oauth.get_auth_url(state)

    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        return await dropbox_oauth.exchange_code_for_tokens(code)

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh Dropbox access token (usually not needed)"""
        if not refresh_token:
            raise DropboxError("Refresh token is required")
        return await dropbox_oauth.refresh_token(refresh_token)

    async def ensure_valid_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure token is valid - Dropbox tokens typically don't expire"""
        # Dropbox tokens don't expire by default, so just return the token data
        # But we can test the connection to make sure it's still valid
        try:
            test_result = await self.test_connection(token_data)
            if test_result.get("success"):
                return token_data
            else:
                # If connection test fails, try to refresh if we have a refresh token
                if token_data.get("refresh_token"):
                    logger.info("Dropbox connection test failed, attempting refresh...")
                    refreshed_tokens = await self.refresh_access_token(
                        token_data["refresh_token"]
                    )

                    # Merge old data with new tokens
                    updated_token_data = {
                        **token_data,
                        "access_token": refreshed_tokens["access_token"],
                        "refresh_token": refreshed_tokens.get(
                            "refresh_token", token_data.get("refresh_token")
                        ),
                    }

                    logger.info("Successfully refreshed Dropbox token")
                    return updated_token_data
                else:
                    logger.warning(
                        "Dropbox token invalid and no refresh token available"
                    )
                    return token_data
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return token_data

    async def save_cv(self, tokens: dict, cv_data: Dict) -> str:
        """Save CV to Dropbox"""
        try:
            logger.info(
                f"💾 Starting Dropbox save for CV: {cv_data.get('title', 'Unknown')}"
            )

            # Convert frontend schema to what CompleteCV expects
            converted_data = self._convert_frontend_to_backend_schema(cv_data)

            # Validate with CompleteCV schema
            try:
                complete_cv = CompleteCV(**converted_data)
                logger.info("✅ CV data validated successfully")
            except ValidationError as ve:
                logger.error(f"❌ CV validation failed: {ve}")
                raise DropboxError(f"Invalid CV data: {ve}")

            # Prepare for storage
            file_name = self._format_cv_filename(complete_cv.title)
            content = self._prepare_cv_for_storage(complete_cv)

            # Upload to Dropbox
            access_token = tokens["access_token"]

            async with DropboxProvider(access_token) as provider:
                file_id = await provider.upload_file(file_name, content)

            logger.info(f"✅ CV saved to Dropbox successfully: {file_id}")
            return file_id

        except ValidationError as e:
            logger.error(f"❌ CV validation failed: {e}")
            raise DropboxError(f"Invalid CV data: {e}")
        except Exception as e:
            logger.error(f"❌ Dropbox save failed: {e}")
            raise DropboxError(f"Failed to save CV: {str(e)}")

    async def load_cv(self, token_data: Dict[str, Any], file_id: str) -> CompleteCV:
        """Load CV from Dropbox"""
        if not token_data or not token_data.get("access_token"):
            raise DropboxError("No Dropbox access token available")

        access_token = token_data["access_token"]

        try:
            async with DropboxProvider(access_token) as provider:
                content = await provider.download_file(file_id)
                cv_data = self._parse_cv_from_storage(content)
                logger.info(f"CV loaded from Dropbox: {file_id}")
                return cv_data
        except Exception as e:
            logger.error(f"Failed to load CV from Dropbox: {e}")
            raise DropboxError(f"Failed to load CV: {str(e)}")

    async def list_cvs(self, token_data: Dict[str, Any]) -> List[CloudFileMetadata]:
        """List all CVs from Dropbox"""
        if not token_data or not token_data.get("access_token"):
            raise DropboxError("No Dropbox access token available")

        access_token = token_data["access_token"]

        try:
            async with DropboxProvider(access_token) as provider:
                files = await provider.list_files("CVs")
                logger.info(f"Listed {len(files)} CVs from Dropbox")
                return files
        except Exception as e:
            logger.error(f"Failed to list CVs from Dropbox: {e}")
            raise DropboxError(f"Failed to list CVs: {str(e)}")

    async def delete_cv(self, token_data: Dict[str, Any], file_id: str) -> bool:
        """Delete CV from Dropbox"""
        if not token_data or not token_data.get("access_token"):
            raise DropboxError("No Dropbox access token available")

        access_token = token_data["access_token"]

        try:
            async with DropboxProvider(access_token) as provider:
                success = await provider.delete_file(file_id)

                if success:
                    logger.info(f"CV deleted from Dropbox: {file_id}")
                else:
                    logger.warning(f"Failed to delete CV from Dropbox: {file_id}")

                return success
        except Exception as e:
            logger.error(f"Failed to delete CV from Dropbox: {e}")
            raise DropboxError(f"Failed to delete CV: {str(e)}")

    async def update_cv(self, tokens: dict, file_id: str, cv_data: Dict) -> bool:
        """Update an existing CV in Dropbox"""
        try:
            logger.info(f"🔄 Updating CV in Dropbox: {file_id}")

            # Convert and validate data
            converted_data = self._convert_frontend_to_backend_schema(cv_data)
            complete_cv = CompleteCV(**converted_data)

            # Prepare content for storage
            content = self._prepare_cv_for_storage(complete_cv)

            access_token = tokens["access_token"]

            async with DropboxProvider(access_token) as provider:
                success = await provider.update_file(file_id, content)

            if success:
                logger.info(f"✅ CV updated successfully: {file_id}")

            return success

        except Exception as e:
            logger.error(f"❌ CV update failed: {e}")
            raise DropboxError(f"Failed to update CV: {str(e)}")


# Global Dropbox service instance
dropbox_service = DropboxService()
