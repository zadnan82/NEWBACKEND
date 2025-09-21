# app/cloud/onedrive.py
"""
OneDrive provider - Separated implementation for Microsoft OneDrive integration
"""

import json
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

from ..config import get_settings
from ..schemas import CloudProvider, CloudFileMetadata

logger = logging.getLogger(__name__)
settings = get_settings()


class OneDriveError(Exception):
    """OneDrive specific errors"""

    pass


class OneDriveProvider:
    """Microsoft OneDrive integration - Separated and focused"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.api_base = "https://graph.microsoft.com/v1.0"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Microsoft Graph API"""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        # Merge with any existing headers
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
            kwargs["headers"] = headers
        else:
            kwargs["headers"] = headers

        logger.info(f"🔄 OneDrive API Request: {method} {url}")

        try:
            async with self.session.request(method, url, **kwargs) as response:
                logger.info(f"📊 Response Status: {response.status}")

                if response.status == 401:
                    error_text = await response.text()
                    logger.error(f"❌ OneDrive token expired: {error_text}")
                    raise OneDriveError("Access token expired or invalid")
                elif response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"❌ OneDrive API error {response.status}: {error_text}"
                    )
                    raise OneDriveError(f"API error {response.status}: {error_text}")

                # Handle different content types
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    result = await response.json()
                    logger.info(f"✅ JSON Response received")
                    return result
                else:
                    # For file downloads
                    text_result = await response.text()
                    logger.info(f"✅ Text Response received ({len(text_result)} chars)")
                    return {"content": text_result}

        except aiohttp.ClientError as e:
            logger.error(f"❌ OneDrive request failed: {str(e)}")
            raise OneDriveError(f"Request failed: {str(e)}")

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection and get basic user info"""
        try:
            url = f"{self.api_base}/me"

            # ADD THIS DEBUG LINE
            logger.info(f"🔍 Testing with token: {self.access_token[:20]}...")

            result = await self._make_request("GET", url)
            return {
                "success": True,
                "user": {
                    "name": result.get("displayName", ""),
                    "email": result.get("mail") or result.get("userPrincipalName", ""),
                },
            }
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_or_create_folder(self, folder_name: str = "CVs") -> str:
        """Get or create CV folder in OneDrive - FIXED VERSION"""
        logger.info(f"🔍 Looking for folder: {folder_name}")

        try:
            # Method 1: Search for folder using search endpoint (more reliable)
            search_url = f"{self.api_base}/me/drive/root/search(q='{folder_name}')"
            result = await self._make_request("GET", search_url)

            # Check if we found our folder
            if result.get("value"):
                for item in result["value"]:
                    if (
                        item.get("name") == folder_name
                        and item.get("folder")
                        and item.get("parentReference", {}).get("path")
                        == "/drive/root:"
                    ):
                        folder_id = item["id"]
                        logger.info(f"✅ Found existing folder via search: {folder_id}")
                        return folder_id

            # Method 2: List children and filter manually
            children_url = f"{self.api_base}/me/drive/root/children"
            result = await self._make_request("GET", children_url)

            if result.get("value"):
                for item in result["value"]:
                    if item.get("name") == folder_name and item.get("folder"):
                        folder_id = item["id"]
                        logger.info(f"✅ Found existing folder in root: {folder_id}")
                        return folder_id

            # Method 3: If folder doesn't exist, create it
            logger.info(f"📁 Creating new folder: {folder_name}")
            create_data = {
                "name": folder_name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "fail",  # Fail if exists rather than rename
            }

            create_url = f"{self.api_base}/me/drive/root/children"
            folder = await self._make_request("POST", create_url, json=create_data)
            folder_id = folder["id"]
            logger.info(f"✅ Created folder: {folder_id}")
            return folder_id

        except Exception as e:
            logger.error(f"❌ Folder operation failed: {e}")
            # If creation fails due to conflict, try to find it again
            if "nameAlreadyExists" in str(e) or "itemAlreadyExists" in str(e):
                logger.info("🔄 Folder might already exist, trying to find it...")
                # Retry finding the folder
                children_url = f"{self.api_base}/me/drive/root/children"
                result = await self._make_request("GET", children_url)

                if result.get("value"):
                    for item in result["value"]:
                        if item.get("name") == folder_name and item.get("folder"):
                            folder_id = item["id"]
                            logger.info(f"✅ Found folder after conflict: {folder_id}")
                            return folder_id

            raise OneDriveError(f"Failed to get/create folder: {e}")

    async def verify_folder_id(self, folder_id: str) -> bool:
        """Verify if a folder ID is valid and accessible"""
        try:
            url = f"{self.api_base}/me/drive/items/{folder_id}"
            result = await self._make_request("GET", url)
            return result.get("folder") is not None
        except Exception:
            return False

    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in OneDrive folder - FIXED VERSION"""
        try:
            # First get the folder ID
            folder_id = await self._get_or_create_folder(folder_name)
            logger.info(f"✅ Using folder ID for listing: {folder_id}")

            # Verify the folder ID is valid
            if not await self.verify_folder_id(folder_id):
                logger.warning(
                    f"⚠️ Folder ID {folder_id} is invalid, recreating folder..."
                )
                # If invalid, try to recreate the folder
                folder_id = await self._get_or_create_folder(folder_name)
                logger.info(f"✅ Using new folder ID: {folder_id}")

            # Use folder ID based API call
            url = f"{self.api_base}/me/drive/items/{folder_id}/children"
            params = {
                "$filter": "file ne null",  # Only get files, not folders
                "$orderby": "lastModifiedDateTime desc",
            }

            result = await self._make_request("GET", url, params=params)

            files = []
            for file_data in result.get("value", []):
                try:
                    # Skip folders, only process files
                    if file_data.get("folder"):
                        continue

                    # Only include JSON files for CVs
                    if not file_data["name"].endswith(".json"):
                        continue

                    files.append(
                        CloudFileMetadata(
                            file_id=file_data["id"],
                            name=file_data["name"],
                            provider=CloudProvider.ONEDRIVE,
                            created_at=datetime.fromisoformat(
                                file_data["createdDateTime"].replace("Z", "+00:00")
                            ),
                            last_modified=datetime.fromisoformat(
                                file_data["lastModifiedDateTime"].replace("Z", "+00:00")
                            ),
                            size_bytes=file_data.get("size", 0),
                        )
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse file data: {e}")
                    continue

            logger.info(f"✅ Found {len(files)} files using folder ID access")
            return files

        except Exception as e:
            logger.error(f"❌ List files failed: {e}")
            # For 404 errors, try to recreate the folder on next attempt
            if "404" in str(e) or "itemNotFound" in str(e):
                logger.info(f"📁 Folder not found, returning empty list: {folder_name}")
                return []  # Return empty list for not found folder
            raise OneDriveError(f"Failed to list files: {e}")

    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload CV file to OneDrive"""
        try:
            logger.info(f"📤 Starting upload for file: {file_name}")

            folder_id = await self._get_or_create_folder(folder_name)
            logger.info(f"📁 Folder ID retrieved: {folder_id}")

            # Use OneDrive simple upload for files < 4MB
            upload_url = (
                f"{self.api_base}/me/drive/items/{folder_id}:/{file_name}:/content"
            )

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            logger.info(f"🔄 Making upload request to OneDrive")
            async with self.session.put(
                upload_url, headers=headers, data=content.encode("utf-8")
            ) as response:
                logger.info(f"📊 Upload response status: {response.status}")

                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"❌ Upload failed: {error_text}")
                    raise OneDriveError(f"Upload failed: {error_text}")

                result = await response.json()
                file_id = result["id"]
                logger.info(f"✅ File uploaded successfully: {file_id}")
                return file_id

        except Exception as e:
            logger.error(f"❌ File upload failed: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise OneDriveError(f"Failed to upload file: {e}")

    async def download_file(self, file_id: str) -> str:
        """Download file content from OneDrive"""
        try:
            logger.info(f"📥 Downloading file: {file_id}")

            url = f"{self.api_base}/me/drive/items/{file_id}/content"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            logger.info(f"🔄 Making download request to: {url}")

            async with self.session.get(url, headers=headers) as response:
                logger.info(f"📊 Download response status: {response.status}")

                if response.status == 401:
                    error_text = await response.text()
                    logger.error(f"❌ OneDrive token expired: {error_text}")
                    raise OneDriveError("Access token expired or invalid")
                elif response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"❌ OneDrive API error {response.status}: {error_text}"
                    )
                    raise OneDriveError(f"API error {response.status}: {error_text}")

                # Get the raw file content
                content = await response.text()
                logger.info(f"✅ File downloaded successfully ({len(content)} chars)")

                return content

        except Exception as e:
            logger.error(f"❌ File download failed: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            raise OneDriveError(f"Failed to download file: {e}")

    async def update_file(self, file_id: str, content: str) -> bool:
        """Update an existing file in OneDrive"""
        try:
            logger.info(f"📝 Updating file in OneDrive: {file_id}")

            # Use the content endpoint to update file content
            url = f"{self.api_base}/me/drive/items/{file_id}/content"

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            async with self.session.put(
                url, headers=headers, data=content.encode("utf-8")
            ) as response:
                logger.info(f"📊 Update response status: {response.status}")

                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"❌ Update failed: {error_text}")
                    return False

                logger.info(f"✅ File updated successfully: {file_id}")
                return True

        except Exception as e:
            logger.error(f"❌ File update failed: {e}")
            return False

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file from OneDrive"""
        try:
            logger.info(f"🗑️ Deleting file from OneDrive: {file_id}")

            url = f"{self.api_base}/me/drive/items/{file_id}"

            await self._make_request("DELETE", url)

            logger.info(f"✅ File deleted successfully: {file_id}")
            return True

        except OneDriveError as e:
            logger.error(f"❌ Failed to delete file {file_id}: {e}")
            if "404" in str(e) or "ItemNotFound" in str(e):
                # File already doesn't exist, consider it a success
                logger.info(f"📝 File {file_id} was already deleted or doesn't exist")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error deleting file {file_id}: {e}")
            return False

    async def get_storage_quota(self) -> Optional[Dict[str, int]]:
        """Get OneDrive storage quota information"""
        try:
            url = f"{self.api_base}/me/drive"

            result = await self._make_request("GET", url)

            quota = result.get("quota", {})
            if quota:
                total = quota.get("total", 0)
                used = quota.get("used", 0)
                return {
                    "total": total,
                    "used": used,
                    "available": total - used,
                }
            return None

        except Exception as e:
            logger.warning(f"Failed to get storage quota: {e}")
            return None

    async def test_basic_connection(self) -> Dict[str, Any]:
        """Test with the most basic endpoint"""
        try:
            # Try the simplest possible endpoint
            url = "https://graph.microsoft.com/v1.0/me/drive"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "drive_id": result.get("id")}
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Status {response.status}: {error_text}",
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}


class OneDriveOAuth:
    """Handle OneDrive OAuth flow"""

    def __init__(self):
        self.client_id = settings.microsoft_client_id
        self.client_secret = settings.microsoft_client_secret
        self.redirect_uri = settings.microsoft_redirect_uri

    def get_auth_url(self, state: str) -> str:
        """Generate Microsoft OAuth URL"""
        base_url = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "https://graph.microsoft.com/Files.ReadWrite https://graph.microsoft.com/User.Read offline_access",  # Full scope URLs
            "response_type": "code",
            "state": state,
        }

        auth_url = f"{base_url}?{urlencode(params)}"
        logger.info(f"🔗 Generated OneDrive OAuth URL")
        logger.info(f"🔗 Redirect URI: {self.redirect_uri}")
        return auth_url

    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "scope": "https://graph.microsoft.com/Files.ReadWrite https://graph.microsoft.com/User.Read offline_access",
        }

        logger.info(f"🔄 Exchanging code for tokens...")

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Token exchange failed: {error_text}")
                    raise OneDriveError(f"Token exchange failed: {response.status}")

                token_data = await response.json()

                # Calculate expiry time
                expires_in = token_data.get("expires_in", 3600)
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                result = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_at": expires_at.isoformat(),
                    "expires_in": expires_in,
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope", ""),
                }

                logger.info(f"✅ Tokens exchanged successfully")
                return result

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/Files.ReadWrite offline_access",
        }

        logger.info(f"🔄 Refreshing token...")

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Token refresh failed: {error_text}")
                    raise OneDriveError(f"Token refresh failed: {response.status}")

                token_data = await response.json()

                # Calculate expiry time
                expires_in = token_data.get("expires_in", 3600)
                expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

                result = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token", refresh_token),
                    "expires_at": expires_at.isoformat(),
                    "expires_in": expires_in,
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope", ""),
                }

                logger.info(f"✅ Token refreshed successfully")
                return result


# Factory function for backwards compatibility
def get_onedrive_provider(access_token: str) -> OneDriveProvider:
    """Factory function to create OneDrive provider"""
    return OneDriveProvider(access_token)


# OAuth instance
onedrive_oauth = OneDriveOAuth()
