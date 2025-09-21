# app/cloud/dropbox.py
"""
Dropbox provider - Following the same pattern as Google Drive
"""

import json
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

from ..config import get_settings
from ..schemas import CloudProvider, CloudFileMetadata, CompleteCV

logger = logging.getLogger(__name__)
settings = get_settings()


class DropboxError(Exception):
    """Dropbox specific errors"""

    pass


class DropboxProvider:
    """Dropbox integration - Following Google Drive pattern"""

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.api_base = "https://api.dropboxapi.com/2"
        self.content_api_base = "https://content.dropboxapi.com/2"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Dropbox API"""
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

        logger.info(f"🔄 Dropbox API Request: {method} {url}")

        try:
            async with self.session.request(method, url, **kwargs) as response:
                logger.info(f"📊 Response Status: {response.status}")

                if response.status == 401:
                    error_text = await response.text()
                    logger.error(f"❌ Dropbox token expired: {error_text}")
                    raise DropboxError("Access token expired or invalid")
                elif response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"❌ Dropbox API error {response.status}: {error_text}"
                    )
                    raise DropboxError(f"API error {response.status}: {error_text}")

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
            logger.error(f"❌ Dropbox request failed: {str(e)}")
            raise DropboxError(f"Request failed: {str(e)}")

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection and get basic user info"""
        try:
            url = f"{self.api_base}/users/get_current_account"

            result = await self._make_request("POST", url)

            return {
                "success": True,
                "user": {
                    "name": result.get("name", {}).get("display_name", ""),
                    "email": result.get("email", ""),
                },
            }
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return {"success": False, "error": str(e)}

    async def _ensure_folder_exists(self, folder_path: str):
        """Ensure folder exists, create if it doesn't"""
        try:
            url = f"{self.api_base}/files/create_folder_v2"
            data = {"path": folder_path, "autorename": False}

            await self._make_request("POST", url, json=data)
            logger.info(f"✅ Created folder: {folder_path}")
        except DropboxError as e:
            if "conflict" in str(e).lower():
                # Folder already exists
                logger.info(f"📁 Folder already exists: {folder_path}")
            else:
                logger.warning(f"⚠️ Failed to create folder {folder_path}: {e}")

    async def list_files(self, folder_name: str = "CVs") -> List[CloudFileMetadata]:
        """List CV files in Dropbox folder"""
        try:
            folder_path = f"/{folder_name}"
            logger.info(f"📋 Listing files in folder: {folder_path}")

            url = f"{self.api_base}/files/list_folder"
            data = {"path": folder_path, "recursive": False}

            try:
                result = await self._make_request("POST", url, json=data)
            except DropboxError as e:
                if "not_found" in str(e).lower():
                    # Folder doesn't exist, return empty list
                    logger.info(
                        f"📁 Folder {folder_path} doesn't exist, returning empty list"
                    )
                    return []
                raise

            files = []
            for entry in result.get("entries", []):
                if (
                    entry.get(".tag") == "file"
                    and entry["name"].endswith(".json")
                    and "cv_" in entry["name"].lower()
                ):
                    try:
                        files.append(
                            CloudFileMetadata(
                                file_id=entry[
                                    "path_display"
                                ],  # Dropbox uses paths as IDs
                                name=entry["name"],
                                provider=CloudProvider.DROPBOX,
                                created_at=datetime.fromisoformat(
                                    entry["client_modified"].replace("Z", "+00:00")
                                ),
                                last_modified=datetime.fromisoformat(
                                    entry["server_modified"].replace("Z", "+00:00")
                                ),
                                size_bytes=entry.get("size", 0),
                            )
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to parse file data: {e}")
                        continue

            logger.info(f"✅ Found {len(files)} files")
            return files

        except Exception as e:
            logger.error(f"❌ List files failed: {e}")
            raise DropboxError(f"Failed to list files: {e}")

    async def upload_file(
        self, file_name: str, content: str, folder_name: str = "CVs"
    ) -> str:
        """Upload CV file to Dropbox"""
        try:
            logger.info(f"📤 Starting upload for file: {file_name}")

            folder_path = f"/{folder_name}"
            file_path = f"{folder_path}/{file_name}"

            # Ensure folder exists
            await self._ensure_folder_exists(folder_path)

            # Upload file
            url = f"{self.content_api_base}/files/upload"

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/octet-stream",
                "Dropbox-API-Arg": json.dumps(
                    {"path": file_path, "mode": "overwrite", "autorename": True}
                ),
            }

            async with self.session.post(
                url, headers=headers, data=content.encode("utf-8")
            ) as response:
                logger.info(f"📊 Upload response status: {response.status}")

                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"❌ Upload failed: {error_text}")
                    raise DropboxError(f"Upload failed: {error_text}")

                result = await response.json()
                file_id = result["path_display"]
                logger.info(f"✅ File uploaded successfully: {file_id}")
                return file_id

        except Exception as e:
            logger.error(f"❌ File upload failed: {e}")
            raise DropboxError(f"Failed to upload file: {e}")

    async def download_file(self, file_id: str) -> str:
        """Download file content from Dropbox"""
        try:
            logger.info(f"📥 Downloading file: {file_id}")

            url = f"{self.content_api_base}/files/download"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Dropbox-API-Arg": json.dumps({"path": file_id}),
            }

            async with self.session.post(url, headers=headers) as response:
                logger.info(f"📊 Download response status: {response.status}")

                if response.status == 401:
                    error_text = await response.text()
                    logger.error(f"❌ Dropbox token expired: {error_text}")
                    raise DropboxError("Access token expired or invalid")
                elif response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"❌ Dropbox API error {response.status}: {error_text}"
                    )
                    raise DropboxError(f"API error {response.status}: {error_text}")

                content = await response.text()
                logger.info(f"✅ File downloaded successfully ({len(content)} chars)")
                return content

        except Exception as e:
            logger.error(f"❌ File download failed: {e}")
            raise DropboxError(f"Failed to download file: {e}")

    async def update_file(self, file_id: str, content: str) -> bool:
        """Update an existing file in Dropbox"""
        try:
            logger.info(f"📝 Updating file in Dropbox: {file_id}")

            url = f"{self.content_api_base}/files/upload"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/octet-stream",
                "Dropbox-API-Arg": json.dumps({"path": file_id, "mode": "overwrite"}),
            }

            async with self.session.post(
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
        """Delete a file from Dropbox"""
        try:
            logger.info(f"🗑️ Deleting file from Dropbox: {file_id}")

            url = f"{self.api_base}/files/delete_v2"
            data = {"path": file_id}

            await self._make_request("POST", url, json=data)

            logger.info(f"✅ File deleted successfully: {file_id}")
            return True

        except DropboxError as e:
            logger.error(f"❌ Failed to delete file {file_id}: {e}")
            if "not_found" in str(e).lower():
                # File already doesn't exist, consider it a success
                logger.info(f"📝 File {file_id} was already deleted or doesn't exist")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error deleting file {file_id}: {e}")
            return False

    async def get_storage_quota(self) -> Optional[Dict[str, int]]:
        """Get Dropbox storage quota information"""
        try:
            url = f"{self.api_base}/users/get_space_usage"

            result = await self._make_request("POST", url)

            used = result.get("used", 0)
            allocation = result.get("allocation", {})

            if allocation.get(".tag") == "individual":
                total = allocation.get("allocated", 0)
            else:
                total = 0

            return {
                "total": total,
                "used": used,
                "available": max(0, total - used),
            }

        except Exception as e:
            logger.warning(f"Failed to get storage quota: {e}")
            return None


class DropboxOAuth:
    """Handle Dropbox OAuth flow"""

    def __init__(self):
        self.client_id = settings.dropbox_app_key
        self.client_secret = settings.dropbox_app_secret
        self.redirect_uri = settings.dropbox_redirect_uri

    def get_auth_url(self, state: str) -> str:
        """Generate Dropbox OAuth URL"""
        base_url = "https://www.dropbox.com/oauth2/authorize"

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
            "token_access_type": "offline",  # Get refresh token
        }

        auth_url = f"{base_url}?{urlencode(params)}"
        logger.info(f"🔗 Generated Dropbox OAuth URL")
        return auth_url

    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        token_url = "https://api.dropboxapi.com/oauth2/token"

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }

        logger.info(f"🔄 Exchanging code for tokens...")

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Token exchange failed: {error_text}")
                    raise DropboxError(f"Token exchange failed: {response.status}")

                token_data = await response.json()

                # Dropbox tokens don't expire by default
                result = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token"),
                    "expires_at": None,  # Dropbox tokens don't expire
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope", ""),
                }

                logger.info(f"✅ Tokens exchanged successfully")
                return result

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token (Dropbox tokens typically don't expire)"""
        if not refresh_token:
            raise DropboxError("No refresh token available")

        token_url = "https://api.dropboxapi.com/oauth2/token"

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        logger.info(f"🔄 Refreshing token...")

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Token refresh failed: {error_text}")
                    raise DropboxError(f"Token refresh failed: {response.status}")

                token_data = await response.json()

                result = {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data.get("refresh_token", refresh_token),
                    "expires_at": None,  # Dropbox tokens don't expire
                    "token_type": token_data.get("token_type", "Bearer"),
                    "scope": token_data.get("scope", ""),
                }

                logger.info(f"✅ Token refreshed successfully")
                return result


# Factory function for backwards compatibility
def get_dropbox_provider(access_token: str) -> DropboxProvider:
    """Factory function to create Dropbox provider"""
    return DropboxProvider(access_token)


# OAuth instance
dropbox_oauth = DropboxOAuth()
