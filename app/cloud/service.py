# app/cloud/service.py - UPDATED to include OneDrive support
"""
Unified cloud service that abstracts all cloud providers including OneDrive
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
import logging

from ..config import get_settings
from ..schemas import (
    CloudProvider,
    CloudFileMetadata,
    CompleteCV,
    CVFileMetadata,
    CloudSession,
    CloudConnectionStatus,
)

# Import both Google Drive and OneDrive services
from .google_drive_service import google_drive_service, GoogleDriveError
from .onedrive_service import onedrive_service, OneDriveError

logger = logging.getLogger(__name__)
settings = get_settings()


class CloudProviderError(Exception):
    """Unified cloud provider error"""

    pass


class CloudService:
    """Unified service for managing CV files across multiple cloud providers"""

    def __init__(self):
        # Use persistent encryption key from settings
        self.encryption_key = settings.get_encryption_key_bytes()
        self.cipher = Fernet(self.encryption_key)

        # Initialize provider services
        self.providers = {
            CloudProvider.GOOGLE_DRIVE: google_drive_service,
            CloudProvider.ONEDRIVE: onedrive_service,
            # Future providers can be added here
            # CloudProvider.DROPBOX: dropbox_service,
            # CloudProvider.BOX: box_service,
        }

        logger.info("CloudService initialized with Google Drive and OneDrive support")

    def _encrypt_tokens(self, tokens: Dict[str, Any]) -> str:
        """Encrypt cloud provider tokens securely"""
        try:
            tokens_json = json.dumps(tokens, default=str)
            encrypted = self.cipher.encrypt(tokens_json.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise CloudProviderError(f"Failed to encrypt tokens: {str(e)}")

    def _decrypt_tokens(self, encrypted_tokens: str) -> Dict[str, Any]:
        """Decrypt cloud provider tokens securely"""
        try:
            decrypted = self.cipher.decrypt(encrypted_tokens.encode())
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise CloudProviderError(f"Failed to decrypt tokens: {str(e)}")

    async def save_cv(
        self,
        session_tokens: Dict[str, Any],
        provider: CloudProvider,
        cv_data: CompleteCV,
        file_name: Optional[str] = None,
    ) -> str:
        """Save CV to specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        try:
            provider_service = self.providers.get(provider)
            if not provider_service:
                raise CloudProviderError(f"Provider {provider.value} not supported yet")

            # Convert CompleteCV to dict for the service
            cv_dict = cv_data.dict() if hasattr(cv_data, "dict") else cv_data

            file_id = await provider_service.save_cv(
                session_tokens[provider.value], cv_dict
            )
            logger.info(f"CV saved to {provider.value}: {file_id}")
            return file_id

        except (GoogleDriveError, OneDriveError) as e:
            logger.error(f"Provider error saving CV to {provider.value}: {e}")
            raise CloudProviderError(f"Failed to save CV to {provider.value}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error saving CV to {provider.value}: {e}")
            raise CloudProviderError(f"Failed to save CV: {str(e)}")

    async def load_cv(
        self, session_tokens: Dict[str, Any], provider: CloudProvider, file_id: str
    ) -> CompleteCV:
        """Load CV from specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        try:
            provider_service = self.providers.get(provider)
            if not provider_service:
                raise CloudProviderError(f"Provider {provider.value} not supported yet")

            cv_data = await provider_service.load_cv(
                session_tokens[provider.value], file_id
            )
            logger.info(f"CV loaded from {provider.value}: {file_id}")
            return cv_data

        except (GoogleDriveError, OneDriveError) as e:
            logger.error(f"Provider error loading CV from {provider.value}: {e}")
            raise CloudProviderError(
                f"Failed to load CV from {provider.value}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error loading CV from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to load CV: {str(e)}")

    async def list_cvs(
        self, session_tokens: Dict[str, Any], provider: CloudProvider
    ) -> List[CloudFileMetadata]:
        """List all CVs from specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        try:
            provider_service = self.providers.get(provider)
            if not provider_service:
                raise CloudProviderError(f"Provider {provider.value} not supported yet")

            files = await provider_service.list_cvs(session_tokens[provider.value])
            logger.info(f"Listed {len(files)} CVs from {provider.value}")
            return files

        except (GoogleDriveError, OneDriveError) as e:
            logger.error(f"Provider error listing CVs from {provider.value}: {e}")
            raise CloudProviderError(
                f"Failed to list CVs from {provider.value}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error listing CVs from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to list CVs: {str(e)}")

    async def delete_cv(
        self, session_tokens: Dict[str, Any], provider: CloudProvider, file_id: str
    ) -> bool:
        """Delete CV from specified cloud provider"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No access token for {provider.value}")

        try:
            provider_service = self.providers.get(provider)
            if not provider_service:
                raise CloudProviderError(f"Provider {provider.value} not supported yet")

            success = await provider_service.delete_cv(
                session_tokens[provider.value], file_id
            )

            if success:
                logger.info(f"CV deleted from {provider.value}: {file_id}")
            else:
                logger.warning(f"Failed to delete CV from {provider.value}: {file_id}")

            return success

        except (GoogleDriveError, OneDriveError) as e:
            logger.error(f"Provider error deleting CV from {provider.value}: {e}")
            raise CloudProviderError(
                f"Failed to delete CV from {provider.value}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error deleting CV from {provider.value}: {e}")
            raise CloudProviderError(f"Failed to delete CV: {str(e)}")

    async def get_connection_status(
        self, session_tokens: Dict[str, Any], provider: CloudProvider
    ) -> CloudConnectionStatus:
        """Get connection status for a cloud provider"""

        if provider.value not in session_tokens:
            return CloudConnectionStatus(provider=provider, connected=False)

        try:
            provider_service = self.providers.get(provider)
            if not provider_service:
                return CloudConnectionStatus(
                    provider=provider,
                    connected=False,
                    error=f"Provider {provider.value} not supported yet",
                )

            status = await provider_service.get_connection_status(
                session_tokens[provider.value]
            )
            return status

        except Exception as e:
            logger.warning(f"Connection check failed for {provider.value}: {e}")
            return CloudConnectionStatus(
                provider=provider, connected=False, error=str(e)
            )

    async def get_all_connection_statuses(
        self, session_tokens: Dict[str, Any]
    ) -> List[CloudConnectionStatus]:
        """Get connection status for all supported cloud providers"""

        statuses = []

        # Check all supported providers
        for provider in [CloudProvider.GOOGLE_DRIVE, CloudProvider.ONEDRIVE]:
            status = await self.get_connection_status(session_tokens, provider)
            statuses.append(status)

        # Add unsupported providers as disconnected
        for provider in [CloudProvider.DROPBOX, CloudProvider.BOX]:
            statuses.append(
                CloudConnectionStatus(
                    provider=provider,
                    connected=False,
                    error="Provider not yet implemented",
                )
            )

        return statuses

    async def ensure_valid_tokens(
        self, session_tokens: Dict[str, Any], provider: CloudProvider
    ) -> Dict[str, Any]:
        """Ensure tokens are valid for a provider, refresh if necessary"""

        if provider.value not in session_tokens:
            raise CloudProviderError(f"No tokens for {provider.value}")

        try:
            provider_service = self.providers.get(provider)
            if not provider_service:
                raise CloudProviderError(f"Provider {provider.value} not supported yet")

            # Use the provider service's ensure_valid_token method
            valid_tokens = await provider_service.ensure_valid_token(
                session_tokens[provider.value]
            )
            return valid_tokens

        except (GoogleDriveError, OneDriveError) as e:
            logger.error(f"Token validation failed for {provider.value}: {e}")
            raise CloudProviderError(f"Token validation failed: {str(e)}")
        except Exception as e:
            logger.error(
                f"Unexpected error validating tokens for {provider.value}: {e}"
            )
            raise CloudProviderError(f"Token validation failed: {str(e)}")

    async def test_connection(
        self, session_tokens: Dict[str, Any], provider: CloudProvider
    ) -> Dict[str, Any]:
        """Test connection to a specific provider"""

        if provider.value not in session_tokens:
            return {
                "success": False,
                "error": f"No tokens for {provider.value}",
                "provider": provider.value,
            }

        try:
            provider_service = self.providers.get(provider)
            if not provider_service:
                return {
                    "success": False,
                    "error": f"Provider {provider.value} not supported yet",
                    "provider": provider.value,
                }

            result = await provider_service.test_connection(
                session_tokens[provider.value]
            )
            return result

        except (GoogleDriveError, OneDriveError) as e:
            logger.error(f"Connection test failed for {provider.value}: {e}")
            return {"success": False, "error": str(e), "provider": provider.value}
        except Exception as e:
            logger.error(f"Unexpected error testing {provider.value}: {e}")
            return {"success": False, "error": str(e), "provider": provider.value}

    def get_supported_providers(self) -> List[CloudProvider]:
        """Get list of currently supported providers"""
        return list(self.providers.keys())

    def is_provider_supported(self, provider: CloudProvider) -> bool:
        """Check if a provider is supported"""
        return provider in self.providers

    async def search_cvs(
        self,
        session_tokens: Dict[str, Any],
        search_term: str,
        providers: Optional[List[CloudProvider]] = None,
    ) -> List[CloudFileMetadata]:
        """Search for CVs across multiple providers"""

        if providers is None:
            # Use only supported providers that have tokens
            providers = [
                provider
                for provider in self.get_supported_providers()
                if provider.value in session_tokens
            ]

        all_files = []

        for provider in providers:
            if not self.is_provider_supported(provider):
                logger.warning(f"Skipping unsupported provider: {provider.value}")
                continue

            try:
                files = await self.list_cvs(session_tokens, provider)
                # Filter files by search term
                matching_files = [
                    file for file in files if search_term.lower() in file.name.lower()
                ]
                all_files.extend(matching_files)
            except Exception as e:
                logger.warning(f"Search failed for {provider.value}: {e}")
                continue

        # Sort by last modified date
        all_files.sort(key=lambda x: x.last_modified, reverse=True)
        return all_files

    async def backup_cv(
        self,
        session_tokens: Dict[str, Any],
        source_provider: CloudProvider,
        file_id: str,
        backup_providers: List[CloudProvider],
    ) -> Dict[CloudProvider, str]:
        """Backup CV to multiple cloud providers"""

        # Load CV from source
        cv_data = await self.load_cv(session_tokens, source_provider, file_id)

        # Save to backup providers
        backup_results = {}

        for provider in backup_providers:
            if provider == source_provider:
                continue

            if not self.is_provider_supported(provider):
                logger.warning(
                    f"Backup to unsupported provider skipped: {provider.value}"
                )
                backup_results[provider] = None
                continue

            try:
                backup_file_id = await self.save_cv(
                    session_tokens,
                    provider,
                    cv_data,
                )
                backup_results[provider] = backup_file_id
                logger.info(f"CV backed up to {provider.value}: {backup_file_id}")
            except Exception as e:
                logger.error(f"Backup to {provider.value} failed: {e}")
                backup_results[provider] = None

        return backup_results

    async def sync_cv_across_providers(
        self,
        session_tokens: Dict[str, Any],
        cv_data: CompleteCV,
        providers: List[CloudProvider],
    ) -> Dict[CloudProvider, str]:
        """Sync CV across multiple cloud providers"""

        results = {}

        for provider in providers:
            if not self.is_provider_supported(provider):
                logger.warning(
                    f"Sync to unsupported provider skipped: {provider.value}"
                )
                results[provider] = None
                continue

            try:
                file_id = await self.save_cv(session_tokens, provider, cv_data)
                results[provider] = file_id
                logger.info(f"CV synced to {provider.value}: {file_id}")
            except Exception as e:
                logger.error(f"Sync to {provider.value} failed: {e}")
                results[provider] = None

        return results

    def validate_cv_data(self, cv_data: Dict[str, Any]) -> bool:
        """Validate CV data structure"""
        try:
            CompleteCV.parse_obj(cv_data)
            return True
        except Exception as e:
            logger.warning(f"CV validation failed: {e}")
            return False

    async def get_provider_health(
        self, provider: CloudProvider, session_tokens: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check health of a specific cloud provider"""
        health_info = {
            "provider": provider.value,
            "status": "unknown",
            "response_time_ms": None,
            "supported": self.is_provider_supported(provider),
            "has_tokens": provider.value in session_tokens,
            "error": None,
        }

        if not self.is_provider_supported(provider):
            health_info.update(
                {
                    "status": "unsupported",
                    "error": f"Provider {provider.value} not yet implemented",
                }
            )
            return health_info

        if provider.value not in session_tokens:
            health_info.update(
                {"status": "disconnected", "error": "No access tokens available"}
            )
            return health_info

        try:
            start_time = datetime.utcnow()

            # Test connection
            connection_result = await self.test_connection(session_tokens, provider)

            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000

            if connection_result.get("success"):
                health_info.update(
                    {
                        "status": "healthy",
                        "response_time_ms": round(response_time, 2),
                        "user_info": connection_result.get("user"),
                    }
                )
            else:
                health_info.update(
                    {
                        "status": "unhealthy",
                        "error": connection_result.get("error"),
                        "response_time_ms": round(response_time, 2),
                    }
                )

        except Exception as e:
            health_info.update({"status": "unhealthy", "error": str(e)})

        return health_info

    async def get_oauth_url(self, provider: CloudProvider, state: str) -> str:
        """Get OAuth URL for a provider"""
        if not self.is_provider_supported(provider):
            raise CloudProviderError(f"Provider {provider.value} not supported yet")

        provider_service = self.providers[provider]
        return provider_service.get_oauth_url(state)

    async def exchange_code_for_tokens(
        self, provider: CloudProvider, code: str
    ) -> Dict[str, Any]:
        """Exchange OAuth code for tokens"""
        if not self.is_provider_supported(provider):
            raise CloudProviderError(f"Provider {provider.value} not supported yet")

        provider_service = self.providers[provider]
        return await provider_service.exchange_code_for_tokens(code)


# Global cloud service instance
cloud_service = CloudService()
