"""
User Manager for Jarvis
Handles per-user document storage and LEANN index management
"""

import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging

from .config import get_config

logger = logging.getLogger(__name__)


class UserManager:
    """Manages user directories and indexes"""

    def __init__(self):
        self.config = get_config()
        self.users_dir = self.config.storage.users_dir
        self.users_dir.mkdir(parents=True, exist_ok=True)

    def get_user_dir(self, user_id: str) -> Path:
        """Get user directory path"""
        user_dir = self.users_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_user_documents_dir(self, user_id: str) -> Path:
        """Get user documents directory"""
        docs_dir = self.get_user_dir(user_id) / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs_dir

    def get_user_index_path(self, user_id: str) -> Path:
        """Get user LEANN index path (base path without extension)"""
        # LEANN uses base path and creates multiple files:
        # e.g., harish.index, harish.leann.meta.json, etc.
        index_path = self.get_user_dir(user_id) / user_id
        return index_path

    def get_user_metadata_path(self, user_id: str) -> Path:
        """Get user metadata file path"""
        return self.get_user_dir(user_id) / "metadata.json"

    def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        return self.get_user_dir(user_id).exists()

    def create_user(self, user_id: str, metadata: Optional[Dict] = None) -> Dict:
        """Create a new user"""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")

        user_dir = self.get_user_dir(user_id)
        metadata_path = self.get_user_metadata_path(user_id)

        # Check if metadata already exists
        if metadata_path.exists():
            logger.info(f"User {user_id} already exists")
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading metadata for {user_id}: {e}")
                # Continue to recreate metadata

        # Create directories
        self.get_user_documents_dir(user_id)

        # Create metadata
        user_metadata = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "document_count": 0,
            "index_built": False,
            "custom_metadata": metadata or {}
        }

        self._save_user_metadata(user_id, user_metadata)
        logger.info(f"Created user: {user_id}")
        return user_metadata

    def get_user_metadata(self, user_id: str) -> Dict:
        """Get user metadata"""
        metadata_path = self.get_user_metadata_path(user_id)
        if not metadata_path.exists():
            # Create default metadata if doesn't exist
            logger.info(f"Metadata not found for {user_id}, creating default")
            user_metadata = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "document_count": 0,
                "index_built": False,
                "custom_metadata": {}
            }
            self._save_user_metadata(user_id, user_metadata)
            return user_metadata

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {user_id}: {e}")
            # Return default metadata
            return {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "document_count": 0,
                "index_built": False,
                "custom_metadata": {}
            }

    def _save_user_metadata(self, user_id: str, metadata: Dict):
        """Save user metadata"""
        metadata["updated_at"] = datetime.now().isoformat()
        metadata_path = self.get_user_metadata_path(user_id)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def update_user_metadata(self, user_id: str, updates: Dict):
        """Update user metadata"""
        metadata = self.get_user_metadata(user_id)
        metadata.update(updates)
        self._save_user_metadata(user_id, metadata)

    def save_document(self, user_id: str, filename: str, content: bytes) -> Path:
        """Save a document for user"""
        docs_dir = self.get_user_documents_dir(user_id)
        file_path = docs_dir / filename

        # Handle duplicate filenames
        counter = 1
        original_stem = file_path.stem
        while file_path.exists():
            file_path = docs_dir / f"{original_stem}_{counter}{file_path.suffix}"
            counter += 1

        with open(file_path, 'wb') as f:
            f.write(content)

        # Update metadata
        metadata = self.get_user_metadata(user_id)
        metadata["document_count"] = metadata.get("document_count", 0) + 1
        metadata["index_built"] = False  # Index needs rebuild
        self._save_user_metadata(user_id, metadata)

        logger.info(f"Saved document for user {user_id}: {file_path.name}")
        return file_path

    def get_user_documents(self, user_id: str) -> List[Path]:
        """Get all documents for user"""
        docs_dir = self.get_user_documents_dir(user_id)
        if not docs_dir.exists():
            return []

        # Get all files (exclude hidden files)
        documents = [
            f for f in docs_dir.iterdir()
            if f.is_file() and not f.name.startswith('.')
        ]
        return sorted(documents, key=lambda x: x.stat().st_mtime, reverse=True)

    def delete_document(self, user_id: str, filename: str) -> bool:
        """Delete a document"""
        docs_dir = self.get_user_documents_dir(user_id)
        file_path = docs_dir / filename

        if not file_path.exists():
            logger.warning(f"Document not found: {filename}")
            return False

        file_path.unlink()

        # Update metadata
        metadata = self.get_user_metadata(user_id)
        metadata["document_count"] = max(0, metadata.get("document_count", 1) - 1)
        metadata["index_built"] = False  # Index needs rebuild
        self._save_user_metadata(user_id, metadata)

        logger.info(f"Deleted document for user {user_id}: {filename}")
        return True

    def has_index(self, user_id: str) -> bool:
        """Check if user has a built index"""
        # LEANN creates files with base name, not a directory
        # Check for the main index file or meta file
        user_dir = self.get_user_dir(user_id)
        index_file = user_dir / f"{user_id}.index"
        meta_file = user_dir / f"{user_id}.leann.meta.json"
        return index_file.exists() or meta_file.exists()

    def mark_index_built(self, user_id: str):
        """Mark that index has been built"""
        metadata = self.get_user_metadata(user_id)
        metadata["index_built"] = True
        metadata["index_updated_at"] = datetime.now().isoformat()
        self._save_user_metadata(user_id, metadata)

    def delete_index(self, user_id: str) -> bool:
        """Delete user index"""
        if not self.has_index(user_id):
            return False

        # Delete all LEANN-related files
        user_dir = self.get_user_dir(user_id)
        deleted = False

        for pattern in ['*.index', '*.leann.meta.json', '*.leann.passages.*', '*.ids.txt', '*.csr.tmp']:
            for file_path in user_dir.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    deleted = True
                    logger.info(f"Deleted: {file_path.name}")

        # Update metadata
        if deleted:
            metadata = self.get_user_metadata(user_id)
            metadata["index_built"] = False
            self._save_user_metadata(user_id, metadata)
            logger.info(f"Deleted index for user {user_id}")

        return deleted

    def delete_user(self, user_id: str) -> bool:
        """Delete user and all their data"""
        user_dir = self.get_user_dir(user_id)
        if not user_dir.exists():
            return False

        shutil.rmtree(user_dir)
        logger.info(f"Deleted user: {user_id}")
        return True

    def list_users(self) -> List[Dict]:
        """List all users"""
        users = []
        for user_dir in self.users_dir.iterdir():
            if user_dir.is_dir() and not user_dir.name.startswith('.'):
                try:
                    metadata = self.get_user_metadata(user_dir.name)
                    users.append(metadata)
                except Exception as e:
                    logger.error(f"Error reading metadata for {user_dir.name}: {e}")
                    continue

        return sorted(users, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        metadata = self.get_user_metadata(user_id)
        documents = self.get_user_documents(user_id)
        user_dir = self.get_user_dir(user_id)

        # Calculate total document size
        total_size = sum(doc.stat().st_size for doc in documents)

        # Calculate index size - LEANN creates multiple files with user_id as prefix
        # e.g., harish.index, harish.meta.json, harish.passages.jsonl, harish.ids.txt
        index_size = 0

        # Sum up all LEANN index files (NOT passages - those are source data)
        # Only count the compressed index files, not the original passages
        for pattern in [f'{user_id}.index', f'{user_id}.meta.json', f'{user_id}.ids.txt']:
            for file_path in user_dir.glob(pattern):
                if file_path.is_file():
                    index_size += file_path.stat().st_size

        return {
            "user_id": user_id,
            "document_count": len(documents),
            "total_document_size_mb": round(total_size / (1024 * 1024), 2),
            "index_size_mb": round(index_size / (1024 * 1024), 2),
            "storage_savings_percent": (
                round((1 - index_size / total_size) * 100, 1)
                if total_size > 0 else 0
            ),
            "index_built": metadata.get("index_built", False),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at")
        }


if __name__ == "__main__":
    # Test user manager
    logging.basicConfig(level=logging.INFO)

    manager = UserManager()

    # Create test user
    user_id = "test_user"
    manager.create_user(user_id)

    # Get stats
    stats = manager.get_user_stats(user_id)
    print(f"User stats: {json.dumps(stats, indent=2)}")

    # List users
    users = manager.list_users()
    print(f"\nAll users: {len(users)}")
    for user in users:
        print(f"  - {user['user_id']}: {user.get('document_count', 0)} docs")
