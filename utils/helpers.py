"""
Utility helper functions for Jarvis
"""

import logging
from pathlib import Path
from typing import List, Optional
import mimetypes

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_file_type(filename: str) -> str:
    """Get file type from filename"""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type.split('/')[0]  # e.g., 'application', 'text', 'image'
    return 'unknown'


def is_supported_document(filename: str) -> bool:
    """Check if file type is supported for document processing"""
    supported_extensions = {
        '.pdf', '.txt', '.md', '.doc', '.docx',
        '.ppt', '.pptx', '.xls', '.xlsx',
        '.html', '.htm', '.rtf', '.odt',
        '.py', '.js', '.java', '.cpp', '.c',
        '.json', '.xml', '.csv', '.log'
    }
    return Path(filename).suffix.lower() in supported_extensions


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_user_id(user_id: str) -> str:
    """Sanitize user ID for safe filename usage"""
    import re
    # Remove special characters, keep alphanumeric, underscore, hyphen
    sanitized = re.sub(r'[^\w\-]', '_', user_id)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_').lower()


def validate_user_id(user_id: str) -> tuple[bool, Optional[str]]:
    """Validate user ID format"""
    if not user_id or not user_id.strip():
        return False, "User ID cannot be empty"

    if len(user_id) < 3:
        return False, "User ID must be at least 3 characters"

    if len(user_id) > 50:
        return False, "User ID must be less than 50 characters"

    # Check for valid characters
    import re
    if not re.match(r'^[\w\-]+$', user_id):
        return False, "User ID can only contain letters, numbers, underscore and hyphen"

    return True, None
