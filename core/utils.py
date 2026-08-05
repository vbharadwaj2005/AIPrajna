"""
DocInsight — Utility and Helper Functions.
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".docx"}


def validate_file_path(path: str | Path) -> Path:
    """Validate file existence and supported extension."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{resolved.suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return resolved
