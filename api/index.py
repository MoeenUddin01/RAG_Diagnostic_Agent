"""Vercel serverless function entry point for FastAPI application."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.main import app

# Vercel expects a handler function
handler = app
