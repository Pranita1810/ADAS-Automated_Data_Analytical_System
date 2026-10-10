import os
import secrets
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ---------------------------------------------------------------------------
# 1. Environment Configuration
# ---------------------------------------------------------------------------
# Load .env from services/config_/.env if present, otherwise fallback to root
config_env = Path(__file__).resolve().parents[3] / "config_" / ".env"
if config_env.is_file():
    load_dotenv(dotenv_path=config_env)
else:
    load_dotenv()

USERNAME = (os.getenv("AUTH_USERNAME") or "").strip("'\"")
PASSWORD = (os.getenv("AUTH_PASSWORD") or "").strip("'\"")

# ---------------------------------------------------------------------------
# 2. Security Setup
# ---------------------------------------------------------------------------
security = HTTPBasic()


def verify_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
) -> str:
    """
    Validates Basic Auth credentials against environment variables.
    Uses secrets.compare_digest to prevent timing attacks.
    """
    if not USERNAME or not PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication credentials are not configured in .env",
        )

    username_correct = secrets.compare_digest(
        credentials.username, USERNAME
    )
    password_correct = secrets.compare_digest(
        credentials.password, PASSWORD
    )

    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return True
