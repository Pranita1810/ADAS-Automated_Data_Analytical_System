"""
ADAS AI Decision Service Package.
Provides natural language executive querying and standalone HTML report generation.
"""

from .ai_service import AIService, get_ai_service
from .report_generator import generate_executive_html_report

__all__ = [
    "AIService",
    "get_ai_service",
    "generate_executive_html_report",
]
