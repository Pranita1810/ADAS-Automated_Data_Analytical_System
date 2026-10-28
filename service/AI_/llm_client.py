"""
ADAS AI - Free LLM Provider Integration Client.
Supports Google Gemini (Free Tier), Groq Cloud (Free Tier), OpenRouter (Free Tier), and OpenAI.
Falls back gracefully to local deterministic analytical intelligence if no key is provided.
"""

from __future__ import annotations
import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
import requests

from mahgaurd import get_logger, timed

logger = get_logger("ADAS.AI_LLM")


class FreeLLMClient:
    """
    Client for interacting with free-tier LLM APIs:
    - Google Gemini (gemini-1.5-flash / gemini-2.0-flash via Google AI Studio free tier)
    - Groq Cloud (llama-3.3-70b-versatile / llama-3.1-8b-instant via Groq free tier)
    - OpenRouter (meta-llama/llama-3.3-70b-instruct:free / deepseek/deepseek-r1:free)
    - OpenAI-compatible endpoints
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.provider = provider.lower()
        self.api_key = api_key or self._detect_api_key()
        if self.provider == "auto":
            self.provider = self._detect_provider()
        logger.info(f"FreeLLMClient initialized (Provider: {self.provider}, Key configured: {bool(self.api_key)})")

    def _detect_api_key(self) -> Optional[str]:
        """Check standard environment variables for available free LLM keys."""
        for var in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
            val = os.getenv(var)
            if val and val.strip():
                return val.strip()
        return None

    def _detect_provider(self) -> str:
        """Automatically select provider based on detected environment keys."""
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        elif os.getenv("GROQ_API_KEY"):
            return "groq"
        elif os.getenv("OPENROUTER_API_KEY"):
            return "openrouter"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "none"

    def is_configured(self) -> bool:
        """Returns True if a valid API key and supported provider are active."""
        return bool(self.api_key and self.provider != "none")

    @timed("FreeLLMClient.generate")
    def generate(
        self,
        prompt: str,
        system_instruction: str = "You are ADAS AI Executive Decision Copilot. Provide concise, strategic business insights grounded in the provided factual data.",
        temperature: float = 0.3,
        timeout: int = 12
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Send a prompt to the configured free LLM provider.
        Returns: (response_text, error_message)
        """
        if not self.is_configured():
            return None, "No free LLM API key configured. Using local analytical intelligence."

        try:
            if self.provider == "gemini":
                return self._call_gemini(prompt, system_instruction, temperature, timeout)
            elif self.provider == "groq":
                return self._call_groq(prompt, system_instruction, temperature, timeout)
            elif self.provider == "openrouter":
                return self._call_openrouter(prompt, system_instruction, temperature, timeout)
            elif self.provider == "openai":
                return self._call_openai(prompt, system_instruction, temperature, timeout)
            else:
                return None, f"Unknown provider: {self.provider}"
        except Exception as e:
            logger.warning(f"LLM call to {self.provider} failed: {e}")
            return None, str(e)

    def _call_gemini(self, prompt: str, system: str, temperature: float, timeout: int) -> Tuple[Optional[str], Optional[str]]:
        """Call Google Gemini 1.5 Flash Free Tier."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\nUser Question:\n{prompt}"}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 800}
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip(), None
        return None, f"Gemini API returned status {resp.status_code}: {resp.text[:150]}"

    def _call_groq(self, prompt: str, system: str, temperature: float, timeout: int) -> Tuple[Optional[str], Optional[str]]:
        """Call Groq Cloud Free Tier (Llama 3.3 70B)."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 800
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text.strip(), None
        return None, f"Groq API returned status {resp.status_code}: {resp.text[:150]}"

    def _call_openrouter(self, prompt: str, system: str, temperature: float, timeout: int) -> Tuple[Optional[str], Optional[str]]:
        """Call OpenRouter Free Tier."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 800
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text.strip(), None
        return None, f"OpenRouter API returned status {resp.status_code}: {resp.text[:150]}"

    def _call_openai(self, prompt: str, system: str, temperature: float, timeout: int) -> Tuple[Optional[str], Optional[str]]:
        """Call OpenAI-compatible endpoint."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 800
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text.strip(), None
        return None, f"OpenAI API returned status {resp.status_code}: {resp.text[:150]}"
