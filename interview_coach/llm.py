"""LLM client for OpenAI API."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    """OpenAI API client with optional custom base URL."""
    
    def __init__(self, provider=None):
        """Initialize OpenAI client. Provider argument kept for compatibility."""
        base_url = os.getenv("OPENAI_BASE_URL")
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url if base_url else None
        )
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    def chat(self, system_prompt: str, messages: list[dict]) -> str:
        """Send a chat completion request."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
        )
        return response.choices[0].message.content
