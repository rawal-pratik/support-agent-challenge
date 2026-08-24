from __future__ import annotations

import json

import httpx

from support_agent.config import settings


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request fails."""


class OpenRouterClient:
    """Minimal OpenRouter chat-completions client."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else settings.openrouter_api_key
        )

        self.model = (
            model
            if model is not None
            else settings.openrouter_model
        )

        self.timeout = timeout

        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not configured."
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat-completion request to OpenRouter."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise OpenRouterError(
                f"OpenRouter request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise OpenRouterError(
                "OpenRouter returned "
                f"{response.status_code}: "
                f"{response.text}"
            )

        try:
            data = response.json()

            return data["choices"][0]["message"]["content"]

        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise OpenRouterError(
                f"Unexpected OpenRouter response: "
                f"{response.text}"
            ) from exc