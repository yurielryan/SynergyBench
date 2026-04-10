"""OpenRouter-backed evaluator model using the OpenAI Python SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .base_model_generative import BaseModel, BaseModelConfig
from .utils import build_base64_image_content


@dataclass
class OpenAIModelConfig(BaseModelConfig):
	"""Config for OpenRouter calls through the OpenAI SDK."""

	model_id: str = "openai/gpt-5-mini"
	api_key: str | None = None
	base_url: str = "https://openrouter.ai/api/v1"
	timeout: float = 60.0


class OpenAIGeneratorModel(BaseModel):
	"""Generator wrapper that calls OpenRouter via OpenAI chat completions."""

	config: OpenAIModelConfig

	def load_model(self) -> OpenAI:
		api_key = (
			self.config.api_key
			or os.getenv("OPENROUTER_API_KEY")
			or os.getenv("OPENAI_API_KEY")
		)
		if not api_key:
			raise ValueError(
				"Missing API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY, "
				"or pass api_key in OpenAIModelConfig."
			)

		return OpenAI(
			base_url=self.config.base_url,
			api_key=api_key,
			timeout=self.config.timeout,
		)

	def _build_messages(self, text: str | None, image: str | Path | None) -> list[dict[str, Any]]:
		if text is None and image is None: # big problemo
			raise ValueError("At least one of text or image must be provided.")

		messages: list[dict[str, Any]] = [] # init list for messages
  
        # 1) add system prompt.
		if self.config.system_prompt: # add system prompt if provided (in config)
			messages.append({"role": "system","content": self.config.system_prompt})

        # 2) add user message
		user_content: list[dict[str, Any]] = []
		# TODO: Need to prompt engineer the user message further, current response is not rewriting the original text.
		user_content.append({"type": "text", "text": text,})

		user_content.append(build_base64_image_content(image)) # this generic help function is implemented in utils.py.

		messages.append({"role": "user", "content": user_content})
		return messages

	def _create_chat_completion(
		self,
		messages: list[dict[str, Any]],
	) -> Any:
		request_kwargs: dict[str, Any] = {
			"model": self.config.model_id,
			"messages": messages,
			"max_tokens": self.config.max_new_tokens,
			"temperature": self.config.temperature,
		}
		# On this route, reasoning cannot be disabled; use low effort to keep overhead down.
		# see https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
		request_kwargs["extra_body"] = {
			"reasoning": {
				"effort": "low", # btw, docs for openrouter reasoning tokens are outdated - gpt5 mini cannot use "none", and cannot exclude for reasoning.
				"exclude": True,
			},
		}

		response = self.model.chat.completions.create(**request_kwargs)
		choices = getattr(response, "choices", None)
		if not choices:
			error_payload: Any = None
			if hasattr(response, "model_dump"):
				response_data = response.model_dump()
				if isinstance(response_data, dict):
					error_payload = response_data.get("error")
			if error_payload:
				raise RuntimeError(f"Provider returned error payload: {error_payload}")
			raise RuntimeError("Provider returned no choices in completion response.")

		return response

	def inference(self, text: str | None = None, image: str | Path | None = None) -> str:
		messages = self._build_messages(text=text, image=image)
		response = self._create_chat_completion(messages=messages)

		message = response.choices[0].message
		content = getattr(message, "content", "")
		if content is None:
			return ""

		if isinstance(content, str):
			return content.strip()

		if isinstance(content, list): # handling multiple content parts
			chunks: list[str] = []
			for part in content:
				if isinstance(part, dict):
					text_value = part.get("text")
					if isinstance(text_value, str):
						chunks.append(text_value)
			return " ".join(chunks).strip()

		return str(content).strip()


def load_openai_generator(
	model_id: str = "openai/gpt-5-mini",
	api_key: str | None = None,
	base_url: str = "https://openrouter.ai/api/v1",
	max_new_tokens: int = 250,
	system_prompt: str | None = None,
	temperature: float = 0.0,
	timeout: float = 60.0,
) -> OpenAIGeneratorModel:
	"""Convenience loader used by config-driven pipelines."""

	config = OpenAIModelConfig(
		model_id=model_id,
		api_key=api_key,
		base_url=base_url,
		max_new_tokens=max_new_tokens,
		system_prompt=system_prompt,
		temperature=temperature,
		timeout=timeout,
	)
	return OpenAIGeneratorModel(config)
