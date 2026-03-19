"""OpenRouter-backed evaluator model using the OpenAI Python SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from .base_model_evaluation import BaseModel, BaseModelConfig # from base_model_evaluation.py
from .utils import build_base64_image_content # from utils.py


@dataclass
class OpenAIModelConfig(BaseModelConfig):
	"""Config for OpenRouter calls through the OpenAI SDK."""

	model_id: str = "openai/gpt-5-mini"
	api_key: str | None = None
	base_url: str = "https://openrouter.ai/api/v1"
	timeout: float = 60.0
	reasoning_enabled: bool = False # additional parameter to control reasoning.


class OpenAIEvaluatorModel(BaseModel):
	"""Evaluator wrapper that calls OpenRouter via OpenAI chat completions."""

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
        # 2a) no image
		if image is None: # if no image, just add text as user message (if text is None, content will be empty string)
			messages.append({"role": "user", "content": text})
			return messages

        # 2b) have image
		user_content: list[dict[str, Any]] = []
		if text is not None:
			user_content.append({"type": "text", "text": text,})

		user_content.append(build_base64_image_content(image)) # this generic help function is implemented in utils.py.

		messages.append({"role": "user", "content": user_content})
		return messages

	def evaluate(self, text: str | None = None, image: str | Path | None = None) -> str:
		messages = self._build_messages(text=text, image=image)
		reasoning_flag = "true" if self.config.reasoning_enabled else "false"

        # see docs: https://openrouter.ai/docs/quickstart#using-the-openai-sdk
		response = self.model.chat.completions.create(
			model=self.config.model_id,
			messages=messages,
			max_tokens=self.config.max_new_tokens,
			temperature=self.config.temperature,
            # disabling reasoning cus it eats up max tokens. see docs: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
			extra_body={
				"reasoning": {
					"enabled": reasoning_flag,
				},
			},
		)

		message = response.choices[0].message
		content = getattr(message, "content", "")

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


def load_openai_evaluator(
	model_id: str = "openai/gpt-5-mini",
	api_key: str | None = None,
	base_url: str = "https://openrouter.ai/api/v1",
	max_new_tokens: int = 16,
	system_prompt: str | None = None,
	temperature: float = 0.0,
	timeout: float = 60.0,
	reasoning_enabled: bool = False,
) -> OpenAIEvaluatorModel:
	"""Convenience loader used by config-driven pipelines."""

	config = OpenAIModelConfig(
		model_id=model_id,
		api_key=api_key,
		base_url=base_url,
		max_new_tokens=max_new_tokens,
		system_prompt=system_prompt
		or "You are a sarcasm classifier. Answer with exactly one token: yes or no.",
		temperature=temperature,
		timeout=timeout,
		reasoning_enabled=reasoning_enabled,
	)
	return OpenAIEvaluatorModel(config)
