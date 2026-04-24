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

	model_id: str = "openai/gpt-5.4-mini"
	api_key: str | None = None
	base_url: str = "https://openrouter.ai/api/v1"
	reasoning: str = "none"
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
		if text is None or image is None:
			raise ValueError("Both text and image must be provided.")

		messages: list[dict[str, Any]] = [] # init list for messages
  
        # 1) add system prompt.
		if self.config.system_prompt: # add system prompt if provided (in config)
			messages.append({"role": "system","content": self.config.system_prompt})

        # 2) add user message
		user_content: list[dict[str, Any]] = []
		user_content.append({
			"type": "input_text",
			"text": self.config.user_prompt.format(text=text or ""),
		})

		user_content.append(build_base64_image_content(image)) # this generic help function is implemented in utils.py.

		messages.append({"role": "user", "content": user_content})
		return messages

	def _create_chat_completion(
		self,
		messages: list[dict[str, Any]],
	) -> Any:
		request_kwargs: dict[str, Any] = {
			"model": self.config.model_id,
			"input": messages,
			"max_output_tokens": self.config.max_new_tokens,
			"temperature": 1,
			"reasoning": {
				"effort": self.config.reasoning,
				"summary": "detailed",
			},
		}

		if self.config.reasoning in ['medium', 'high']:
			request_kwargs["max_output_tokens"] = max(self.config.max_new_tokens, 2048) # minimum 2048 tokens for reasoning to avoid out-of-budget errors.
			print(f"Reasoning enabled with effort '{self.config.reasoning}', setting max_tokens to {request_kwargs['max_output_tokens']} to ensure sufficient tokens for reasoning output.")

		response = self.model.responses.create(**request_kwargs)
		print(response)
		choices = getattr(response, "output", None)
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

		# message = response.choices[0].message
		message = response.output[1].content[0].text
		output = getattr(response, "output", None)

		for out in output:
			print(out)
			if out.type == "reasoning":
				reasoning = out.content
			elif out.type == "message":
				content = out.content
			else:
				raise RuntimeError(f"Unexpected output type in response: {out.type}")

		if content is None:
			content = ""
		else:
			content = content[0].text.strip()

		if reasoning is not None:
			reasoning = reasoning[0].text.strip()
		return content, reasoning

def load_openai_generator(
	model_id: str = "openai/gpt-5.4-mini",
	api_key: str | None = None,
	base_url: str = "https://openrouter.ai/api/v1",
	max_new_tokens: int = 1024,
	reasoning: str = "none",
	temperature: float = 1, # Any values other than 1 will encounter error
	timeout: float = 60.0,
) -> OpenAIGeneratorModel:
	"""Convenience loader used by config-driven pipelines."""

	config = OpenAIModelConfig(
		model_id=model_id,
		api_key=api_key,
		base_url=base_url,
		max_new_tokens=max_new_tokens,
		reasoning=reasoning,
		temperature=temperature,
		timeout=timeout,
	)
	return OpenAIGeneratorModel(config)
