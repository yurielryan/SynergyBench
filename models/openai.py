# TODO: CHECK

"""OpenAI API wrapper implementing the benchmark inference interface.

Expected public method:
	inference(text: str | None, image: str | Path | None) -> str
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpenAIConfig:
	model_id: str = "gpt-4.1-mini"
	api_key: str | None = None
	max_new_tokens: int = 16
	system_prompt: str = (
		"You are a sarcasm classifier. Answer with exactly one token: yes or no."
	)
	timeout: float = 60.0


def _image_to_data_uri(image_path: str | Path) -> str:
	path = Path(image_path)
	mime_type, _ = mimetypes.guess_type(str(path))
	mime_type = mime_type or "image/jpeg"
	encoded = base64.b64encode(path.read_bytes()).decode("ascii")
	return f"data:{mime_type};base64,{encoded}"


class OpenAIModel:
	def __init__(self, config: OpenAIConfig) -> None:
		try:
			from openai import OpenAI
		except ImportError as exc:
			raise ImportError(
				"openai package is required for provider='openai'. Install with: pip install openai"
			) from exc

		self.config = config
		api_key = config.api_key or os.getenv("OPENAI_API_KEY")
		if not api_key:
			raise ValueError(
				"OPENAI_API_KEY is not set and model.api_key was not provided in config."
			)
		self.client = OpenAI(api_key=api_key, timeout=config.timeout)

	def inference(self, text: str | None = None, image: str | Path | None = None) -> str:
		if text is None and image is None:
			raise ValueError("Either text or image must be provided.")

		content: list[dict[str, object]] = []
		if text is not None:
			content.append({"type": "text", "text": text})
		if image is not None:
			content.append(
				{
					"type": "image_url",
					"image_url": {"url": _image_to_data_uri(image)},
				}
			)

		if text is None:
			content.insert(
				0,
				{
					"type": "text",
					"text": "Classify this image for sarcasm and answer with exactly: yes or no.",
				},
			)

		response = self.client.chat.completions.create(
			model=self.config.model_id,
			messages=[
				{"role": "system", "content": self.config.system_prompt},
				{"role": "user", "content": content},
			],
			max_tokens=self.config.max_new_tokens,
			temperature=0,
		)
		return (response.choices[0].message.content or "").strip()


def load_openai(
	model_id: str = "gpt-4.1-mini",
	api_key: str | None = None,
	max_new_tokens: int = 16,
	system_prompt: str | None = None,
	timeout: float = 60.0,
) -> OpenAIModel:
	config = OpenAIConfig(
		model_id=model_id,
		api_key=api_key,
		max_new_tokens=max_new_tokens,
		system_prompt=system_prompt
		or "You are a sarcasm classifier. Answer with exactly one token: yes or no.",
		timeout=timeout,
	)
	return OpenAIModel(config)