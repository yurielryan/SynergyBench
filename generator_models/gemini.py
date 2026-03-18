# TODO: CHECK

"""Gemini API wrapper implementing the benchmark inference interface.

Expected public method:
	inference(text: str | None, image: str | Path | None) -> str
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class GeminiConfig:
	model_id: str = "gemini-2.0-flash"
	api_key: str | None = None
	max_new_tokens: int = 16
	temperature: float = 0.0
	system_prompt: str = (
		"You are a sarcasm classifier. Answer with exactly one token: yes or no."
	)


class GeminiModel:
	def __init__(self, config: GeminiConfig) -> None:
		try:
			import google.generativeai as genai
		except ImportError as exc:
			raise ImportError(
				"google-generativeai package is required for provider='gemini'. Install with: pip install google-generativeai"
			) from exc

		self.genai = genai
		self.config = config
		api_key = config.api_key or os.getenv("GEMINI_API_KEY")
		if not api_key:
			raise ValueError(
				"GEMINI_API_KEY is not set and model.api_key was not provided in config."
			)

		genai.configure(api_key=api_key)
		self.model = genai.GenerativeModel(
			model_name=config.model_id,
			system_instruction=config.system_prompt,
		)

	def inference(self, text: str | None = None, image: str | Path | None = None) -> str:
		if text is None and image is None:
			raise ValueError("Either text or image must be provided.")

		payload: list[object] = []
		if image is not None:
			payload.append(Image.open(Path(image)).convert("RGB"))
		if text is not None:
			payload.append(text)
		elif image is not None:
			payload.append(
				"Classify this image for sarcasm and answer with exactly: yes or no."
			)

		response = self.model.generate_content(
			payload,
			generation_config=self.genai.GenerationConfig(
				temperature=self.config.temperature,
				max_output_tokens=self.config.max_new_tokens,
			),
		)
		return (response.text or "").strip()


def load_gemini(
	model_id: str = "gemini-2.0-flash",
	api_key: str | None = None,
	max_new_tokens: int = 16,
	system_prompt: str | None = None,
	temperature: float = 0.0,
) -> GeminiModel:
	config = GeminiConfig(
		model_id=model_id,
		api_key=api_key,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		system_prompt=system_prompt
		or "You are a sarcasm classifier. Answer with exactly one token: yes or no.",
	)
	return GeminiModel(config)