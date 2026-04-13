"""Shared base classes for generator models.

Every concrete generator model must implement two methods:
- load_model(): build and return the provider-specific model client/object
- inference(...): run generation and return output text as a string
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BaseModelConfig:
	"""Default configs shared across model wrappers."""
	max_new_tokens: int = 512

	system_prompt: str = """Your task is to generate an image given a piece of text such that the resulting text–image pair expresses sarcasm through synergy.

Definition of synergistic sarcasm:
- The text alone should NOT fully communicate the sarcastic meaning.
- The image alone should NOT fully communicate the sarcastic meaning.
- The sarcastic intent should emerge only when the text and image are interpreted together.

Requirements:
1. Preserve the intended situation and topic of the text.
2. Do NOT make the image sarcastic on its own.
3. Avoid redundancy with the text.
4. Introduce subtle visual context that creates cross-modal contrast.
5. Ensure the image requires the text to interpret the sarcasm.
6. Use realistic and coherent scenes.
7. Do NOT introduce unrelated elements.
8. Prefer understatement over exaggeration.

Generate an image that satisfies the above constraints."""
	user_prompt: str = """Generate an image that, when combined with the following text, expresses sarcasm through cross-modal interaction.

Goal:
- The image alone must be non-sarcastic.
- Do NOT repeat or directly illustrate the text.
- The sarcasm must arise only when text and image are combined.
- Introduce subtle contextual contrast with the text.
- Keep the image relevant, realistic, and coherent.
- Avoid obvious or exaggerated sarcasm cues.

Text:
{text}"""
	device_map: str = "auto"
	torch_dtype: str = "auto"
	temperature: float = 0.0


class BaseModel(ABC):
	"""Abstract parent class for all generator model wrappers."""

	def __init__(self, config: BaseModelConfig) -> None:
		self.config = config
		self.model = self.load_model()

    # force all child classes to implement these methods: load_model and inference
	@abstractmethod
	def load_model(self) -> Any:
		"""Load and return the provider-specific model object."""

	@abstractmethod
	def inference(self, text: str | None = None, image: str | Path | None = None) -> str:
		"""Run inference and return output text."""
