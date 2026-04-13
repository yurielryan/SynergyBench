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

	system_prompt: str = """Your task is to generate an image from a text-image pair such that the resulting image becomes synergistic with the accompanying text and being sarcastic.

Definition of synergistic:
- The image alone should NOT fully communicate the sarcastic meaning.
- The text alone should NOT fully communicate the sarcastic meaning.
- Only when BOTH text and image are combined should the sarcastic intent become understandable.

Requirements:
1. Preserve the original sarcastic meaning.
2. Remove visual elements that directly duplicate what is stated in the text.
3. Remove or weaken visual cues that make the sarcasm obvious without the text.
4. Ensure the sarcastic meaning depends on the text.
5. Keep the image realistic and coherent.
6. Do NOT introduce unrelated or unsupported elements.
7. Prefer subtle contrast, implication, or incomplete visual context.

Respond by generating or editing the image appropriately."""
	user_prompt: str = """Edit the following image such that understanding the sarcasm requires combining it with the accompanying text.

Goal:
- Remove visual elements that duplicate the text.
- Preserve sarcastic/ironic meaning.
- Make the image alone is insufficient to understand the sarcasm.
- Ensure the sarcastic meaning becomes clear only when combined with the image.

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
