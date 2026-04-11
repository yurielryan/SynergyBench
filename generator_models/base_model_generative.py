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
	# TODO: Need to further prompt engineering for the system prompt
	max_new_tokens: int = 512
	system_prompt: str = """You are an expert linguist specializing in sarcasm.

Your task is to rewrite the text from a text-image pair such that the rewritten text becomes synergistic with the accompanying image to be sarcastic.

Definition of synergistic:
- The rewritten text alone should NOT fully communicate the sarcastic meaning.
- The image alone should NOT fully communicate the sarcastic meaning.
- Only when BOTH text and image are combined should the sarcastic intent become understandable.

Requirements:
1. Preserve the original sarcastic meaning.
2. Remove redundant information that can be observed in the image.
3. Reduce explicit sarcasm cues that make the sarcasm understandable without the image.
4. Rewrite the text such that the sarcastic meaning depends on the visual content.
5. Do NOT invent new facts.
6. Do NOT make the text meaningless or random.
7. Prefer subtle implication, indirect phrasing, contrast, or incomplete statements that require the image for interpretation.

Respond with ONLY the rewritten text, and do NOT include any extra explanation or commentary."""
	user_prompt: str = """Rewrite the following text such that understanding sarcasm requires combining it with the accompanying image.

Goal:
- Remove redundant information visible in the image.
- Preserve sarcastic/ironic meaning.
- Make the text alone insufficient to understand the sarcasm.
- Ensure the sarcasm becomes clear only when text and image are viewed together.
- Ensure the sarcastic meaning becomes clear only when combined with the image.

Original Text:
{text}

Return only the rewritten text without any additional information."""
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
