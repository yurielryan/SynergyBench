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

Your task is to rewrite the text from a sarcastic text-image pair such that the rewritten text becomes synergistic with the accompanying image.

Definition of synergistic:
- The rewritten text alone should NOT fully communicate the sarcastic meaning.
- The image alone should NOT fully communicate the sarcastic meaning.
- Only when BOTH text and image are combined should the sarcastic intent become understandable.

Requirements:
1. Preserve the original sarcastic meaning.
2. Remove redundant information that can be observed in the image.
3. Rewrite the text so that it references, depends on, or contrasts with visual content without explicitly stating it.
4. Keep the sarcasm natural, coherent, and fluent.
5. Do NOT invent new facts.
6. Do NOT make the text meaningless or random.
7. Prefer subtle implication, indirect phrasing, contrast, or incomplete statements that require the image for interpretation.

Respond with ONLY the rewritten text, and do NOT include any extra explanation or commentary."""
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
