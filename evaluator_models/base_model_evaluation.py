"""Shared base classes for evaluator models.

Every concrete evaluator model must implement two methods:
- load_model(): build and return the provider-specific model client/object
- evaluate(...): run evaluation and return output text as a string
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BaseModelConfig:
	"""Default configs shared across model wrappers."""
	max_new_tokens: int = 16
	system_prompt: str = (
		"You are a sarcasm classifier. Answer with exactly one token: yes or no."
	)
	device_map: str = "auto"
	torch_dtype: str = "auto"
	temperature: float = 0.0


class BaseModel(ABC):
	"""Abstract parent class for all evaluator model wrappers."""

	def __init__(self, config: BaseModelConfig) -> None:
		self.config = config
		self.model = self.load_model()

    # force all child classes to implement these methods: load_model and evaluate
	@abstractmethod
	def load_model(self) -> Any:
		"""Load and return the provider-specific model object."""

	@abstractmethod
	def evaluate(self, text: str | None = None, image: str | Path | None = None) -> str:
		"""Run evaluation and return output text."""
