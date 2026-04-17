from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor

from .base_model_generative import BaseModel, BaseModelConfig
from .utils import build_base64_image_content


@dataclass
class Qwen3VLModelConfig(BaseModelConfig):
	"""Config for OpenRouter calls through the OpenAI SDK."""

	model_id: str = "Qwen/Qwen3-VL-8B-Instruct"


class Qwen3VLGeneratorModel(BaseModel):
	"""Generator wrapper that calls OpenRouter via OpenAI chat completions."""

	config: Qwen3VLModelConfig

	def load_model(self) -> Any:
		if re.search(r'A\d+B', self.config.model_id):
			model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
				self.config.model_id, 
				dtype=torch.bfloat16, 
				device_map="auto"
			)
		else:
			model = Qwen3VLForConditionalGeneration.from_pretrained(
				self.config.model_id, 
				dtype=torch.bfloat16, 
				device_map="cuda:0" if torch.cuda.is_available() else "cpu"
			)
		
		self.processor = AutoProcessor.from_pretrained(self.config.model_id)
		return model

	def _build_messages(self, text: str | None, image: str | Path | None) -> list[dict[str, Any]]:
		if text is None and image is None: # big problemo
			raise ValueError("At least one of text or image must be provided.")

		messages: list[dict[str, Any]] = [] # init list for messages
  
		# 1) add system prompt.
		if self.config.system_prompt: # add system prompt if provided (in config)
			messages.append({"role": "system","content": self.config.system_prompt})

		# 2) add user message
		user_content: list[dict[str, Any]] = []
		user_content.append({
			"type": "text",
			"text": self.config.user_prompt.format(text=text or ""),
		})

		user_content.append(build_base64_image_content(image))

		messages.append({"role": "user", "content": user_content})
		return messages


	def inference(self, text: str | None = None, image: str | Path | None = None) -> str:
		messages = self._build_messages(text=text, image=image)

		inputs = self.processor.apply_chat_template(
			messages,
			tokenize=True,
			add_generation_prompt=True,
			return_dict=True,
			return_tensors="pt"
		)
		inputs = inputs.to(self.model.device)

		generated_ids = self.model.generate(
			**inputs, 
			max_new_tokens=self.config.max_new_tokens,
			temperature=self.config.temperature,
		)
		generated_ids_trimmed = [
			out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
		]
		output_text = self.processor.batch_decode(
			generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)[0]
		# NOTE: The current model is instruct-based model, does not include reasoning
		return str(output_text).strip(), None


def load_qwen3vl_generator(
	model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
	max_new_tokens: int = 1024,
	system_prompt: str | None = None,
	temperature: float = 1e-5,
) -> Qwen3VLGeneratorModel:
	"""Convenience loader used by config-driven pipelines."""

	config = Qwen3VLModelConfig(
		model_id=model_id,
		max_new_tokens=max_new_tokens,
		system_prompt=system_prompt,
		temperature=temperature,
	)
	return Qwen3VLGeneratorModel(config)
