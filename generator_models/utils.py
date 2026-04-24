"""Shared helper utilities for evaluator model wrappers."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

# see docs here: https://openrouter.ai/docs/guides/overview/multimodal/images
# NOTE: in this code base, we are mostly going to use locally download images. Thus, most of the wrappers will encode images to base 64 for the payload.

def encode_image_to_base64(image_path: str | Path) -> str:
	"""Read an image file and return raw base64 content."""
	path = Path(image_path)
	return base64.b64encode(path.read_bytes()).decode("ascii")


def encode_image_to_data_url(image_path: str | Path) -> str:
	"""Return a data URL suitable for image_url payloads in chat completions."""
	path = Path(image_path)
	mime_type, _ = mimetypes.guess_type(str(path))
	if mime_type is None:
		mime_type = "image/jpeg"
	base64_image = encode_image_to_base64(path)
	return f"data:{mime_type};base64,{base64_image}"


def build_base64_image_content(image_path: str | Path) -> dict[str, Any]:
	"""Build an OpenAI/OpenRouter image payload using base64-encoded data URL."""
	# return {
	# 	"type": "image_url",
	# 	"image_url": {
	# 		"url": encode_image_to_data_url(image_path),
	# 	},
	# }

	return {
		"type": "input_image",
		"image_url": encode_image_to_data_url(image_path),
	}
