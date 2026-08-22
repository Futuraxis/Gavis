#!/usr/bin/env python3
"""Compatibility entrypoint for Layer 1 LLM training.

Use ``demos.train_layer1_lora`` for the current QLoRA SFT workflow.
"""

from __future__ import annotations

from .train_layer1_lora import parse_args, train

if __name__ == "__main__":
    train(parse_args())
