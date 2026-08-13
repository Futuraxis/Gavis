"""LLM solvers — natural-language policy via local ollama models.

Requires a running ollama server (``http://localhost:11434``).  Imports
fail gracefully when ``requests``-free stdlib path is used (no deps), so
the package always imports; only instantiation requires the server.
"""

from .ollama_solver import OllamaConfig, OllamaSolver

__all__ = ["OllamaSolver", "OllamaConfig"]
