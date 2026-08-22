"""LLM solvers — natural-language policy via local ollama models.

The solver talks to a running ollama server (``http://localhost:11434``)
through stdlib ``urllib`` — there are no third-party deps, so the
package always imports; only request time requires the server.
"""

from .ollama_solver import OllamaConfig, OllamaSolver

__all__ = ["OllamaSolver", "OllamaConfig"]
