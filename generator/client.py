"""Unified client for generating text via a vLLM-served model using the
OpenAI-compatible API (``/v1/completions`` and ``/v1/chat/completions``)."""

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


class VLLMGenerator:
    """Thin wrapper around the OpenAI client that targets a vLLM server.

    Parameters
    ----------
    model_name : str
        The model identifier as registered on the vLLM server
        (e.g. ``"Qwen/Qwen3-32B"``).
    base_url : str
        Root URL of the OpenAI-compatible API exposed by vLLM.
    api_key : str
        API key (vLLM does not require one by default; ``"EMPTY"`` is the
        conventional placeholder).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # Text (prompt) completion  –  /v1/completions
    # ------------------------------------------------------------------

    def complete(self, prompt, max_tokens=8, temperature=0.0, **kwargs):
        """Single text completion for one prompt string."""
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].text

    def complete_batch(self, prompts, max_tokens=8, temperature=0.0, **kwargs):
        """Text completion for a list of prompt strings (single API call).

        vLLM accepts a list of prompts in one request and schedules them
        internally, which is more efficient than issuing individual calls.
        """
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        sorted_choices = sorted(response.choices, key=lambda c: c.index)
        return [c.text for c in sorted_choices]

    # ------------------------------------------------------------------
    # Chat completion  –  /v1/chat/completions
    # ------------------------------------------------------------------

    def chat(self, messages, max_tokens=8, temperature=0.7, **kwargs):
        """Chat completion for a single conversation (list of message dicts)."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content

    def chat_batch(
        self,
        messages_list,
        max_tokens=8,
        temperature=0.7,
        max_workers=8,
        **kwargs,
    ):
        """Chat completion for multiple conversations via concurrent requests.

        Each element of *messages_list* is a full conversation (list of message
        dicts).  Results are returned in the same order as the input.
        """
        results = [None] * len(messages_list)

        def _call(idx, msgs):
            return idx, self.chat(
                msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        n_workers = min(max_workers, len(messages_list))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_call, i, m)
                for i, m in enumerate(messages_list)
            ]
            for fut in as_completed(futures):
                idx, text = fut.result()
                results[idx] = text

        return results
