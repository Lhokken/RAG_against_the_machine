#!/usr/bin/env python3

import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Utility class wrapping a lightweight Hugging Face causal-LM for fast,
    low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* we automatically
        select ``mps`` when available on macOS,
        ``cuda`` when available, otherwise we fall back to ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. When using a GPU or MPS
        we default to ``float16`` to keep memory
        usage reasonable; on CPU we keep ``float32`` for maximum compatibility.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in \
                ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )

        self._model.to(self._device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> list[int]: #torch.Tensor
        """
        Tokenise *text* and return a 2-D
        ``input_ids`` tensor on the target device.
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        # return torch.tensor([ids], device=self._device, dtype=torch.long)
        return ids

    def decode(self, ids: torch.Tensor | list[int]) -> str | list[str]:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        result: str | list[str]
        result = self._tokenizer.decode(ids, skip_special_tokens=True)

        return result

    def get_logits_from_input_ids(
            self,
            input_ids: list[int] | int,
            past_key_values: Any | None=None
            ) -> tuple[list[float], Any]:
        """
        Given a list of input token ids, return the raw logits
        (no softmax) for the next token.
        """
        if isinstance(input_ids, int):
            tensor_data = [[input_ids]]
        else:
            tensor_data = [input_ids]
        input_tensor = torch.tensor(
            tensor_data, device=self._device, dtype=torch.long
            )
        with torch.no_grad():
            out = self._model(
                input_ids=input_tensor,
                past_key_values=past_key_values,
                use_cache=True
                )
        logits = out.logits[0, -1].tolist()
        print(logits)
        return ([float(x) for x in logits], out.past_key_valus)
