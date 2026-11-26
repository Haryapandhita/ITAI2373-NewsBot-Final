"""
Minimal wrapper around Hugging Face encoder-decoder models (BART, MarianMT).

``transformers`` 5.x removed the ``summarization`` and ``translation``
pipelines, so NewsBot calls ``AutoModelForSeq2SeqLM.generate`` directly.
This works with transformers 4.x and 5.x, on GPU when available.
"""

from __future__ import annotations

from typing import List


class Seq2SeqModel:
    """Load a seq2seq checkpoint once and generate text for batches of inputs."""

    def __init__(self, model_name: str, max_input_tokens: int = 1024):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    def generate(self, texts: List[str], max_new_tokens: int = 128, min_new_tokens: int = 0,
                 num_beams: int = 4, batch_size: int = 8) -> List[str]:
        """Generate one output string per input string."""
        outputs: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.max_input_tokens).to(self.device)
            with self.torch.no_grad():
                ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                          num_beams=num_beams, no_repeat_ngram_size=3, early_stopping=True)
            outputs.extend(t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return outputs
