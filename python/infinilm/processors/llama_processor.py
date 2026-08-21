from tokenizers import decoders as _dec

from .basic_llm_processor import BasicLLMProcessor
from .processor import register_processor


@register_processor("llama")
class LlamaProcessor(BasicLLMProcessor):
    def __init__(self, model_dir_path: str):
        super().__init__(model_dir_path)
        self._fix_tokenizer_decoder(self.tokenizer)

    @staticmethod
    def _fix_tokenizer_decoder(tokenizer):
        """Fix tokenizer decoder for llama models.

        Fast tokenizers often end with Strip(content=" ", start=1), which drops the
        leading space produced by ▁→" " when decoding a *single* token. Incremental
        generation then concatenates words (Thecorrectansweris...) and thinking
        fills max_tokens=128 so extract_answer returns empty.

        Older trees also had a Prepend("▁") normalizer; require Strip alone so the
        fix still applies when normalizer is None (9g_8b_thinking / FM9G llama).
        """
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        dec = getattr(target, "decoder", None)
        sd = repr(dec)[:800] if dec is not None else ""
        if "Strip" not in sd:
            return
        target.decoder = _dec.Sequence(
            [
                _dec.Replace("▁", " "),
                _dec.ByteFallback(),
                _dec.Fuse(),
            ]
        )
