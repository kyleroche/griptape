import sentencepiece as sp
from attr import define, field
from griptape.tokenizers import BaseTokenizer

@define(frozen=True)
class SentencePieceTokenizer(BaseTokenizer):
    DEFAULT_MODEL = "chat-bison@001"
    MAX_TOKENS = 1024

    model: str = field(default=DEFAULT_MODEL, kw_only=True)
    client: cohere.Client = field(kw_only=True)

    @property
    def max_tokens(self) -> int:
        return self.MAX_TOKENS

    def encode(self, text: str) -> list[int]:
        return self.client.tokenize(text=text).tokens

    def decode(self, tokens: list[int]) -> str:
        return self.client.detokenize(tokens=tokens).text
