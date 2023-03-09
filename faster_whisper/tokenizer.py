from functools import cached_property
from typing import List, Optional

import tokenizers


class Tokenizer:
    """Simple wrapper around a tokenizers.Tokenizer."""

    def __init__(
        self,
        tokenizer: tokenizers.Tokenizer,
        multilingual: bool,
        task: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.tokenizer = tokenizer

        if multilingual:
            self.task = self.tokenizer.token_to_id("<|%s|>" % task)
            if self.task is None:
                raise ValueError("%s is not a valid task" % task)

            self.language = self.tokenizer.token_to_id("<|%s|>" % language)
            if self.language is None:
                raise ValueError("%s is not a valid language code" % language)

        else:
            self.task = None
            self.language = None

    @cached_property
    def sot(self) -> int:
        return self.tokenizer.token_to_id("<|startoftranscript|>")

    @cached_property
    def sot_prev(self) -> int:
        return self.tokenizer.token_to_id("<|startofprev|>")

    @cached_property
    def eot(self) -> int:
        return self.tokenizer.token_to_id("<|endoftext|>")

    @cached_property
    def no_timestamps(self) -> int:
        return self.tokenizer.token_to_id("<|notimestamps|>")

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps + 1

    @property
    def sot_sequence(self) -> List[int]:
        sequence = [self.sot]

        if self.language is not None:
            sequence.append(self.language)

        if self.task is not None:
            sequence.append(self.task)

        return sequence

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: List[int]) -> str:
        text_tokens = [token for token in tokens if token < self.eot]
        return self.tokenizer.decode(text_tokens)
