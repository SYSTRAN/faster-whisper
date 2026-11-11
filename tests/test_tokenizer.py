from types import SimpleNamespace

import pytest

from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_suppressed_tokens


@pytest.fixture
def prompt_model():
    model = WhisperModel.__new__(WhisperModel)
    model.max_length = 448
    return model


@pytest.fixture
def prompt_tokenizer():
    # One token per character, with disjoint text and special-token IDs.
    return SimpleNamespace(
        encode=lambda text: list(map(ord, text)),
        sot_prev=10000,
        sot_sequence=[10001, 10002, 10003],
        no_timestamps=10004,
        timestamp_begin=10005,
    )


@pytest.mark.parametrize("without_timestamps", [False, True])
@pytest.mark.parametrize(
    "hotwords", ["Ada", "h" * 221, "h" * 222, "abcdefghijklmnopqrstuvwxyz" * 24]
)
def test_offline_prompt_hotwords_share_history_budget(
    prompt_model, prompt_tokenizer, without_timestamps, hotwords
):
    previous = list(range(1000, 1500))
    original = previous.copy()
    hotword_tokens = prompt_tokenizer.encode(" " + hotwords)[:223]
    remaining = 223 - len(hotword_tokens)
    history = previous[-remaining:] if remaining else []

    prompt = prompt_model.get_prompt(
        prompt_tokenizer,
        previous,
        hotwords=hotwords,
        without_timestamps=without_timestamps,
    )

    expected = [prompt_tokenizer.sot_prev] + hotword_tokens + history
    expected += prompt_tokenizer.sot_sequence
    if without_timestamps:
        expected.append(prompt_tokenizer.no_timestamps)
    assert prompt == expected
    assert len(prompt) < 448
    assert previous == original


@pytest.mark.parametrize("without_timestamps", [False, True])
@pytest.mark.parametrize("previous", [[], [1000, 1001]])
@pytest.mark.parametrize("hotwords", [None, "  Ada  "])
@pytest.mark.parametrize("prefix", [None, "  hello  "])
def test_offline_prompt_short_inputs_are_compatible(
    prompt_model, prompt_tokenizer, without_timestamps, previous, hotwords, prefix
):
    expected = []
    if previous or (hotwords and not prefix):
        expected.append(prompt_tokenizer.sot_prev)
        if hotwords and not prefix:
            expected += prompt_tokenizer.encode(" " + hotwords.strip())
        expected += previous
    expected += prompt_tokenizer.sot_sequence
    if without_timestamps:
        expected.append(prompt_tokenizer.no_timestamps)
    if prefix:
        if not without_timestamps:
            expected.append(prompt_tokenizer.timestamp_begin)
        expected += prompt_tokenizer.encode(" " + prefix.strip())

    assert (
        prompt_model.get_prompt(
            prompt_tokenizer,
            previous,
            hotwords=hotwords,
            prefix=prefix,
            without_timestamps=without_timestamps,
        )
        == expected
    )


@pytest.mark.parametrize("without_timestamps", [False, True])
@pytest.mark.parametrize("previous", [[], list(range(1000, 1500))])
def test_offline_prompt_long_prefix_reserves_generation_token(
    prompt_model, prompt_tokenizer, without_timestamps, previous
):
    prefix = "abcdefghijklmnopqrstuvwxyz" * 24
    expected = [prompt_tokenizer.sot_prev] + previous[-223:] if previous else []
    expected += prompt_tokenizer.sot_sequence
    expected.append(
        prompt_tokenizer.no_timestamps
        if without_timestamps
        else prompt_tokenizer.timestamp_begin
    )
    prefix_budget = min(223, 448 - len(expected) - 1)
    expected += prompt_tokenizer.encode(" " + prefix)[:prefix_budget]

    prompt = prompt_model.get_prompt(
        prompt_tokenizer,
        previous,
        hotwords="ignored when a prefix is present" * 30,
        prefix=prefix,
        without_timestamps=without_timestamps,
    )

    assert prompt == expected
    assert len(prompt) + 1 <= 448


def test_suppressed_tokens_minus_1():
    model = WhisperModel("tiny.en")

    tokenizer = Tokenizer(model.hf_tokenizer, False)
    tokens = get_suppressed_tokens(tokenizer, [-1])
    assert tokens == (
        1,
        2,
        7,
        8,
        9,
        10,
        14,
        25,
        26,
        27,
        28,
        29,
        31,
        58,
        59,
        60,
        61,
        62,
        63,
        90,
        91,
        92,
        93,
        357,
        366,
        438,
        532,
        685,
        705,
        796,
        930,
        1058,
        1220,
        1267,
        1279,
        1303,
        1343,
        1377,
        1391,
        1635,
        1782,
        1875,
        2162,
        2361,
        2488,
        3467,
        4008,
        4211,
        4600,
        4808,
        5299,
        5855,
        6329,
        7203,
        9609,
        9959,
        10563,
        10786,
        11420,
        11709,
        11907,
        13163,
        13697,
        13700,
        14808,
        15306,
        16410,
        16791,
        17992,
        19203,
        19510,
        20724,
        22305,
        22935,
        27007,
        30109,
        30420,
        33409,
        34949,
        40283,
        40493,
        40549,
        47282,
        49146,
        50257,
        50357,
        50358,
        50359,
        50360,
        50361,
    )


def test_suppressed_tokens_minus_value():
    model = WhisperModel("tiny.en")

    tokenizer = Tokenizer(model.hf_tokenizer, False)
    tokens = get_suppressed_tokens(tokenizer, [13])
    assert tokens == (13, 50257, 50357, 50358, 50359, 50360, 50361)


def test_split_on_unicode():
    model = WhisperModel("tiny")
    tokenizer = Tokenizer(model.hf_tokenizer, False)

    tokens = [8404, 871, 287, 6, 246, 526, 3210, 20378]
    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert words == [" elle", " est", " l", "'", "\ufffd", "é", "rit", "oire"]
    assert word_tokens == [[8404], [871], [287], [6], [246], [526], [3210], [20378]]
