from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_suppressed_tokens


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
    )


def test_suppressed_tokens_minus_value():
    model = WhisperModel("tiny.en")

    tokenizer = Tokenizer(model.hf_tokenizer, False)
    tokens = get_suppressed_tokens(tokenizer, [13])
    assert tokens == (13, 50257, 50357, 50358, 50359, 50360)


def test_split_on_unicode():
    model = WhisperModel("tiny")
    tokenizer = Tokenizer(model.hf_tokenizer, False)

    tokens = [8404, 871, 287, 6, 246, 526, 3210, 20378]
    words, word_tokens = tokenizer.split_tokens_on_unicode(tokens)

    assert words == [" elle", " est", " l", "'", "\ufffd", "é", "rit", "oire"]
    assert word_tokens == [[8404], [871], [287], [6], [246], [526], [3210], [20378]]
