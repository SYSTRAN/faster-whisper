import os
import unittest

from faster_whisper import WhisperModel, decode_audio
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import get_suppressed_tokens


class TestTranscribe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        cls.jfk_path = os.path.join(cls.data_dir, "jfk.flac")
        cls.model_tiny = WhisperModel("tiny")
        cls.model_tiny_en = WhisperModel("tiny.en")

    def test_supported_languages(self):
        self.assertEqual(self.model_tiny_en.supported_languages, ["en"])

    def test_transcribe(self):
        segments, info = self.model_tiny.transcribe(self.jfk_path, word_timestamps=True)
        self.assertIsNotNone(info.all_language_probs)

        self.assertEqual(info.language, "en")
        self.assertGreater(info.language_probability, 0.9)
        self.assertEqual(info.duration, 11)

        # Get top language info from all results, which should match the
        # already existing metadata
        top_lang, top_lang_score = info.all_language_probs[0]
        self.assertEqual(info.language, top_lang)
        self.assertAlmostEqual(info.language_probability, top_lang_score, places=16)

        segments = list(segments)

        self.assertEqual(len(segments), 1)

        segment = segments[0]

        self.assertEqual(
            segment.text,
            (
                " And so my fellow Americans ask not what your country can do for you, "
                "ask what you can do for your country."
            ),
        )

        self.assertEqual(segment.text, "".join(word.word for word in segment.words))
        self.assertEqual(segment.start, segment.words[0].start)
        self.assertEqual(segment.end, segment.words[-1].end)

    def test_prefix_with_timestamps(self):
        segments, _ = self.model_tiny.transcribe(
            self.jfk_path, prefix="And so my fellow Americans"
        )
        segments = list(segments)

        self.assertEqual(len(segments), 1)

        segment = segments[0]

        self.assertEqual(
            segment.text,
            (
                " And so my fellow Americans ask not what your country can do for you, "
                "ask what you can do for your country."
            ),
        )

        self.assertEqual(segment.start, 0)
        self.assertGreater(segment.end, 10)
        self.assertLess(segment.end, 11)

    def test_vad(self):
        segments, info = self.model_tiny.transcribe(
            self.jfk_path,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        )
        segments = list(segments)

        self.assertEqual(len(segments), 1)
        segment = segments[0]

        self.assertEqual(
            segment.text,
            (
                " And so my fellow Americans ask not what your country can do for you, "
                "ask what you can do for your country."
            ),
        )

        self.assertGreater(segment.start, 0)
        self.assertLess(segment.start, 1)
        self.assertGreater(segment.end, 10)
        self.assertLess(segment.end, 11)

        self.assertEqual(info.vad_options.min_silence_duration_ms, 500)
        self.assertEqual(info.vad_options.speech_pad_ms, 200)

    def test_stereo_diarization(self):
        audio_path = os.path.join(self.data_dir, "stereo_diarization.wav")
        left, right = decode_audio(audio_path, split_stereo=True)

        segments, _ = self.model_tiny.transcribe(left)
        transcription = "".join(segment.text for segment in segments).strip()
        self.assertEqual(
            transcription,
            (
                "He began a confused complaint against the wizard, "
                "who had vanished behind the curtain on the left."
            ),
        )

        segments, _ = self.model_tiny.transcribe(right)
        transcription = "".join(segment.text for segment in segments).strip()
        self.assertEqual(transcription, "The horizon seems extremely distant.")

    def test_suppressed_tokens_minus_1(self):
        tokenizer = Tokenizer(self.model_tiny_en.hf_tokenizer, False)
        tokens = get_suppressed_tokens(tokenizer, [-1])

        gt_tokens = [
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
        ]
        self.assertListEqual(tokens, gt_tokens)

    def test_suppressed_tokens_minus_value(self):
        tokenizer = Tokenizer(self.model_tiny_en.hf_tokenizer, False)
        tokens = get_suppressed_tokens(tokenizer, [13])
        self.assertListEqual(tokens, [13, 50257, 50357, 50358, 50359, 50360])

    def test_language_detection(self):
        lang_code, lang_prob, lang_probs, _ = self.model_tiny.detect_language(
            self.jfk_path
        )
        self.assertEqual(lang_code, "en")
        self.assertIsNotNone(lang_prob)
        self.assertIsNotNone(lang_probs)

        lang_code, lang_prob, lang_probs, _ = self.model_tiny_en.detect_language(
            self.jfk_path
        )
        self.assertEqual(lang_code, "en")
        self.assertEqual(lang_prob, 1.0)
        self.assertIsNone(lang_probs)
