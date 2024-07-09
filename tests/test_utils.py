import os
import tempfile
import unittest

from faster_whisper import available_models, download_model


class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for each test
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Cleanup the temporary directory after each test
        self.tmpdir.cleanup()

    def test_available_models(self):
        models = available_models()
        self.assertIsInstance(models, list)
        self.assertIn("tiny", models)

    def test_download_model(self):
        output_dir = os.path.join(self.tmpdir.name, "model")
        model_dir = download_model("tiny", output_dir=output_dir)

        self.assertEqual(model_dir, output_dir)
        self.assertTrue(os.path.isdir(model_dir))
        self.assertFalse(os.path.islink(model_dir))

        for filename in os.listdir(model_dir):
            path = os.path.join(model_dir, filename)
            self.assertFalse(os.path.islink(path))

    def test_download_model_in_cache(self):
        cache_dir = os.path.join(self.tmpdir.name, "model")
        download_model("tiny", cache_dir=cache_dir)
        self.assertTrue(os.path.isdir(cache_dir))
