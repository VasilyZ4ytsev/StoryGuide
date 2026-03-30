import unittest
from pathlib import Path
from unittest.mock import patch

from src import dataset_loader, logic, poster_matcher


class FailFastCleanupTests(unittest.TestCase):
    def setUp(self):
        self.missing_root = Path.cwd() / "build-check" / "fail-fast-missing"

    def tearDown(self):
        dataset_loader.load_movie_metadata.cache_clear()
        dataset_loader.match_movie_title.cache_clear()
        dataset_loader._load_movie_localizations.cache_clear()
        poster_matcher.load_poster_index.cache_clear()

    def test_missing_localizations_file_raises_instead_of_falling_back(self):
        missing_path = str(self.missing_root / "missing_localizations.json")
        required_paths = [
            missing_path if path == dataset_loader.MOVIE_LOCALIZATIONS_PATH else path
            for path in dataset_loader.MOVIE_METADATA_REQUIRED_PATHS
        ]
        with patch("src.dataset_loader.MOVIE_LOCALIZATIONS_PATH", missing_path):
            with patch("src.dataset_loader.MOVIE_METADATA_REQUIRED_PATHS", required_paths):
                with self.assertRaises(FileNotFoundError):
                    dataset_loader.load_movie_metadata()

    def test_poster_index_requires_full_poster_directory(self):
        missing_dir = self.missing_root / "missing_posters"
        with patch("src.poster_matcher.FULL_POSTER_DIR", missing_dir):
            with self.assertRaises(FileNotFoundError):
                poster_matcher.load_poster_index()

    def test_analyze_query_propagates_nlp_failures(self):
        with patch("src.logic.analyze_text", side_effect=RuntimeError("natasha unavailable")):
            with self.assertRaises(RuntimeError):
                logic.analyze_query("Мне нравится Матрица")

    def test_inflected_russian_title_match_still_works(self):
        match = dataset_loader.match_movie_title("крестного отца")

        self.assertIsNotNone(match)
        self.assertIn("Крестный отец", match["record"]["display_full_title"])


if __name__ == "__main__":
    unittest.main()
