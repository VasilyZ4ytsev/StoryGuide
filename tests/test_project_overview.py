import unittest
from unittest.mock import patch

from src.project_overview import get_project_overview


class ProjectOverviewTests(unittest.TestCase):
    def setUp(self):
        get_project_overview.cache_clear()

    def tearDown(self):
        get_project_overview.cache_clear()

    def test_project_overview_contains_summary_and_startup_commands(self):
        overview = get_project_overview()

        self.assertGreaterEqual(len(overview.get("summary_metrics", [])), 4)
        self.assertGreaterEqual(overview.get("catalog_metrics", {}).get("movie_count", 0), 1)
        self.assertIn("supported_types", overview)
        self.assertGreaterEqual(len(overview.get("startup_commands", [])), 3)
        self.assertGreaterEqual(len(overview.get("architecture_layers", [])), 3)
        self.assertIn(
            "streamlit run app.py",
            [item.get("command") for item in overview.get("startup_commands", [])],
        )

    def test_project_overview_fails_fast_when_catalog_is_unavailable(self):
        with patch("src.project_overview.load_movie_metadata", side_effect=FileNotFoundError("missing dataset")):
            with self.assertRaises(FileNotFoundError):
                get_project_overview()


if __name__ == "__main__":
    unittest.main()
