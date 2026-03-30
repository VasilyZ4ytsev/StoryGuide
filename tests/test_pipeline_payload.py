import unittest

from src.integration_pipeline import run_integrated_pipeline


class PipelinePayloadTests(unittest.TestCase):
    def test_ui_payload_contains_rows_and_metrics(self):
        result = run_integrated_pipeline("Мне нравится Матрица", [])
        payload = result.get("ui_payload", {})

        self.assertEqual(payload.get("search_mode"), "title_match")
        self.assertTrue(payload.get("source_movie"))
        self.assertGreater(len(payload.get("recommendation_rows", [])), 0)
        self.assertEqual(
            payload.get("metrics", {}).get("recommendation_count"),
            len(payload.get("recommendation_rows", [])),
        )

    def test_ui_payload_keeps_requested_limit(self):
        result = run_integrated_pipeline("выдай мне 7 фильмов как терминатор", [])
        payload = result.get("ui_payload", {})

        self.assertEqual(payload.get("requested_limit"), 7)
        self.assertLessEqual(len(payload.get("recommendation_rows", [])), 7)


if __name__ == "__main__":
    unittest.main()
