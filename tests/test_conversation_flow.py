import unittest
from unittest.mock import patch

from src.integration_pipeline import run_integrated_pipeline


class UploadStub:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


class ConversationFlowTests(unittest.TestCase):
    def test_anchor_movie_is_stored_after_first_turn(self):
        result = run_integrated_pipeline("Мне нравится Матрица", [])
        state = result["conversation_state"]

        self.assertIsNotNone(state["anchor_movie"])
        self.assertIn("Матрица", state["anchor_movie"]["display_full_title"])

    def test_follow_up_year_filter_is_accumulated(self):
        first_turn = run_integrated_pipeline("Мне нравится Матрица", [])
        second_turn = run_integrated_pipeline(
            "после 2010",
            [],
            conversation_state=first_turn["conversation_state"],
        )

        state = second_turn["conversation_state"]
        self.assertEqual(state["filters"]["year_filter"].get("min_year"), 2010)
        self.assertIn("Фильтр по годам: после 2010.", second_turn["response"])

    def test_follow_up_excluded_genre_is_accumulated(self):
        first_turn = run_integrated_pipeline("Мне нравится Матрица", [])
        second_turn = run_integrated_pipeline(
            "без комедии",
            [],
            conversation_state=first_turn["conversation_state"],
        )

        state = second_turn["conversation_state"]
        self.assertIn("комедия", state["filters"]["exclude_genres"])
        self.assertIn("Исключены жанры: комедия.", second_turn["response"])

    def test_relative_follow_up_newer_infers_year_from_anchor(self):
        first_turn = run_integrated_pipeline("Мне нравится Матрица", [])
        second_turn = run_integrated_pipeline(
            "а что поновее?",
            [],
            conversation_state=first_turn["conversation_state"],
        )

        state = second_turn["conversation_state"]
        self.assertEqual(state["filters"]["year_filter"].get("min_year"), 2000)
        self.assertIn("Фильтр по годам: после 2000.", second_turn["response"])

    def test_watched_phrase_can_set_anchor_movie(self):
        result = run_integrated_pipeline("Я вчера посмотрел Интерстеллар, выдай похожие фильмы", [])

        state = result["conversation_state"]
        self.assertIsNotNone(state["anchor_movie"])
        self.assertIn("Интерстеллар", state["anchor_movie"]["display_full_title"])

    def test_anchor_reference_phrase_uses_current_context(self):
        first_turn = run_integrated_pipeline("Мне нравится Матрица", [])
        second_turn = run_integrated_pipeline(
            "выдай мне список фильмов похожий на этот",
            [],
            conversation_state=first_turn["conversation_state"],
        )

        self.assertIn("Матрица", second_turn["response"])
        self.assertEqual(
            second_turn["conversation_state"]["anchor_movie"]["display_full_title"],
            first_turn["conversation_state"]["anchor_movie"]["display_full_title"],
        )

    def test_greeting_and_free_order_request_are_understood(self):
        result = run_integrated_pipeline(
            "эй привет а выдайка мне фильмы похожие на крестный отец 10 штук",
            [],
        )

        state = result["conversation_state"]
        self.assertIsNotNone(state["anchor_movie"])
        self.assertIn("Крестный отец", state["anchor_movie"]["display_full_title"])
        self.assertEqual(state["result_limit"], 10)
        recommendation_lines = [
            line for line in result["response"].splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        self.assertEqual(len(recommendation_lines), 10)

    def test_explicit_new_movie_replaces_previous_anchor(self):
        first_turn = run_integrated_pipeline("выдай мне 10 фильмов как крестный отец", [])
        second_turn = run_integrated_pipeline(
            "выдай мне 10 фильмов как терминатор",
            [],
            conversation_state=first_turn["conversation_state"],
        )

        state = second_turn["conversation_state"]
        self.assertIsNotNone(state["anchor_movie"])
        self.assertIn("Терминатор", state["anchor_movie"]["display_full_title"])
        self.assertNotIn("Крестный отец", second_turn["response"])

    def test_inflected_russian_title_is_understood(self):
        result = run_integrated_pipeline("выдай мне 5 фильмов как крестного отца", [])

        state = result["conversation_state"]
        self.assertIsNotNone(state["anchor_movie"])
        self.assertIn("Крестный отец", state["anchor_movie"]["display_full_title"])

    def test_numeric_request_extracts_titanic_title(self):
        result = run_integrated_pipeline("выдай мне 10 фильмов как титаник", [])

        state = result["conversation_state"]
        self.assertIsNotNone(state["anchor_movie"])
        self.assertIn("Титаник", state["anchor_movie"]["display_full_title"])

    def test_short_new_title_replaces_previous_anchor_even_with_filters(self):
        first_turn = run_integrated_pipeline("Терминатор", [])
        second_turn = run_integrated_pipeline(
            "Порекомендуй какой-нибудь триллер",
            [],
            conversation_state=first_turn["conversation_state"],
        )
        third_turn = run_integrated_pipeline(
            "Аватар",
            [],
            conversation_state=second_turn["conversation_state"],
        )

        self.assertIn("Аватар", third_turn["conversation_state"]["anchor_movie"]["display_full_title"])
        self.assertNotIn("Жанровый фокус: триллер.", third_turn["response"])

    def test_napodobie_phrase_sets_new_anchor(self):
        first_turn = run_integrated_pipeline("Терминатор", [])
        second_turn = run_integrated_pipeline(
            "Наподобие Однажды в Голливуде",
            [],
            conversation_state=first_turn["conversation_state"],
        )

        anchor = second_turn["conversation_state"]["anchor_movie"]
        if anchor is not None:
            self.assertNotIn("Терминатор", anchor["display_full_title"])
        self.assertNotIn("Похоже, вам нравится Терминатор", second_turn["response"])

    @patch("src.integration_pipeline.analyze_uploaded_image")
    def test_file_turn_can_replace_anchor_movie(self, mock_analyze_uploaded_image):
        mock_analyze_uploaded_image.return_value = {
            "summary": "Постер фильма",
            "storyguide_query": "Мне нравится Начало",
            "extracted_data": {
                "recognition_label": "Найден по постеру: Inception (2010)",
                "detected_genres": ["фантастика"],
                "matched_imdb_id": "1375666",
            },
        }

        first_turn = run_integrated_pipeline("Мне нравится Матрица", [])
        upload = UploadStub("poster.jpg", b"fake-image")
        second_turn = run_integrated_pipeline(
            "",
            [upload],
            conversation_state=first_turn["conversation_state"],
        )

        state = second_turn["conversation_state"]
        self.assertIsNotNone(state["anchor_movie"])
        self.assertTrue(
            "Inception" in state["anchor_movie"]["display_full_title"]
            or "Начало" in state["anchor_movie"]["display_full_title"]
        )


if __name__ == "__main__":
    unittest.main()
