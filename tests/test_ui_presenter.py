import unittest

from src.ui_presenter import (
    build_genre_distribution,
    build_rating_chart_frame,
    build_recommendation_dataframe,
    build_supported_types_caption,
    build_year_distribution,
)


class UIPresenterTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "rank": 1,
                "title": "Inception (2010)",
                "year": 2010,
                "rating": 8.1,
                "genres": "фантастика, триллер",
                "score": 0.91,
            },
            {
                "rank": 2,
                "title": "Interstellar (2014)",
                "year": 2014,
                "rating": 8.3,
                "genres": "фантастика, драма",
                "score": 0.88,
            },
        ]

    def test_build_recommendation_dataframe_renames_columns(self):
        dataframe = build_recommendation_dataframe(self.rows)

        self.assertEqual(list(dataframe.columns), ["№", "Фильм", "Год", "Рейтинг", "Жанры", "Score"])
        self.assertEqual(len(dataframe), 2)

    def test_build_genre_distribution_counts_genres(self):
        genre_frame = build_genre_distribution(self.rows)
        counts = dict(zip(genre_frame["Жанр"], genre_frame["Количество"]))

        self.assertEqual(counts["фантастика"], 2)
        self.assertEqual(counts["триллер"], 1)
        self.assertEqual(counts["драма"], 1)

    def test_build_year_distribution_counts_years(self):
        year_frame = build_year_distribution(self.rows)
        counts = dict(zip(year_frame["Год"], year_frame["Количество"]))

        self.assertEqual(counts[2010], 1)
        self.assertEqual(counts[2014], 1)

    def test_build_rating_chart_frame_uses_movie_titles_as_index(self):
        rating_frame = build_rating_chart_frame(self.rows)

        self.assertIn("Inception (2010)", rating_frame.index)
        self.assertIn("Рейтинг", rating_frame.columns)

    def test_build_supported_types_caption_formats_extensions(self):
        caption = build_supported_types_caption(["jpg", "png", "txt"])

        self.assertEqual(caption, ".jpg, .png, .txt")


if __name__ == "__main__":
    unittest.main()
