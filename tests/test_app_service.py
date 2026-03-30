import os
import shutil
import unittest
import uuid
from unittest.mock import patch

from src import app_service


class UploadStub:
    def __init__(self, name):
        self.name = name


class AppServiceTests(unittest.TestCase):
    def test_normalize_session_id_removes_unsafe_chars(self):
        normalized = app_service.normalize_session_id("test:/bad?id")
        self.assertEqual(normalized, "testbadid")

    def test_build_user_message_includes_files(self):
        user_text = "\u0445\u043e\u0447\u0443 \u043f\u043e\u0445\u043e\u0436\u0438\u0435 \u0444\u0438\u043b\u044c\u043c\u044b"
        message = app_service.build_user_message(user_text, [UploadStub("poster.jpg")])
        self.assertIn(user_text, message)
        self.assertIn("[\u0424\u0430\u0439\u043b] poster.jpg", message)

    def test_process_chat_turn_persists_messages_and_state(self):
        fake_pipeline_result = {
            "response": "\u0413\u043e\u0442\u043e\u0432\u043e",
            "conversation_state": {
                "anchor_movie": {
                    "imdb_id": "1375666",
                    "display_full_title": "\u041d\u0430\u0447\u0430\u043b\u043e (2010)",
                },
                "filters": {"year_filter": {}, "include_genres": [], "exclude_genres": []},
                "result_limit": 5,
                "last_query": "\u041d\u0430\u0447\u0430\u043b\u043e",
                "turn_count": 1,
            },
            "ui_payload": {"metrics": {"recommendation_count": 3}},
        }

        temp_root = os.path.join(os.getcwd(), "build-check")
        temp_dir = os.path.join(temp_root, f"app-service-test-{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            with patch("src.app_service.run_integrated_pipeline", return_value=fake_pipeline_result):
                result = app_service.process_chat_turn(
                    temp_dir,
                    "session-1",
                    user_text="\u041c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f \u041d\u0430\u0447\u0430\u043b\u043e",
                    uploaded_files=[],
                )

                self.assertEqual(result["response"], "\u0413\u043e\u0442\u043e\u0432\u043e")
                self.assertEqual(len(result["messages"]), 2)
                self.assertEqual(result["messages"][0]["role"], "user")
                self.assertEqual(result["messages"][1]["role"], "assistant")

                reloaded = app_service.load_chat_session(temp_dir, "session-1")
                self.assertEqual(len(reloaded["messages"]), 2)
                self.assertEqual(
                    reloaded["conversation_state"]["anchor_movie"]["display_full_title"],
                    "\u041d\u0430\u0447\u0430\u043b\u043e (2010)",
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.isdir(temp_root) and not os.listdir(temp_root):
                os.rmdir(temp_root)


if __name__ == "__main__":
    unittest.main()
