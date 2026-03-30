import unittest
from unittest.mock import patch

from src import main


class MainEntrypointTests(unittest.TestCase):
    def test_main_prints_hint_in_bare_mode(self):
        with patch("src.main.get_script_run_ctx", return_value=None):
            with patch("builtins.print") as mock_print:
                with patch("src.main.st.set_page_config") as mock_set_page_config:
                    main.main()

        mock_print.assert_called_once_with(main.get_bare_mode_message())
        mock_set_page_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
