import os
import tempfile
import unittest

from fixer.patcher import replace_function_in_file
from fixer.file_ops import read_text, write_text


SAMPLE = """
class Bot:
    async def aitrader_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        pass
"""


class TestPatcher(unittest.TestCase):
    def test_replace(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bot.py")
            write_text(path, SAMPLE)
            replace_function_in_file(path, "aitrader_command")
            out = read_text(path)
            self.assertIn("async def aitrader_command", out)
            self.assertIn("try:", out)
            self.assertIn("except Exception as e:", out)


if __name__ == "__main__":
    unittest.main()