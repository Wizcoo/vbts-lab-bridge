"""Synthetic filesystem tests only; no collected measurements."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('session_integrity', Path(__file__).resolve().parents[1] / 'tools/session_integrity.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / '嵌套 空间').mkdir()
        self.sample = self.root / '嵌套 空间' / 'sample.txt'
        self.sample.write_text('SYNTHETIC TEST FILE ONLY', encoding='utf-8')

    def test_unicode_roundtrip(self):
        self.assertEqual(tool.seal(self.root), 1)
        self.assertEqual(tool.verify(self.root), 1)

    def test_mutation_rejected(self):
        tool.seal(self.root)
        self.sample.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            tool.verify(self.root)

    def test_addition_rejected(self):
        tool.seal(self.root)
        (self.root / 'extra.txt').write_text('extra', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            tool.verify(self.root)

    def test_missing_rejected(self):
        tool.seal(self.root)
        self.sample.unlink()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            tool.verify(self.root)

    def test_traversal_rejected(self):
        (self.root / tool.MANIFEST).write_text('0' * 64 + '  ../outside.txt\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            tool.verify(self.root)

    def test_duplicate_rejected(self):
        tool.seal(self.root)
        p = self.root / tool.MANIFEST
        text = p.read_text(encoding='utf-8')
        p.write_text(text + text, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            tool.verify(self.root)

    def test_reseal_preserves_existing(self):
        tool.seal(self.root)
        p = self.root / tool.MANIFEST
        original = p.read_bytes()
        with self.assertRaises(FileExistsError):
            tool.seal(self.root)
        self.assertEqual(p.read_bytes(), original)

    def test_empty_rejected(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(ValueError, 'empty'):
                tool.seal(empty)


if __name__ == '__main__':
    unittest.main()
