"""Portable export tests using explicitly synthetic device fixtures; no hardware."""
import csv
import json
from pathlib import Path
import unittest
import zipfile

import test_audit_sync_session as audit_fixtures
from export_session import export_session
from inventory_device import inventory, sha256
from session_integrity import verify


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = audit_fixtures.AuditTests('test_closed_fake_session_reconstructs')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = self.fixture.root
        self.output = self.fixture.base / 'portable'

    def test_all_rows_invalids_and_reference_bytes_preserved(self):
        before = {p.name: sha256(p) for p in self.source.iterdir() if p.is_file()}
        result = export_session(self.source, self.output, archive=True, origin='synthetic')
        self.assertTrue(result['source_audit_passed'])
        self.assertEqual(result['provenance_kind'], 'synthetic')
        self.assertEqual(result['counts'], {'saved_pairs': 2, 'raw_force_reads': 2})
        for name, value in before.items():
            self.assertEqual(sha256(self.source / name), value)
            self.assertEqual(sha256(self.output / 'raw_source' / name), value)
        with (self.output / 'alignment.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['valid'], 'False')
        self.assertEqual(rows[1]['reason'], 'NO_RIGHT_BRACKET')
        self.assertGreater(verify(self.output), 10)
        with zipfile.ZipFile(self.output.with_suffix('.zip')) as bundle:
            self.assertIsNone(bundle.testzip())
        receipt = json.loads(self.output.with_suffix('.receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['sha256'], sha256(self.output.with_suffix('.zip')))
        self.assertFalse((self.output / 'tensors').exists())
        self.assertFalse((self.output / 'actions.csv').exists())

    def test_failed_audit_requires_explicit_diagnostic_override(self):
        self.fixture.change_csv('capture_sync.csv', lambda rows: rows[0].update(Fz='999'))
        with self.assertRaisesRegex(ValueError, 'audit failed'):
            export_session(self.source, self.output)
        self.assertFalse(self.output.exists())
        result = export_session(self.source, self.output, allow_audit_fail=True, origin='synthetic')
        self.assertFalse(result['source_audit_passed'])
        self.assertEqual(result['qualification_status'], 'NOT_READY')
        self.assertGreater(verify(self.output), 0)

    def test_output_cannot_overlap_source_or_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            export_session(self.source, self.source / 'export')
        export_session(self.source, self.output)
        with self.assertRaises(FileExistsError):
            export_session(self.source, self.output)

    def test_identity_is_explicit_file_inventory_not_hardware_readiness(self):
        report = inventory(self.fixture.base / 'test_profile.txt', self.fixture.base / 'missing.dll')
        self.assertFalse(report['hardware_opened'])
        self.assertEqual(report['status'], 'MISSING_FILES')
        self.assertFalse(report['sdk_abi_verified'])


if __name__ == '__main__':
    unittest.main()
