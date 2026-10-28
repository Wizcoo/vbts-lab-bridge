"""Offline collector state tests. No cameras/DLLs are created or opened."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from collect_sync_dataset import CollectorController, parse_args as _parse_args, run_collector as _run_collector

FIXTURE = Path(__file__).with_name("test_force_profile.txt")
def parse_args(argv=None):
    return _parse_args(["--profile", str(FIXTURE)] + list(argv or []))
def run_collector(argv=None):
    return _run_collector(["--profile", str(FIXTURE)] + list(argv or []))


class FakeClock:
    def __init__(self):
        self.now = 1_000
    def __call__(self):
        return self.now


class FakeSensor:
    def __init__(self, clock):
        self.clock = clock
        self.tare_epoch = 0
        self.last_tare_end_ns = 0
        self.zero_ok = True
        self.match_valid = True
        self.calls = []
    def zero(self, reason):
        if self.zero_ok:
            self.tare_epoch += 1
            self.last_tare_end_ns = self.clock()
        return self.zero_ok
    def match(self, target_ns, *, max_gap_ns, expected_tare_epoch):
        self.calls.append((target_ns, max_gap_ns, expected_tare_epoch))
        valid = self.match_valid and expected_tare_epoch == self.tare_epoch
        return SimpleNamespace(valid=valid, status="valid" if valid else "missing_right",
                               label=(1,2,3,4,5,6) if valid else None,
                               tare_epoch=expected_tare_epoch)


class FakeRecorder:
    def __init__(self):
        self.events, self.references, self.pairs = [], [], []
        self.fail_reference = self.fail_pair = False
    def event(self, name, **fields):
        self.events.append((name, fields))
    def save_reference(self, packet, match, reason):
        if self.fail_reference:
            raise OSError("mock reference PNG failure")
        self.references.append((packet, match))
        return f"reference_{len(self.references):06d}"
    def record_pair(self, packet, match, **fields):
        self.pairs.append((packet, match, fields))
        if self.fail_pair:
            raise OSError("mock PNG failure")
        return {"image_written": 1}


def packet(seq, start=1100, end=1110):
    return SimpleNamespace(seq=seq, grab_start_ns=start, grab_end_ns=end)


class CollectorStateTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.sensor = FakeSensor(self.clock)
        self.recorder = FakeRecorder()
        self.c = CollectorController(self.sensor, self.recorder, clock_ns=self.clock,
                                     pair_wait_ns=100, reference_wait_ns=1000)

    def establish_reference(self):
        self.assertTrue(self.c.request_reference())
        self.c.step(packet(1))
        self.assertIsNotNone(self.c.reference_id)

    def test_R_without_reference_is_rejected(self):
        self.c.step(packet(1))
        self.assertFalse(self.c.toggle_recording())
        self.assertEqual(self.recorder.references, [])
        self.assertEqual(self.recorder.pairs, [])
        self.assertIn("press T", self.c.status_text)

    def test_tare_rejects_frame_acquired_across_tare(self):
        self.c.step(packet(7, 980, 990))
        self.c.request_reference()
        self.c.step(packet(7, 980, 990))
        self.c.step(packet(8, 995, 1010))
        self.assertEqual(len(self.recorder.references), 0)
        self.c.step(packet(9, 1001, 1020))
        self.assertEqual(self.recorder.references[0][0].seq, 9)
        self.assertFalse(self.c.recording)
        self.assertEqual(self.sensor.calls[-1][2], 1)

    def test_reference_candidate_waits_for_right_side_without_chasing_latest(self):
        self.sensor.match_valid = False
        self.c.request_reference()
        self.c.step(packet(1))
        self.c.step(packet(2, 1120, 1130))
        self.assertEqual(len(self.recorder.references), 0)
        self.sensor.match_valid = True
        self.c.step(packet(2, 1120, 1130))
        self.assertEqual(self.recorder.references[0][0].seq, 1)
        self.assertFalse(self.c.recording)

    def test_unique_sequence_is_written_only_once(self):
        self.establish_reference()
        self.c.toggle_recording()
        self.c.step(packet(1))  # reference already consumed, no duplicate
        self.c.step(packet(2))
        self.c.step(packet(2))
        self.c.step(packet(2))
        self.assertEqual([p[0].seq for p in self.recorder.pairs], [2])
        self.assertEqual(self.c.saved_count, 1)

    def test_invalid_bracket_is_retained_with_no_label_after_bounded_wait(self):
        self.establish_reference()
        self.c.toggle_recording()
        self.sensor.match_valid = False
        self.c.step(packet(2))
        self.assertEqual(self.recorder.pairs, [])
        self.clock.now += 101
        self.c.step(packet(2))
        self.assertEqual(len(self.recorder.pairs), 1)
        self.assertIsNone(self.recorder.pairs[0][1].label)
        self.assertEqual(self.recorder.pairs[0][1].status, "missing_right")

    def test_pair_write_error_does_not_increment_or_retry_same_frame_on_exit(self):
        self.establish_reference()
        self.c.toggle_recording()
        self.recorder.fail_pair = True
        with self.assertRaises(OSError):
            self.c.step(packet(2))
        self.assertEqual(self.c.saved_count, 0)
        self.c.finish("error")
        self.assertEqual(len(self.recorder.pairs), 1)

    def test_tare_failure_invalidates_previous_reference_and_keeps_paused(self):
        self.establish_reference()
        self.c.toggle_recording()
        self.sensor.zero_ok = False
        self.assertFalse(self.c.request_reference())
        self.assertIsNone(self.c.reference_id)
        self.assertFalse(self.c.recording)
        self.assertFalse(self.c.toggle_recording())
        self.assertEqual(len(self.recorder.references), 1)

    def test_reference_write_failure_is_not_applied(self):
        self.c.request_reference()
        self.recorder.fail_reference = True
        with self.assertRaises(OSError):
            self.c.step(packet(1))
        self.assertIsNone(self.c.reference_id)
        self.assertIsNone(self.c.pending_reference)
        self.assertFalse(self.c.recording)
        self.assertEqual(len(self.recorder.references), 0)

    def test_reference_timeout_keeps_recording_disabled(self):
        self.c.request_reference()
        self.clock.now += 1001
        self.c.step(None)
        self.assertIsNone(self.c.pending_reference)
        self.assertIsNone(self.c.reference_id)
        self.assertFalse(self.c.toggle_recording())

    def test_pause_resume_and_camera_gaps_are_auditable(self):
        self.establish_reference()
        self.c.toggle_recording()
        self.c.step(packet(3))
        self.c.toggle_recording()
        self.c.step(packet(4))
        self.c.toggle_recording()
        self.c.step(packet(5))
        self.c.finish()
        self.assertEqual([p[0].seq for p in self.recorder.pairs], [3,5])
        names = [x[0] for x in self.recorder.events]
        self.assertEqual(names.count("recording_resume"), 2)
        self.assertEqual(names.count("recording_pause"), 2)
        self.assertIn("camera_sequence_gap", names)
        self.assertIn("collector_exit", names)

    def test_offset_is_explicit_and_default_does_not_use_profile_latency(self):
        self.assertEqual(parse_args([]).align_offset_ms, 0)
        self.assertEqual(parse_args([]).max_pair_gap_ms, 30)
        self.c.align_offset_ns = 50
        self.establish_reference()
        self.assertEqual(self.sensor.calls[-1], (1060, 30_000_000, 1))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(run_collector(["--check-config"]), 0)
        self.assertIn('"profile_validation_latency_applied": false', stream.getvalue())
        self.assertIn('"align_offset_ns": 0', stream.getvalue())

    def test_cli_aliases_preserve_existing_destinations_and_check_config(self):
        old = parse_args(["--align-offset-ms", "12.5", "--max-pair-gap-ms", "40"])
        new = parse_args(["--alignment-offset-ms", "12.5", "--max-gap-ms", "40"])
        self.assertEqual((old.align_offset_ms, old.max_pair_gap_ms),
                         (new.align_offset_ms, new.max_pair_gap_ms))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(run_collector(["--check-config", "--alignment-offset-ms", "12.5",
                                            "--max-gap-ms", "40"]), 0)
        self.assertIn('"align_offset_ns": 12500000', stream.getvalue())
        self.assertIn('"max_pair_gap_ns": 40000000', stream.getvalue())


class RecorderIntegrationTests(unittest.TestCase):
    def setUp(self):
        from sync_acquisition import ForceBuffer, SessionRecorder
        self.test_root = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(prefix="vbts_collector_test_", dir=self.test_root)
        self.recorder = SessionRecorder(self.temp.name, profile_path=FIXTURE, metadata={"offline_test": True, "align_offset_ns": 0})
        self.clock = FakeClock()
        self.sensor = FakeSensor(self.clock)
        self.buffer = ForceBuffer()
        original_zero = self.sensor.zero
        def zero(reason):
            result = original_zero(reason)
            if result:
                self.buffer.reset(self.sensor.tare_epoch, self.sensor.last_tare_end_ns)
            return result
        self.sensor.zero = zero
        self.sensor.match = self.buffer.match
        self.c = CollectorController(self.sensor, self.recorder, clock_ns=self.clock,
                                     pair_wait_ns=100, reference_wait_ns=1000)
        self.pixel_data = np.arange(12*16*3, dtype=np.uint8).reshape(12,16,3)
        self.raw_seq = 0

    def tearDown(self):
        try:
            self.recorder.close()
        except RuntimeError:
            # Failure-path tests intentionally create a failed session.
            pass
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root))
        self.temp.cleanup()

    def frame(self, seq, start, end):
        from sync_acquisition import FramePacket
        return FramePacket(seq, self.pixel_data.copy(), start, end, end+1, 1_700_000_000_000_000_000+end)

    def force(self, midpoint, fz=2):
        from sync_acquisition import ForceSample
        self.raw_seq += 1
        sample = ForceSample(self.raw_seq, midpoint-1, midpoint+1, midpoint, midpoint+100000,
                             self.raw_seq, 1, 10000, "", (0,0,fz,0,0,0), (0,0,fz,0,0,0),
                             self.sensor.tare_epoch, False, True, "OK")
        self.recorder.record_force(sample)
        self.buffer.add(sample)

    def reference(self, seq, start, end):
        self.c.request_reference()
        self.force(end-5)
        self.force(end+5)
        self.c.step(self.frame(seq, start, end))
        self.assertIsNotNone(self.c.reference_id)

    def read_csv(self, filename):
        with (self.recorder.session_dir/filename).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_lossless_png_unique_reference_raw_pause_and_reconstructable_pair(self):
        self.reference(1, 1100, 1110)
        first_ref = self.c.reference_id
        first_bytes = (self.recorder.session_dir/(first_ref+".png")).read_bytes()
        self.c.toggle_recording()
        self.force(1190, 1)
        self.force(1210, 3)
        second = self.frame(2, 1195, 1200)
        self.c.step(second)
        self.c.step(second)
        self.c.toggle_recording()
        self.force(1220, 4)  # Raw FT is independent of image pause.
        self.c.step(self.frame(3, 1225, 1230))
        self.clock.now = 2000
        self.reference(4, 2001, 2010)
        self.assertNotEqual(first_ref, self.c.reference_id)
        self.assertEqual(first_bytes, (self.recorder.session_dir/(first_ref+".png")).read_bytes())
        self.c.toggle_recording()
        self.force(2095, 2)
        self.force(2105, 6)
        self.c.step(self.frame(5, 2098, 2100))
        self.force(2190)
        self.c.step(self.frame(6, 2195, 2200))
        self.clock.now += 101
        self.c.step(None)
        self.c.finish()
        self.recorder.close()
        rows = self.read_csv("capture_sync.csv")
        self.assertEqual([int(row["frame_seq"]) for row in rows], [2,5,6])
        self.assertEqual([float(rows[i]["Fz"]) for i in (0,1)], [2,4])
        self.assertEqual(rows[2]["Fz"], "")
        self.assertEqual(rows[2]["sync_status"], "NO_RIGHT_BRACKET")
        self.assertEqual(rows[0]["reference_id"], first_ref)
        self.assertNotEqual(rows[0]["reference_id"], rows[1]["reference_id"])
        self.assertEqual(len(self.read_csv("force_raw.csv")), self.raw_seq)
        for row in rows:
            self.assertTrue(np.array_equal(cv2.imread(str(self.recorder.session_dir/row["image_filename"])), self.pixel_data))
            self.assertEqual(json.loads(row["extra_json"])["align_offset_ns"], 0)
            self.assertEqual(row["image_written"], "True")
        for row in rows[:2]:
            self.assertTrue(row["force_left_read_seq"] and row["force_right_ipoc"])
            self.assertAlmostEqual(float(row["force_interp_alpha"]), .5)
        session = json.loads((self.recorder.session_dir/"session.json").read_text(encoding="utf-8"))
        self.assertEqual(session["status"], "closed")
        self.assertEqual(session["references"], 2)
        events = [json.loads(line)["event"] for line in (self.recorder.session_dir/"events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(events.count("reference_applied"), 2)
        self.assertEqual(events.count("recording_resume"), 2)
        self.assertEqual(events.count("recording_pause"), 2)

    def test_actual_png_encoding_failure_produces_no_success_csv_row(self):
        self.reference(1, 1100, 1110)
        self.c.toggle_recording()
        self.force(1190)
        self.force(1210)
        with patch("cv2.imencode", return_value=(False, None)):
            with self.assertRaises(OSError):
                self.c.step(self.frame(2, 1195, 1200))
        self.assertEqual(self.c.saved_count, 0)
        self.assertEqual(self.read_csv("capture_sync.csv"), [])
        self.assertFalse((self.recorder.session_dir/"frame_000000002.png").exists())
        with self.assertRaises(RuntimeError):
            self.recorder.close()
        session = json.loads((self.recorder.session_dir/"session.json").read_text(encoding="utf-8"))
        self.assertEqual(session["status"], "failed")
        self.assertEqual(session["recorded_pairs"], 0)


class CollectorFailureExitTests(unittest.TestCase):
    def test_acquisition_error_marks_session_failed_and_cleanup_preserves_original(self):
        # Inject device doubles at the shared-module boundary. Only the actual
        # SessionRecorder writes files; no DLL or camera is constructed.
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="vbts_collector_exit_", dir=root) as tmp:
            sensor = SimpleNamespace(open=lambda: True, start=lambda: None,
                                     tare_epoch=0, last_tare_end_ns=None)
            def first_failure():
                raise RuntimeError("original FT read failure")
            def cleanup_failure():
                raise OSError("secondary close failure")
            sensor.check_error = first_failure
            sensor.close = cleanup_failure
            camera = SimpleNamespace(start=lambda: True, release=lambda: None)
            with patch("sync_acquisition.SixAxisForceSensor", return_value=sensor), \
                 patch("sync_acquisition.ZeroBufferCamera", return_value=camera), \
                 patch("cv2.namedWindow"), patch("cv2.destroyAllWindows") as destroy, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "^original FT read failure$"):
                    run_collector(["--output-dir", tmp, "--ft-dll", str(root/"collect_sync_dataset.py")])
                destroy.assert_called_once()
            sessions = list(Path(tmp).glob("session_*/session.json"))
            self.assertEqual(len(sessions), 1)
            data = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "failed")
            events = [json.loads(line) for line in (sessions[0].parent/"events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(e["event"] == "session_error" and "original FT" in e["details"]["error"] for e in events))
            self.assertTrue(any(e["event"] == "collector_error" for e in events))
            self.assertTrue(any(e["event"] == "collector_exit" for e in events))
            self.assertTrue(Path(tmp).resolve().is_relative_to(root))


if __name__ == "__main__":
    unittest.main()
