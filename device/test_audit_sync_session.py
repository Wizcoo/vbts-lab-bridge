"""Offline recorder/auditor contract tests. No DLL or camera is opened."""
import base64
import csv
import ctypes
import hashlib
import importlib.util
import itertools
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from audit_sync_session import audit_session, main
from sync_acquisition import (AXES, ForceBuffer, ForceSample, FramePacket, SessionRecorder,
                              HPS_FT_Info, SixAxisForceSensor, load_force_profile,
                              resolve_profile_path)


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWJkAAAAASUVORK5CYII=")


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".audit-test-", dir=Path(__file__).resolve().parent)
        assert Path(self.temp.name).resolve().is_relative_to(Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        profile = self.base / "test_profile.txt"
        profile.write_text("FORCE_AXIS_SIGNS=1,1,1,-1,-1,-1\nFORCE_MOMENT_ARM=0.045\nFORCE_AVERAGE_COUNT=1\n", encoding="utf-8")
        tick = lambda: next(counter)
        counter = itertools.count(1_000_000_000, 1_000_000)
        def encode(recorder, path, image):
            recorder._write_bytes(path, PNG)
            return hashlib.sha256(PNG).hexdigest()
        with patch("sync_acquisition.time.perf_counter_ns", side_effect=tick), patch.object(SessionRecorder,"_image", encode):
            recorder = SessionRecorder(self.base, profile, metadata={"align_offset_ns":0,"max_pair_gap_ns":30_000_000})
            self.root = recorder.session_dir
            recorder.event("force_open_result",ret=1,axis_signs=[1.,1.,1.,-1.,-1.,-1.],moment_arm=.045,accepted_codes=[10000])
            request = tick()
            recorder.event("tare_request", tare_epoch=0, request_ns=request)
            start,end = tick(),tick()
            recorder.event("tare_result",success=True,ret=1,tare_epoch=1,request_ns=request,start_ns=start,end_ns=end)
            buffer = ForceBuffer()
            buffer.reset(1,end)
            def sample(seq, values):
                read_start,read_end = tick(),tick()
                transformed = list(values[:3])+[-v for v in values[3:]]
                transformed[3] += transformed[1]*.045
                transformed[4] -= transformed[0]*.045
                result = ForceSample(seq,read_start,read_end,(read_start+read_end)//2,tick(),100+seq,1,10000,"",tuple(values),tuple(transformed),1,False,True,"OK")
                recorder.record_force(result)
                buffer.add(result)
                return result
            sample(1,[1.,2.,3.,4.,5.,6.])
            packet = FramePacket(1,None,tick(),tick(),tick(),tick())
            sample(2,[2.,3.,4.,5.,6.,7.])
            match = buffer.match(packet.grab_end_ns,30_000_000,1)
            reference = recorder.save_reference(packet,match)
            recorder.record_pair(packet,match,reference_id=reference)
            invalid_packet = FramePacket(2,None,tick(),tick(),tick(),tick())
            invalid_match = buffer.match(invalid_packet.grab_end_ns,30_000_000,1)
            self.assertEqual(invalid_match.status,"NO_RIGHT_BRACKET")
            recorder.record_pair(invalid_packet,invalid_match,reference_id=reference)
            recorder.close()

    def change_csv(self, filename, mutate):
        path = self.root / filename
        with path.open(newline="",encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields,rows = reader.fieldnames,list(reader)
        mutate(rows)
        with path.open("w",newline="",encoding="utf-8") as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)

    def assert_error(self, code):
        result = audit_session(self.root)
        self.assertFalse(result["passed"],result)
        self.assertIn(code,[e["code"] for e in result["errors"]],result)

    def test_closed_fake_session_reconstructs(self):
        before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()}
        result = audit_session(self.root)
        self.assertTrue(result["passed"],result)
        self.assertEqual(result["counts"]["valid_labels"],1)
        self.assertEqual(result["counts"]["invalid_labels"],1)
        self.assertEqual(result["sync_success_rate"],.5)
        self.assertFalse(result["hardware_synchronization_proven"])
        self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()})

    def test_changed_label_fails(self):
        self.change_csv("capture_sync.csv",lambda rows: rows[0].update(Fz="123.4"))
        self.assert_error("LABEL_MISMATCH")

    def test_changed_alpha_fails(self):
        self.change_csv("capture_sync.csv",lambda rows: rows[0].update(force_interp_alpha="0.99"))
        self.assert_error("ALPHA_MISMATCH")

    def test_changed_image_fails(self):
        (self.root / "frame_000000001.png").write_bytes(PNG+b"tampered")
        self.assert_error("IMAGE_HASH_MISMATCH")

    def test_changed_reference_epoch_fails(self):
        path = self.root / "reference_000001.json"
        obj = json.loads(path.read_text(encoding="utf-8")); obj["tare_epoch"] = 2
        path.write_text(json.dumps(obj),encoding="utf-8")
        self.assert_error("PAIR_REFERENCE_EPOCH")

    def test_changed_pair_epoch_fails(self):
        self.change_csv("capture_sync.csv",lambda rows: rows[0].update(tare_epoch="0"))
        self.assert_error("PAIR_REFERENCE_EPOCH")

    def test_invalid_row_fake_zero_fails(self):
        self.change_csv("capture_sync.csv",lambda rows: rows[1].update({axis:"0" for axis in AXES}))
        self.assert_error("INVALID_LABEL_FILLED")

    def test_changed_raw_coordinate_transform_fails(self):
        self.change_csv("force_raw.csv",lambda rows: rows[0].update(transformed_Mx="0"))
        self.assert_error("RAW_TRANSFORM")

    def test_missing_reference_fails(self):
        (self.root / "reference_000001.json").unlink()
        self.assert_error("MISSING_REFERENCE")

    def test_duplicate_frame_fails(self):
        self.change_csv("capture_sync.csv",lambda rows: rows[1].update(frame_seq="1"))
        self.assert_error("DUPLICATE_OR_UNORDERED_FRAME")

    def test_endpoint_ipoc_tamper_fails(self):
        self.change_csv("capture_sync.csv",lambda rows:rows[0].update(force_left_ipoc="999"))
        self.assert_error("RAW_ENDPOINT_METADATA")

    def test_missing_tare_event_fails(self):
        path = self.root / "events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text("\n".join(json.dumps(e) for e in events if e["event"] != "tare_result")+"\n",encoding="utf-8")
        self.assert_error("TARE_EVENT_PAIR_COUNT")

    def test_old_schema_explicitly_rejected(self):
        path = self.root / "session.json"
        obj = json.loads(path.read_text(encoding="utf-8")); obj["schema_version"] = 1
        path.write_text(json.dumps(obj),encoding="utf-8")
        self.assert_error("UNSUPPORTED_SCHEMA")

    def test_report_cannot_overwrite(self):
        path = self.base / "report.json"
        path.write_text("keep",encoding="utf-8")
        with self.assertRaises(FileExistsError):
            main([str(self.root),"--report",str(path)])
        self.assertEqual(path.read_text(encoding="utf-8"),"keep")

    def test_manifest_excludes_invalid_and_preserves_session_group(self):
        path = self.base / "training.jsonl"
        with patch("sys.stdout", new=io.StringIO()):
            code = main([str(self.root),"--manifest",str(path)])
        self.assertEqual(code,0)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["split"],"unassigned")
        self.assertEqual(rows[0]["group_id"],rows[0]["session_id"])
        self.assertTrue(Path(rows[0]["image_path"]).is_absolute())
        self.assertTrue(Path(rows[0]["reference_path"]).is_absolute())

    def test_failed_audit_never_exports_manifest(self):
        path = self.base / "training.jsonl"
        self.change_csv("capture_sync.csv",lambda rows:rows[0].update(Fz="123"))
        with patch("sys.stdout",new=io.StringIO()):
            code = main([str(self.root),"--manifest",str(path)])
        self.assertEqual(code,1)
        self.assertFalse(path.exists())

    def test_real_validator_controller_recorder_audit_roundtrip(self):
        # Load the real controller; stub only the unused GUI/PNG dependency.
        # An injected fake DLL exercises the actual sensor/recorder code without
        # ever loading a shared library, opening a camera, or changing a profile.
        __import__("numpy")  # Keep NumPy loaded across the temporary cv2 stub.
        spec = importlib.util.spec_from_file_location("audit_test_main_match",Path(__file__).with_name("main_match.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"cv2":SimpleNamespace()}), patch("ctypes.CDLL",side_effect=AssertionError("Hardware DLL loading forbidden")):
            spec.loader.exec_module(module)
        class Function:
            def __init__(self, callback): self.callback = callback
            def __call__(self,*args): return self.callback(*args)
        class FakeDLL:
            def __init__(self):
                self.ipoc = 100
                for name in ("hps_ft_initial","hps_ft_uninitial","hps_ft_deleteHandle","hps_ft_zero"):
                    setattr(self,name,Function(lambda *args:1))
                self.hps_ft_createHandle = Function(lambda *args:7)
                self.hps_ft_getData = Function(self.read)
            def read(self, handle, values, info_ptr):
                self.ipoc += 1
                for index in range(6): values[index] = index+self.ipoc/100.
                info = ctypes.cast(info_ptr,ctypes.POINTER(HPS_FT_Info)).contents
                info.code,info.IPOC = 10000,self.ipoc
                return 1
        class FakePipeline:
            def __init__(self): self.resets,self.frames = 0,0
            def reset_reference(self): self.resets += 1; return 0
            def process_frame(self,image):
                self.frames += 1
                return SimpleNamespace(force_valid=1,force_ready=1,force_status=b"OK",force=[0.]*6,raw_force=[0.]*6,frame_id=self.frames,timestamp_ms=self.frames*8.3,contact_detected=False)
        profile_path = self.base / "test_profile.txt"
        original_profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        profile = load_force_profile(profile_path)
        counter = itertools.count(10_000_000_000,1_000_000)
        tick = lambda:next(counter)
        def encode(recorder,path,image):
            recorder._write_bytes(path,PNG)
            return hashlib.sha256(PNG).hexdigest()
        # The main entry point records offset in ms; its alignment_policy event
        # carries max_gap_ns, and each pair uses alignment_offset_ns in extra.
        metadata = {"tool":"main_match","alignment_offset_ms":1.0,
                    "axis_signs":profile["axis_signs"],"moment_arm_m":profile["moment_arm"],
                    "force_average_count_applied":1}
        with patch("sync_acquisition.time.perf_counter_ns",side_effect=tick), patch.object(SessionRecorder,"_image",encode), patch("ctypes.CDLL",side_effect=AssertionError("Hardware DLL loading forbidden")):
            recorder = SessionRecorder(self.base/"controller",profile_path,metadata)
            sensor = SixAxisForceSensor("INJECTED_FAKE_ONLY",axis_signs=profile["axis_signs"],moment_arm=profile["moment_arm"],recorder=recorder,dll=FakeDLL())
            pipeline = FakePipeline()
            try:
                self.assertTrue(sensor.open())
                controller = module.ValidatorController(pipeline,sensor,recorder,alignment_offset_ms=1.,max_gap_ms=30.,clock_ns=tick)
                self.assertTrue(controller.request_tare())
                sensor.read_once()
                packet = FramePacket(1,None,tick(),tick(),tick(),tick())
                sensor.read_once()
                controller.observe(packet)
                result = controller.drain()
                self.assertEqual(len(result),1)
                self.assertTrue(result[0]["match"].valid)
                self.assertEqual(pipeline.resets,1)
                # A second frame has a fresh bracket but reuses the epoch's ref.
                sensor.read_once()
                second = FramePacket(2,None,tick(),tick(),tick(),tick())
                sensor.read_once()
                controller.observe(second)
                self.assertEqual(len(controller.drain()),1)
            finally:
                sensor.close()
                recorder.close()
        report = audit_session(recorder.session_dir)
        self.assertTrue(report["passed"],report)
        self.assertEqual(report["counts"]["valid_labels"],2)
        self.assertEqual(report["counts"]["references"],1)
        self.assertEqual(report["event_counts"]["alignment_policy"],1)
        self.assertFalse(report["hardware_synchronization_proven"])
        with (recorder.session_dir/"capture_sync.csv").open(encoding="utf-8",newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(json.loads(rows[0]["extra_json"])["alignment_offset_ns"],1_000_000)
        self.assertEqual(int(rows[0]["target_ns"]),int(rows[0]["grab_end_ns"])-1_000_000)
        self.assertEqual(hashlib.sha256(profile_path.read_bytes()).hexdigest(),original_profile_hash)


if __name__ == "__main__":
    unittest.main()
