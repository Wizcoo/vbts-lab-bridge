"""Traceable host-clock acquisition. No hardware is opened by importing this module.

Camera grab completion and FT DLL-call midpoint are software timestamps, NOT
exposure/sample hardware timestamps. Pairing never extrapolates or fills zeros.
The caller chooses target_ns = frame.grab_end_ns - alignment_offset_ns; no
physical delay is inferred here. Raw reads, including failures and duplicate
IPOCs, remain available for later synchronization diagnosis.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import csv
import ctypes
import hashlib
import json
import math
import os
import threading
import time
import uuid

AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
UNITS = ("N", "N", "N", "N*m", "N*m", "N*m")
PROFILE_NAME = "HSMVBTS_G01_H20_000011.txt"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _json(value):
    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def resolve_profile_path(directory=None):
    directory = Path(directory or Path(__file__).parent)
    candidates = [directory / PROFILE_NAME, directory / "model" / PROFILE_NAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No profile found in {candidates}")


def load_force_profile(path):
    """Read the existing coordinate contract; never edit the profile."""
    entries = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", ";")) and "=" in line:
            key, value = line.split("=", 1)
            entries[key.strip()] = value.strip()
    signs = tuple(float(v) for v in entries["FORCE_AXIS_SIGNS"].split(","))
    arm = float(entries["FORCE_MOMENT_ARM"])
    if len(signs) != 6 or not all(v in (-1., 1.) for v in signs) or not math.isfinite(arm):
        raise ValueError("Invalid force axis signs or moment arm in profile")
    return {"axis_signs": signs, "moment_arm": arm,
            "configured_average_count": int(entries.get("FORCE_AVERAGE_COUNT", "1"))}


@dataclass(frozen=True)
class FramePacket:
    seq: int
    image: object
    grab_start_ns: int
    grab_end_ns: int
    receive_ns: int
    wall_ns: int
    exposure_ns: int | None = None

    def metadata(self):
        return {"frame_seq": self.seq, "grab_start_ns": self.grab_start_ns,
                "grab_end_ns": self.grab_end_ns, "receive_ns": self.receive_ns,
                "frame_wall_ns": self.wall_ns, "exposure_ns": self.exposure_ns}


class HPS_FT_Info(ctypes.Structure):
    # inc/hps_ft_lib.h hps_ft_info_t: enum(int), int, char[256], SDK 1.2.0.
    _fields_ = [("code", ctypes.c_int), ("IPOC", ctypes.c_int), ("code_info", ctypes.c_char * 256)]


if (ctypes.sizeof(HPS_FT_Info), HPS_FT_Info.IPOC.offset, HPS_FT_Info.code_info.offset) != (264, 4, 8):
    raise RuntimeError("Unexpected HPS FT ABI; require 32-bit int fields and 264-byte info")


@dataclass(frozen=True)
class ForceSample:
    read_seq: int
    read_start_ns: int
    read_end_ns: int
    midpoint_ns: int
    wall_ns: int
    ipoc: int
    ret: int
    code: int
    code_info: str
    raw: tuple
    transformed: tuple | None
    tare_epoch: int
    duplicate_ipoc: bool
    valid: bool
    status: str

    def as_dict(self):
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ForceMatch:
    status: str
    target_ns: int
    tare_epoch: int
    left: ForceSample | None = None
    right: ForceSample | None = None
    alpha: float | None = None
    label: tuple | None = None
    epoch_start_ns: int | None = None

    @property
    def valid(self):
        return self.status == "OK" and self.label is not None

    @property
    def gap_ns(self):
        return self.right.midpoint_ns - self.left.midpoint_ns if self.left and self.right else None

    def as_dict(self):
        return {"sync_status": self.status, "valid": self.valid, "target_ns": self.target_ns,
                "tare_epoch": self.tare_epoch, "epoch_start_ns": self.epoch_start_ns,
                "force_interp_alpha": self.alpha, "force_bracket_gap_ns": self.gap_ns,
                "label": _jsonable(self.label), "left": self.left.as_dict() if self.left else None,
                "right": self.right.as_dict() if self.right else None,
                "timestamp_semantics": "FT DLL-call host midpoint; camera host grab completion; no hardware exposure/sample timestamps"}


class ForceBuffer:
    def __init__(self, maxlen=4000):
        self.samples = deque(maxlen=maxlen)
        self.condition = threading.Condition(threading.RLock())
        self.tare_epoch = 0
        self.epoch_start_ns = None

    def reset(self, tare_epoch, epoch_start_ns):
        with self.condition:
            self.samples.clear()
            self.tare_epoch = int(tare_epoch)
            self.epoch_start_ns = int(epoch_start_ns)
            self.condition.notify_all()

    def add(self, sample):
        with self.condition:
            if sample.tare_epoch != self.tare_epoch:
                raise ValueError("Cannot add a force sample from another tare epoch")
            if sample.duplicate_ipoc:
                return False  # Preserve in raw log, never treat a repeated DLL value as a new sample.
            if self.samples and sample.midpoint_ns <= self.samples[-1].midpoint_ns:
                raise ValueError("Force host timestamps must be strictly increasing")
            self.samples.append(sample)  # Failed reads remain barriers to interpolation.
            self.condition.notify_all()
            return True

    def match(self, target_ns, max_gap_ns=30_000_000, expected_tare_epoch=None):
        target_ns = int(target_ns)
        if max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive")
        with self.condition:
            epoch = self.tare_epoch
            base = dict(target_ns=target_ns, tare_epoch=epoch, epoch_start_ns=self.epoch_start_ns)
            if expected_tare_epoch is not None and epoch != expected_tare_epoch:
                return ForceMatch("TARE_EPOCH_MISMATCH", **base)
            if self.epoch_start_ns is not None and target_ns < self.epoch_start_ns:
                return ForceMatch("BEFORE_TARE_EPOCH", **base)
            values = list(self.samples)
            if not values:
                return ForceMatch("NO_FORCE_SAMPLES", **base)
            index = bisect_right([v.midpoint_ns for v in values], target_ns)
            # Exact endpoint still needs two distinct samples; no one-sided fallback.
            if index == 0:
                return ForceMatch("NO_LEFT_BRACKET", right=values[0], **base)
            if index == len(values):
                return ForceMatch("NO_RIGHT_BRACKET", left=values[-1], **base)
            left, right = values[index - 1], values[index]
            endpoints = dict(left=left, right=right, **base)
            if left.tare_epoch != epoch or right.tare_epoch != epoch:
                return ForceMatch("BRACKET_TARE_EPOCH_MISMATCH", **endpoints)
            if not left.valid or not right.valid:
                return ForceMatch("INVALID_FORCE_BRACKET", **endpoints)
            if left.duplicate_ipoc or right.duplicate_ipoc or left.ipoc == right.ipoc:
                return ForceMatch("DUPLICATE_IPOC_BRACKET", **endpoints)
            gap = right.midpoint_ns - left.midpoint_ns
            if gap <= 0 or gap > max_gap_ns:
                return ForceMatch("FORCE_BRACKET_GAP_TOO_LARGE", **endpoints)
            alpha = (target_ns - left.midpoint_ns) / gap
            label = tuple((1 - alpha) * a + alpha * b for a, b in zip(left.transformed, right.transformed))
            if not all(math.isfinite(x) for x in label):
                return ForceMatch("NONFINITE_INTERPOLATED_FORCE", **endpoints)
            return ForceMatch("OK", alpha=alpha, label=label, **endpoints)


class ZeroBufferCamera:
    def __init__(self, camera_index=0, width=640, height=480, *, capture=None):
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if capture is None:
            import cv2
            capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = capture
        self.lock = threading.Lock()
        self.latest_packet = None
        self.running = False
        self.thread = None
        self.error = None
        self._seq = 0

    def start(self):
        if not self.cap.isOpened():
            return False
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._loop, name="camera-acquisition", daemon=True)
            self.thread.start()
        return True

    def _loop(self):
        try:
            while self.running:
                begin = time.perf_counter_ns()
                grabbed = self.cap.grab()
                end = time.perf_counter_ns()
                if not grabbed:
                    time.sleep(.002)
                    continue
                ok, image = self.cap.retrieve()
                receive = time.perf_counter_ns()
                wall = time.time_ns()
                if ok and image is not None:
                    if image.shape != (self.height, self.width, 3) or str(image.dtype) != "uint8":
                        raise ValueError(f"Camera returned {image.shape}/{image.dtype}; expected "
                                         f"({self.height},{self.width},3)/uint8 BGR; no implicit resize")
                    self._seq += 1
                    packet = FramePacket(self._seq, image.copy(), begin, end, receive, wall)
                    with self.lock:
                        self.latest_packet = packet
        except Exception as exc:
            self.error = exc
            self.running = False

    def check_error(self):
        if self.error is not None:
            raise RuntimeError("Camera acquisition failed") from self.error

    def read_latest(self):
        self.check_error()
        with self.lock:
            p = self.latest_packet
            return replace(p, image=p.image.copy()) if p is not None else None

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("Camera thread did not stop; capture was not released concurrently")
        self.cap.release()
        self.check_error()


class SixAxisForceSensor:
    def __init__(self, dll_path, baud_rate=115200, port=b"", average_count=1,
                 axis_signs=(1., 1., 1., -1., -1., -1.), moment_arm=.045,
                 recorder=None, *, dll=None, accepted_codes=(10000,), poll_interval_sec=.001):
        if average_count != 1:
            raise ValueError("Raw acquisition requires average_count=1; no hidden averaging")
        self.axis_signs = tuple(float(x) for x in axis_signs)
        self.moment_arm = float(moment_arm)
        if len(self.axis_signs) != 6 or not all(x in (-1., 1.) for x in self.axis_signs) or not math.isfinite(self.moment_arm):
            raise ValueError("Invalid force coordinate contract")
        self.dll_path = os.path.abspath(str(dll_path))
        self.baud_rate = int(baud_rate)
        self.port = port.encode() if isinstance(port, str) else port
        self.recorder = recorder
        self.accepted_codes = tuple(int(c) for c in accepted_codes)
        self.poll_interval_sec = max(0., float(poll_interval_sec))
        if dll is not None:
            self.dll = dll
        else:
            dll_dir = os.path.dirname(self.dll_path)
            if hasattr(os, 'add_dll_directory') and os.path.isdir(dll_dir):
                try:
                    os.add_dll_directory(dll_dir)
                except Exception:
                    pass
            self.dll = ctypes.CDLL(self.dll_path)
        self.handle = -1
        self.ft_ready = False
        self.running = False
        self.thread = None
        self.error = None
        self.lock = threading.RLock()  # Every DLL read and tare share this lock.
        self.buffer = ForceBuffer()
        self.tare_epoch = 0
        self.last_tare_end_ns = None
        self.last_tare_result = None
        self._read_seq = 0
        self._last_ipoc = None
        self._setup_prototypes()

    def _setup_prototypes(self):
        c = ctypes
        declarations = {"hps_ft_createHandle": [c.c_int], "hps_ft_initial": [c.c_int, c.c_int, c.c_char_p],
                        "hps_ft_uninitial": [c.c_int], "hps_ft_deleteHandle": [c.POINTER(c.c_int)],
                        "hps_ft_getData": [c.c_int, c.POINTER(c.c_double), c.POINTER(HPS_FT_Info)],
                        "hps_ft_zero": [c.c_int]}
        for name, args in declarations.items():
            function = getattr(self.dll, name)
            function.restype = c.c_int
            function.argtypes = args

    def _event(self, kind, **fields):
        if self.recorder is not None:
            self.recorder.event(kind, **fields)

    def open(self):
        with self.lock:
            if self.ft_ready:
                return True
            self.handle = int(self.dll.hps_ft_createHandle(0))
            if self.handle == -1:
                self._event("force_open_failed", stage="create_handle")
                return False
            ret = int(self.dll.hps_ft_initial(self.handle, self.baud_rate, self.port))
            self._event("force_open_result", ret=ret, handle=self.handle, baud_rate=self.baud_rate,
                        axis_signs=self.axis_signs, moment_arm=self.moment_arm, average_count=1,
                        accepted_codes=self.accepted_codes, tare_epoch=self.tare_epoch,
                        dll_path=self.dll_path, abi_info_size=ctypes.sizeof(HPS_FT_Info), automatic_tare=False)
            if ret != 1:
                handle = ctypes.c_int(self.handle)
                self.dll.hps_ft_deleteHandle(ctypes.byref(handle))
                self.handle = -1
                return False
            self.ft_ready = True
            return True

    def zero(self, reason="user"):
        with self.lock:
            self.check_error()
            request_ns = time.perf_counter_ns()
            self._event("tare_request", reason=reason, tare_epoch=self.tare_epoch, request_ns=request_ns)
            start = time.perf_counter_ns()
            try:
                ret = int(self.dll.hps_ft_zero(self.handle)) if self.ft_ready else 0
            except Exception as exc:
                self.last_tare_result = dict(success=False, ret=None, reason=reason,
                    request_ns=request_ns, start_ns=start, end_ns=time.perf_counter_ns(),
                    tare_epoch=self.tare_epoch, error=repr(exc))
                self._event("tare_result", **self.last_tare_result)
                raise
            end = time.perf_counter_ns()
            success = ret == 1
            if success:
                self.tare_epoch += 1
                self.last_tare_end_ns = end
                # Keep the previous IPOC so a DLL-cached pre-tare packet is
                # rejected until a genuinely different sequence is observed.
                self.buffer.reset(self.tare_epoch, end)
            self.last_tare_result = dict(success=success, ret=ret, reason=reason,
                                         request_ns=request_ns, start_ns=start, end_ns=end, tare_epoch=self.tare_epoch)
            self._event("tare_result", **self.last_tare_result)
            return success

    def read_once(self):
        with self.lock:
            self.check_error()
            if not self.ft_ready:
                raise RuntimeError("FT sensor is not open")
            raw_buffer = (ctypes.c_double * 6)(*[float("nan")] * 6)
            info = HPS_FT_Info(code=0, IPOC=-1)
            start = time.perf_counter_ns()
            read_error = None
            try:
                ret = int(self.dll.hps_ft_getData(self.handle, raw_buffer, ctypes.byref(info)))
            except Exception as exc:
                ret = -1
                read_error = exc
            end = time.perf_counter_ns()
            self._read_seq += 1
            raw = tuple(float(x) for x in raw_buffer)
            duplicate = ret == 1 and self._last_ipoc is not None and info.IPOC == self._last_ipoc
            if ret == 1:
                self._last_ipoc = info.IPOC
            transformed = None
            if read_error is not None:
                status = "DLL_READ_EXCEPTION"
            elif ret != 1:
                status = "DLL_READ_FAILED"
            elif info.code not in self.accepted_codes:
                status = "SENSOR_STATUS_NOT_ACCEPTED"
            elif not all(math.isfinite(v) for v in raw):
                status = "NONFINITE_RAW_FORCE"
            elif duplicate:
                status = "DUPLICATE_IPOC"
            else:
                status = "OK"
            if ret == 1 and all(math.isfinite(v) for v in raw):
                values = [a * b for a, b in zip(raw, self.axis_signs)]
                values[3] += values[1] * self.moment_arm
                values[4] -= values[0] * self.moment_arm
                transformed = tuple(values)
            sample = ForceSample(self._read_seq, start, end, (start + end) // 2, time.time_ns(),
                                 int(info.IPOC), ret, int(info.code), repr(read_error) if read_error else bytes(info.code_info).decode("utf-8", errors="replace"),
                                 raw, transformed, self.tare_epoch, duplicate, status == "OK", status)
            if self.recorder is not None:
                self.recorder.record_force(sample)
            self.buffer.add(sample)
            if read_error is not None:
                raise read_error
            return sample

    def _producer_loop(self):
        try:
            while self.running:
                self.read_once()
                if self.poll_interval_sec:
                    time.sleep(self.poll_interval_sec)
        except Exception as exc:
            self.error = exc
            self.running = False

    def start(self):
        self.check_error()
        if not self.ft_ready:
            raise RuntimeError("Open FT sensor before starting acquisition")
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._producer_loop, name="force-acquisition", daemon=True)
            self.thread.start()

    start_polling_thread = start

    def check_error(self):
        if self.error is not None:
            raise RuntimeError("FT acquisition/logging failed") from self.error
        if self.recorder is not None:
            self.recorder.check_error()

    def match(self, target_ns, max_gap_ns=30_000_000, expected_tare_epoch=None):
        self.check_error()
        return self.buffer.match(target_ns, max_gap_ns, expected_tare_epoch)

    def close(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("FT thread did not stop; handle was not closed concurrently")
        with self.lock:
            if self.handle != -1:
                self.dll.hps_ft_uninitial(self.handle)
                handle = ctypes.c_int(self.handle)
                self.dll.hps_ft_deleteHandle(ctypes.byref(handle))
                self.handle = -1
            self.ft_ready = False
        self.check_error()


class SessionRecorder:
    """Synchronous, failure-visible log writes. No filename or reference is overwritten."""
    def __init__(self, base_dir, profile_path=None, metadata=None):
        self.lock = threading.RLock()
        self.force_lock = threading.RLock()  # Image encoding/writes never hold the raw FT writer lock.
        self.error = None
        self.closed = False
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:12]
        self.session_dir = Path(base_dir).resolve() / ("session_" + self.session_id)
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._seen_frames = set()
        self._references = {}
        self._pair_idx = 0
        self._reference_idx = 0
        profile_path = Path(profile_path or resolve_profile_path()).resolve()
        profile_bytes = profile_path.read_bytes()
        self.profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
        self._write_bytes(self.session_dir / "sensor_profile_snapshot.txt", profile_bytes)
        self.session = {"schema_version":2, "session_id":self.session_id, "status":"open",
                        "started_mono_ns":time.perf_counter_ns(), "started_wall_ns":time.time_ns(),
                        "profile_path":str(profile_path), "profile_sha256":self.profile_sha256,
                        "axis_order":AXES,"units":UNITS,"metadata":metadata or {},
                        "timestamp_semantics":{"monotonic":"time.perf_counter_ns, process host clock",
                            "wall":"time.time_ns, provenance only; not used for pairing",
                            "frame":"grab_start/end and post-retrieve receive; exposure_ns unavailable unless explicitly supplied",
                            "force":"each DLL call start/end/midpoint, not sensor hardware acquisition time"},
                        "pairing":"two distinct valid IPOCs bracketing target; same tare epoch; bounded gap; invalid labels empty",
                        "raw_force_policy":"every DLL call; no averaging, automatic zero, delay inference, or zero label filling"}
        self._write_bytes(self.session_dir / "session.json", (_json(self.session)+"\n").encode("utf-8"))
        self._events = (self.session_dir / "events.jsonl").open("x", encoding="utf-8", newline="")
        self._force_file = (self.session_dir / "force_raw.csv").open("x", encoding="utf-8", newline="")
        self._force_fields = ["session_id","read_seq","read_start_ns","read_end_ns","midpoint_ns","wall_ns",
                              "ipoc","ret","code","code_info","tare_epoch","duplicate_ipoc","valid","status"]
        self._force_fields += ["raw_"+a for a in AXES] + ["transformed_"+a for a in AXES]
        self._force_writer = csv.DictWriter(self._force_file, fieldnames=self._force_fields)
        self._force_writer.writeheader(); self._force_file.flush()
        self._pair_file = (self.session_dir / "capture_sync.csv").open("x", encoding="utf-8", newline="")
        self._pair_fields = ["session_id","pair_idx","image_filename","image_sha256","image_written","reference_id",
                             "frame_seq","grab_start_ns","grab_end_ns","receive_ns","frame_wall_ns","exposure_ns",
                             "image_save_start_ns","image_save_end_ns","target_ns","tare_epoch","epoch_start_ns",
                             "sync_status","label_valid","force_interp_alpha","force_bracket_gap_ns"]
        self._pair_fields += [f"force_{side}_{key}" for side in ("left","right") for key in
                             ("read_seq","read_start_ns","read_end_ns","midpoint_ns","ipoc","ret","code","tare_epoch","valid","status")]
        self._pair_fields += list(AXES)+["pred_"+a for a in AXES]+["prediction_json","extra_json"]
        self._pair_writer = csv.DictWriter(self._pair_file, fieldnames=self._pair_fields)
        self._pair_writer.writeheader(); self._pair_file.flush()
        self.event("session_start", profile_sha256=self.profile_sha256)

    def _write_bytes(self, path, content):
        try:
            with Path(path).open("xb") as stream:
                if stream.write(content) != len(content):
                    raise OSError("Short file write")
                stream.flush()
        except Exception as exc:
            self.error = exc
            raise

    def check_error(self):
        if self.error is not None:
            raise RuntimeError("Session recording failed; stop acquisition") from self.error
        if self.closed:
            raise RuntimeError("Session recorder is closed")

    def event(self, kind, **fields):
        with self.lock:
            self.check_error()
            try:
                entry = {"session_id":self.session_id,"event":kind,"mono_ns":time.perf_counter_ns(),
                         "wall_ns":time.time_ns(),"details":fields}
                self._events.write(_json(entry)+"\n"); self._events.flush()
                if kind == "session_error":
                    self.session["acquisition_error"] = _jsonable(fields)
            except Exception as exc:
                self.error = exc
                raise

    def record_force(self, sample):
        with self.force_lock:
            self.check_error()
            try:
                row = {k:getattr(sample,k) for k in self._force_fields if hasattr(sample,k)}
                row["session_id"] = self.session_id
                for axis, raw in zip(AXES,sample.raw): row["raw_"+axis] = raw
                if sample.transformed is not None:
                    for axis, value in zip(AXES,sample.transformed): row["transformed_"+axis] = value
                self._force_writer.writerow(row); self._force_file.flush()
            except Exception as exc:
                self.error = exc
                raise

    def _image(self, path, image):
        import cv2
        ok, encoded = cv2.imencode(".png",image)
        if not ok:
            raise OSError("PNG encoding failed")
        content = encoded.tobytes()
        self._write_bytes(path,content)
        return hashlib.sha256(content).hexdigest()

    def save_reference(self, packet, match, reason="user"):
        with self.lock:
            self.check_error()
            if not match.valid or match.left.tare_epoch != match.right.tare_epoch:
                raise ValueError("Reference requires a valid same-epoch force bracket")
            if match.epoch_start_ns is not None and packet.grab_start_ns <= match.epoch_start_ns:
                raise ValueError("Reference frame capture began before tare finished")
            self._reference_idx += 1
            reference_id = f"reference_{self._reference_idx:06d}"
            self.event("reference_requested",reference_id=reference_id,reason=reason,packet=packet.metadata(),force_match=match.as_dict())
            try:
                filename = reference_id+".png"
                digest = self._image(self.session_dir/filename,packet.image)
                entry = {"reference_id":reference_id,"image_filename":filename,"image_sha256":digest,
                         "profile_sha256":self.profile_sha256,"reason":reason,"packet":packet.metadata(),
                         "force_match":match.as_dict(),"tare_epoch":match.tare_epoch}
                self._write_bytes(self.session_dir/(reference_id+".json"),(_json(entry)+"\n").encode("utf-8"))
                self.event("reference_applied",**entry)
                self._references[reference_id] = entry
                return reference_id
            except Exception as exc:
                self.error = exc
                raise

    def record_pair(self, packet, match, reference_id=None, prediction=None, extra=None):
        with self.lock:
            self.check_error()
            if packet.seq in self._seen_frames:
                raise ValueError(f"Camera frame {packet.seq} already recorded in this session")
            if reference_id is not None:
                if reference_id not in self._references:
                    raise ValueError("Unknown reference_id")
                if self._references[reference_id]["tare_epoch"] != match.tare_epoch:
                    raise ValueError("Reference and matched force belong to different tare epochs")
            extra = extra or {}
            if set(extra).intersection(self._pair_fields):
                raise ValueError("extra must not override reserved capture fields")
            try:
                filename = f"frame_{packet.seq:09d}.png"
                save_start = time.perf_counter_ns()
                digest = self._image(self.session_dir/filename,packet.image)
                save_end = time.perf_counter_ns()
                row = {"session_id":self.session_id,"pair_idx":self._pair_idx+1,"image_filename":filename,
                       "image_sha256":digest,"image_written":True,"reference_id":reference_id,**packet.metadata(),
                       "image_save_start_ns":save_start,"image_save_end_ns":save_end,"target_ns":match.target_ns,
                       "tare_epoch":match.tare_epoch,"epoch_start_ns":match.epoch_start_ns,
                       "sync_status":match.status,"label_valid":match.valid,"force_interp_alpha":match.alpha,
                       "force_bracket_gap_ns":match.gap_ns,"prediction_json":_json(prediction),"extra_json":_json(extra)}
                for side in ("left","right"):
                    sample = getattr(match,side)
                    if sample is not None:
                        for key in ("read_seq","read_start_ns","read_end_ns","midpoint_ns","ipoc","ret","code","tare_epoch","valid","status"):
                            row[f"force_{side}_{key}"] = getattr(sample,key)
                if match.valid:
                    row.update(zip(AXES,match.label))
                pred_force = prediction.get("force") if isinstance(prediction,dict) else prediction
                if pred_force is not None and len(pred_force)==6:
                    row.update(("pred_"+a,v) for a,v in zip(AXES,pred_force))
                self._pair_writer.writerow(row); self._pair_file.flush()
                self._pair_idx += 1
                self._seen_frames.add(packet.seq)
                return row
            except Exception as exc:
                self.error = exc
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            prior_error = self.error
            try:
                if prior_error is None:
                    self.event("session_end",recorded_pairs=self._pair_idx,references=len(self._references))
            finally:
                with self.force_lock:
                    for stream in (self._force_file,self._pair_file,self._events):
                        try:
                            stream.close()
                        except Exception as exc:
                            self.error = exc
                self.session.update(status="failed" if self.error or "acquisition_error" in self.session else "closed",ended_mono_ns=time.perf_counter_ns(),
                                    ended_wall_ns=time.time_ns(),recorded_pairs=self._pair_idx,references=len(self._references),
                                    error=str(self.error) if self.error else None)
                temporary = self.session_dir/"session.final.json"
                self._write_bytes(temporary,(_json(self.session)+"\n").encode("utf-8"))
                os.replace(temporary,self.session_dir/"session.json")
                self.closed = True
            if self.error is not None:
                raise RuntimeError("Session ended with a recording error") from self.error
