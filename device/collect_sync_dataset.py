# -*- coding: utf-8 -*-
"""Traceable raw camera/FT collection. Importing never opens hardware.

T: operator confirms no contact, pauses recording, tares, then waits for a new
frame and a valid two-sided FT bracket. R: resume/pause with an explicit reference.
ESC: finish. Raw FT continues while image recording is paused.
Host call timestamps are not exposure times. No offset is inferred online.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "local_data" / "captures"
DEFAULT_PROFILE = SCRIPT_DIR / "HSMVBTS_G01_H20_000011.txt"


@dataclass
class PendingFrame:
    packet: object
    queued_ns: int
    reference_id: str
    tare_epoch: int
    recording_epoch: int


@dataclass
class PendingReference:
    tare_epoch: int
    after_seq: int
    after_ns: int
    deadline_ns: int
    candidate: object | None = None
    candidate_since_ns: int = 0


class CollectorController:
    """Hardware-independent state machine for offline fake-device tests."""

    def __init__(self, sensor, recorder, *, align_offset_ns=0,
                 max_gap_ns=30_000_000, pair_wait_ns=150_000_000,
                 reference_wait_ns=3_000_000_000, max_pending_frames=256,
                 clock_ns=time.perf_counter_ns):
        self.sensor, self.recorder = sensor, recorder
        self.align_offset_ns = int(align_offset_ns)
        self.max_gap_ns, self.pair_wait_ns = int(max_gap_ns), int(pair_wait_ns)
        self.reference_wait_ns = int(reference_wait_ns)
        self.max_pending_frames = int(max_pending_frames)
        self.clock_ns = clock_ns
        self.recording = False
        self.reference_id = self.reference_epoch = None
        self.pending_reference = None
        self.last_seen_seq = -1
        self.recording_epoch = self.saved_count = 0
        self.pending = collections.deque()
        self.last_match = None
        self.status_text = "No reference: unload sensor, then press T."

    def _match(self, packet, epoch):
        return self.sensor.match(int(packet.grab_end_ns) - self.align_offset_ns,
                                 max_gap_ns=self.max_gap_ns,
                                 expected_tare_epoch=epoch)

    def _pause(self, reason):
        if self.recording:
            self.recording = False
            self.recorder.event("recording_pause", reason=reason,
                                recording_epoch=self.recording_epoch,
                                reference_id=self.reference_id,
                                last_observed_frame_seq=self.last_seen_seq)

    def toggle_recording(self):
        if self.recording:
            self._pause("R")
            self.status_text = "Paused. Raw FT is still being logged."
            return False
        if self.reference_id is None or self.pending_reference is not None:
            self.recorder.event("recording_resume_rejected", reason="reference_required",
                                last_observed_frame_seq=self.last_seen_seq)
            self.status_text = "Cannot record: unload sensor and press T for a reference."
            return False
        if self.sensor.tare_epoch != self.reference_epoch:
            self.reference_id = self.reference_epoch = None
            self.recorder.event("recording_resume_rejected", reason="reference_tare_epoch_mismatch")
            self.status_text = "Reference invalid after tare. Press T again."
            return False
        self.recording = True
        self.recording_epoch += 1
        self.recorder.event("recording_resume", reason="R",
                            recording_epoch=self.recording_epoch,
                            reference_id=self.reference_id, tare_epoch=self.reference_epoch,
                            last_observed_frame_seq=self.last_seen_seq)
        self.status_text = "Recording unique frames; raw FT continues independently."
        return True

    def request_reference(self):
        """Never apply the already displayed pre-tare frame as a reference."""
        self._pause("T")
        self._flush_pending(force=True)
        previous = self.reference_id
        self.reference_id = self.reference_epoch = None
        self.pending_reference = None
        self.recorder.event("reference_request", reason="T", operator_unloaded_confirmation=True,
                            previous_reference_id=previous, after_frame_seq=self.last_seen_seq)
        # Shared FT owner serializes zero against getData and logs its result.
        if not self.sensor.zero(reason="collector_T"):
            self.status_text = "Tare failed; no reference applied. Check FT, then retry T."
            self.recorder.event("reference_failed", reason="tare_failed")
            return False
        self.pending_reference = PendingReference(
            self.sensor.tare_epoch, self.last_seen_seq, self.sensor.last_tare_end_ns,
            self.clock_ns() + self.reference_wait_ns)
        self.status_text = "Tare succeeded; waiting for a fresh frame and two-sided FT bracket."
        return True

    def _progress_reference(self, packet, is_new, now):
        pending = self.pending_reference
        if pending is None:
            return
        if now >= pending.deadline_ns:
            self.recorder.event("reference_failed", reason="fresh_frame_or_force_timeout",
                                tare_epoch=pending.tare_epoch,
                                candidate_frame_seq=(pending.candidate.seq if pending.candidate else None))
            self.pending_reference = None
            self.status_text = "Reference timed out; recording stays paused. Retry T."
            return
        if (pending.candidate is None and is_new and packet is not None
                and packet.seq > pending.after_seq
                and packet.grab_start_ns > pending.after_ns):
            pending.candidate, pending.candidate_since_ns = packet, now
        if pending.candidate is None:
            return
        match = self._match(pending.candidate, pending.tare_epoch)
        self.last_match = match
        if match.valid:
            try:
                reference_id = self.recorder.save_reference(pending.candidate, match, reason="collector_T")
            except Exception as exc:
                self.recorder.event("reference_failed", reason="reference_write_failed",
                                    error=repr(exc), frame_seq=pending.candidate.seq)
                self.pending_reference = None
                self.status_text = "Reference write failed; recording stays paused."
                raise
            self.reference_id, self.reference_epoch = reference_id, pending.tare_epoch
            self.pending_reference = None
            self.status_text = f"Reference {reference_id} ready. Press R to record."
        elif now - pending.candidate_since_ns >= self.pair_wait_ns:
            self.recorder.event("reference_candidate_rejected", reason=match.status,
                                frame_seq=pending.candidate.seq, tare_epoch=pending.tare_epoch)
            # Retain a candidate while its right FT sample arrives. Replacing it
            # with every latest frame would perpetually chase the right bracket.
            pending.candidate = None

    def _flush_pending(self, *, force=False):
        now = self.clock_ns()
        while self.pending:
            pending = self.pending[0]
            match = self._match(pending.packet, pending.tare_epoch)
            self.last_match = match
            if not match.valid and not force and now - pending.queued_ns < self.pair_wait_ns:
                break
            # ForceMatch keeps invalid labels empty; never zero/nearest fill.
            self.pending.popleft()  # One write attempt per sequence, including failures.
            self.recorder.record_pair(pending.packet, match, reference_id=pending.reference_id,
                                      extra={"align_offset_ns": self.align_offset_ns,
                                             "max_pair_gap_ns": self.max_gap_ns,
                                             "recording_epoch": pending.recording_epoch,
                                             "pair_wait_elapsed_ns": max(0, now-pending.queued_ns)})
            self.saved_count += 1

    def step(self, packet):
        """Progress pending brackets even if camera returns the same frame/None."""
        now = self.clock_ns()
        is_new = packet is not None and packet.seq > self.last_seen_seq
        if is_new:
            if self.last_seen_seq >= 0 and packet.seq > self.last_seen_seq + 1:
                self.recorder.event("camera_sequence_gap", previous_seq=self.last_seen_seq,
                                    next_seq=packet.seq, recording=self.recording)
            self.last_seen_seq = packet.seq
        self._progress_reference(packet, is_new, now)
        if is_new and self.recording:
            if self.sensor.tare_epoch != self.reference_epoch:
                self._pause("unexpected_tare_epoch")
                self.reference_id = self.reference_epoch = None
                self.status_text = "FT epoch changed; recording paused. Press T."
            else:
                if len(self.pending) >= self.max_pending_frames:
                    self.recorder.event("recording_error", reason="pending_frame_queue_overflow")
                    raise RuntimeError("Pending frame queue overflow; refusing silent frame loss")
                self.pending.append(PendingFrame(packet, now, self.reference_id,
                                                 self.reference_epoch, self.recording_epoch))
        self._flush_pending()

    def finish(self, reason="exit"):
        self._pause(reason)
        self._flush_pending(force=True)
        if self.pending_reference is not None:
            self.recorder.event("reference_failed", reason="exit_before_reference_ready")
            self.pending_reference = None
        self.recorder.event("collector_exit", reason=reason, saved_count=self.saved_count,
                            last_observed_frame_seq=self.last_seen_seq, reference_id=self.reference_id)


def read_force_profile(path):
    fields = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    signs = tuple(float(x) for x in fields["FORCE_AXIS_SIGNS"].split(","))
    arm = float(fields["FORCE_MOMENT_ARM"])
    if len(signs) != 6 or not all(x in (-1.0, 1.0) for x in signs) or not math.isfinite(arm):
        raise ValueError("Profile requires six +/-1 FORCE_AXIS_SIGNS and a finite moment arm")
    return fields, signs, arm


def default_ft_dll():
    candidates = [SCRIPT_DIR / "bin" / "hps_ft.dll",
                  SCRIPT_DIR.parents[2] / "common" / "bin" / "hps_ft.dll"]
    return next((p for p in candidates if p.is_file()), candidates[0])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--ft-dll", type=Path, default=default_ft_dll())
    parser.add_argument("--port", default="", help="FT port; empty uses SDK discovery")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--align-offset-ms", "--alignment-offset-ms", dest="align_offset_ms", type=float, default=0.0,
                        help="Explicit diagnostic offset: target=frame grab_end-offset (default 0; not calibrated)")
    parser.add_argument("--max-pair-gap-ms", "--max-gap-ms", dest="max_pair_gap_ms", type=float, default=30.0)
    parser.add_argument("--pair-wait-ms", type=float, default=150.0)
    parser.add_argument("--reference-wait-ms", type=float, default=3000.0)
    parser.add_argument("--check-config", action="store_true", help="Print config without opening hardware")
    args = parser.parse_args(argv)
    for name in ("align_offset_ms", "max_pair_gap_ms", "pair_wait_ms", "reference_wait_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or (name != "align_offset_ms" and value <= 0):
            parser.error(f"{name.replace('_', '-')} must be finite" + (" and > 0" if name != "align_offset_ms" else ""))
    return args


def run_collector(argv=None):
    args = parse_args(argv)
    profile, signs, arm = read_force_profile(args.profile)
    from inventory_device import inventory
    config = {
        "entrypoint": "collect_sync_dataset.py", "entrypoint_schema_version": 2,
        "camera_index": args.camera_index, "camera_width_requested": 640,
        "camera_height_requested": 480, "camera_fps_requested": None,
        "ft_dll_path": str(args.ft_dll.resolve()), "ft_port": args.port,
        "ft_baud_rate": args.baud_rate, "axis_names": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
        "units": ["N", "N", "N", "Nm", "Nm", "Nm"],
        "force_axis_signs": signs, "force_moment_arm_m": arm,
        "force_transform": "signed=raw*axis_signs; Mx+=Fy*arm; My-=Fx*arm",
        "profile_force_average_count": profile.get("FORCE_AVERAGE_COUNT"),
        "raw_reads_per_sample": 1, "force_averaging_applied": False,
        "profile_validation_latency_ms": profile.get("VALIDATION_LATENCY_MS"),
        "profile_validation_latency_applied": False,
        "align_offset_ns": round(args.align_offset_ms * 1_000_000),
        "max_pair_gap_ns": round(args.max_pair_gap_ms * 1_000_000),
        "pair_wait_ns": round(args.pair_wait_ms * 1_000_000),
        "reference_wait_ns": round(args.reference_wait_ms * 1_000_000),
        "pair_target": "frame.grab_end_ns - align_offset_ns",
        "image_format": "PNG lossless BGR camera pixels; no resize/overlays",
        "reference_policy": "explicit operator T; tare success; grab_start after tare_end; same epoch bracket",
        "timestamp_limitation": "Host perf_counter_ns call times, not measured exposure or hardware-synchronized time",
        "bridge_inventory": inventory(args.profile, args.ft_dll),
    }
    if args.check_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print(f"FT DLL exists: {args.ft_dll.is_file()}")
        return 0
    if not args.ft_dll.is_file():
        raise FileNotFoundError(f"FT DLL missing: {args.ft_dll}. Set --ft-dll explicitly.")
    from sync_acquisition import SessionRecorder, SixAxisForceSensor, ZeroBufferCamera
    recorder = SessionRecorder(args.output_dir, profile_path=args.profile, metadata=config)
    sensor = camera = controller = None
    exit_reason = "ESC"
    acquisition_error = None
    win_name = "HSMVBTS Traceable Collector | T: tare/reference | R: record | ESC: exit"
    print(f"数据目录: {recorder.session_dir}")
    print("确认无接触后按 T 建立参考；R 开始/暂停。原始力流全程记录，无损 PNG 保存图像。")
    print(f"配对偏移 {args.align_offset_ms:g} ms；未应用 profile 的 VALIDATION_LATENCY_MS。")
    try:
        sensor = SixAxisForceSensor(str(args.ft_dll), baud_rate=args.baud_rate,
                                    port=args.port.encode("ascii"), axis_signs=signs,
                                    moment_arm=arm, recorder=recorder, average_count=1)
        if not sensor.open():
            raise RuntimeError("FT initialization failed; see session events")
        sensor.start()
        camera = ZeroBufferCamera(args.camera_index, 640, 480)
        if not camera.start():
            raise RuntimeError("Camera initialization failed")
        controller = CollectorController(sensor, recorder,
                                         align_offset_ns=config["align_offset_ns"],
                                         max_gap_ns=config["max_pair_gap_ns"],
                                         pair_wait_ns=config["pair_wait_ns"],
                                         reference_wait_ns=config["reference_wait_ns"])
        recorder.event("collector_started", align_offset_ns=config["align_offset_ns"])
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        previous_message = None
        while True:
            sensor.check_error()
            camera.check_error()
            packet = camera.read_latest()
            controller.step(packet)
            if controller.status_text != previous_message:
                print(controller.status_text)
                previous_message = controller.status_text
            if packet is not None:
                display = packet.image.copy()
                cv2.rectangle(display, (0, 0), (display.shape[1], 82), (20, 20, 20), -1)
                state = "REC" if controller.recording else "PAUSED"
                color = (0, 0, 255) if controller.recording else (180, 180, 180)
                cv2.putText(display, f"{state} | saved {controller.saved_count} | frame {packet.seq}",
                            (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1)
                match = controller.last_match
                text = (f"Fz: {match.label[2]:.3f} N | sync: {match.status}" if match and match.valid
                        else f"FT label unavailable | sync: {match.status if match else 'waiting'}")
                cv2.putText(display, text, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,230,230), 1)
                cv2.putText(display, "T: unload then tare/reference | R: resume/pause", (12, 73),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
                cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("t"), ord("T")):
                controller.request_reference()
            elif key in (ord("r"), ord("R")):
                controller.toggle_recording()
            if packet is None:
                time.sleep(0.002)
    except KeyboardInterrupt:
        exit_reason = "KeyboardInterrupt"
    except BaseException as exc:
        exit_reason = "error"
        acquisition_error = exc
        for event_name in ("session_error", "collector_error"):
            try:
                recorder.event(event_name, error=repr(exc), stage="acquisition")
            except Exception as logging_error:
                print(f"Cannot log {event_name}: {logging_error!r}; original: {exc!r}", file=sys.stderr)
        raise
    finally:
        cleanup_errors = []
        def cleanup(stage, action):
            try:
                action()
            except Exception as exc:
                cleanup_errors.append(exc)
                try:
                    recorder.event("session_error", error=repr(exc), stage=stage)
                except Exception:
                    pass  # A failed recorder cannot be required to log its failure.
                print(f"Collector cleanup failed ({stage}): {exc!r}", file=sys.stderr)
        if controller:
            cleanup("collector_finish", lambda: controller.finish(exit_reason))
        else:
            cleanup("collector_exit", lambda: recorder.event("collector_exit", reason=exit_reason, initialized=False))
        if camera:
            cleanup("camera_release", camera.release)
        if sensor:
            cleanup("force_close", sensor.close)
        cleanup("recorder_close", recorder.close)
        cleanup("window_close", cv2.destroyAllWindows)
        print(f"采集结束: {recorder.session_dir}")
        if cleanup_errors and acquisition_error is None:
            raise cleanup_errors[0]
    return 0


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(run_collector())
