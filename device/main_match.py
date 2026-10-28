# -*- coding: utf-8 -*-
"""Real-time VBTS comparison with replayable raw acquisition records.

Camera and force use the same host monotonic clock. Hardware exposure and force
sample timestamps are unavailable in the current APIs and are never invented.
No inference model, preprocessing profile or fixed latency is changed here.
"""
import argparse
import collections
import ctypes
import os
import json
import sys
from pathlib import Path
import time
from ctypes import c_int,c_uint8,c_uint64,c_float,c_double,c_char,c_char_p,c_void_p,POINTER,Structure,byref,CDLL
import numpy as np
import cv2
from sync_acquisition import ZeroBufferCamera,SixAxisForceSensor,SessionRecorder

SCRIPT_DIR=Path(__file__).resolve().parent
LOG_DIR=SCRIPT_DIR.parent/"local_data"/"captures"


def log_failure(recorder,event,exc,**details):
    """Keep the original acquisition error even if its journal is unavailable."""
    try:
        recorder.event(event,error=repr(exc),**details)
    except Exception as logging_error:
        print(f"Cannot log {event}: {logging_error!r}; original: {exc!r}",file=sys.stderr)


def cleanup_resources(recorder,actions,primary_error=None):
    errors=[]
    for stage,action in actions:
        try:
            action()
        except Exception as exc:
            errors.append(exc)
            log_failure(recorder,"session_error",exc,stage=stage)
    if errors and primary_error is None:
        raise errors[0]

class HSMVBTS_Config(Structure):
    _fields_ = [
        ("sensor_serial", c_char_p),
        ("config_dir", c_char_p),
        ("profile_override_path", c_char_p),
        ("use_cuda", c_int)
    ]

class HSMVBTS_FrameResult(Structure):
    _fields_ = [
        ("frame_id", c_uint64),
        ("timestamp_ms", c_double),

        # 1. 六维力 / 力矩
        ("force_valid", c_int),
        ("force_ready", c_int),
        ("force_status", c_char * 32),
        ("force", c_float * 6),        # [Fx, Fy, Fz, Mx, My, Mz]
        ("raw_force", c_float * 6),

        # 2. 深度
        ("depth_valid", c_int),
        ("max_depth_mm", c_float),
        ("equivalent_depth_mm", c_float),
        ("depth_width", c_int),
        ("depth_height", c_int),
        ("depth_map", POINTER(c_float)),

        # 3. 接触与形貌
        ("contact_detected", c_int),
        ("contact_center_x", c_float),
        ("contact_center_y", c_float),
        ("contact_mask_width", c_int),
        ("contact_mask_height", c_int),
        ("contact_mask", POINTER(c_uint8)),
        ("morphology_width", c_int),
        ("morphology_height", c_int),
        ("morphology", POINTER(c_float)),

        # 4. 三向位移与网格特征
        ("point_count", c_int),
        ("ref_x", POINTER(c_float)),
        ("ref_y", POINTER(c_float)),
        ("deformed_x", POINTER(c_float)),
        ("deformed_y", POINTER(c_float)),
        ("disp_dx", POINTER(c_float)),
        ("disp_dy", POINTER(c_float)),
        ("disp_dz", POINTER(c_float)),
        ("peak_dx", c_float),
        ("peak_dy", c_float),

        # 5. 滑移与耗时
        ("slip_x", c_float),
        ("slip_y", c_float),
        ("tracking_ms", c_double),
        ("force_ms", c_double),
        ("depth_ms", c_double),
        ("total_ms", c_double),
    ]

class HsmvbtsPipeline:
    def __init__(self, sdk_dll_path, config_dir, model_txt_path, sensor_serial="HSMVBTS_G01_H20_000011"):
        self.sdk_dll_path = sdk_dll_path
        self.config_dir = config_dir
        self.model_txt_path = model_txt_path
        self.sensor_serial = sensor_serial
        self.handle = c_void_p(None)

        if not os.path.exists(sdk_dll_path):
            raise FileNotFoundError(f"SDK DLL not found: {sdk_dll_path}")

        self._dll_directory = os.add_dll_directory(os.path.dirname(sdk_dll_path)) if os.name == "nt" else None
        self.sdk = CDLL(sdk_dll_path)
        self._setup_prototypes()

    def _setup_prototypes(self):
        self.sdk.hsmvbts_create.restype = c_int
        self.sdk.hsmvbts_create.argtypes = [POINTER(HSMVBTS_Config), POINTER(c_void_p)]

        self.sdk.hsmvbts_destroy.restype = None
        self.sdk.hsmvbts_destroy.argtypes = [c_void_p]

        self.sdk.hsmvbts_reset_reference.restype = c_int
        self.sdk.hsmvbts_reset_reference.argtypes = [c_void_p]

        self.sdk.hsmvbts_process_image.restype = c_int
        self.sdk.hsmvbts_process_image.argtypes = [
            c_void_p, POINTER(c_uint8), c_int, c_int, c_int, POINTER(HSMVBTS_FrameResult)
        ]

        self.sdk.hsmvbts_get_last_error.restype = c_char_p

    def initialize(self):
        cfg = HSMVBTS_Config()
        cfg.sensor_serial = self.sensor_serial.encode("utf-8")
        cfg.config_dir = self.config_dir.encode("utf-8")
        cfg.profile_override_path = self.model_txt_path.encode("utf-8")
        cfg.use_cuda = 0

        model_dir = os.path.dirname(os.path.abspath(self.model_txt_path))
        prev_cwd = os.getcwd()
        try:
            os.chdir(model_dir)
            res = self.sdk.hsmvbts_create(byref(cfg), byref(self.handle))
        finally:
            os.chdir(prev_cwd)

        if res != 0 or not self.handle:
            raise RuntimeError(f"hsmvbts_create failed with code {res}")
        print(f"[HSMVBTS] Initialized pipeline successfully for {self.sensor_serial}.")

    def reset_reference(self):
        if not self.handle:
            return -1
        return int(self.sdk.hsmvbts_reset_reference(self.handle))

    def process_frame(self, frame_bgr):
        if not self.handle:
            return None
        h, w, c = frame_bgr.shape
        stride = w * c
        bgr_ptr = frame_bgr.ctypes.data_as(POINTER(c_uint8))
        result = HSMVBTS_FrameResult()
        st = self.sdk.hsmvbts_process_image(self.handle, bgr_ptr, w, h, stride, byref(result))
        if st != 0:
            return None
        return result

    def close(self):
        if self.handle:
            self.sdk.hsmvbts_destroy(self.handle)
            self.handle = c_void_p(None)
            print("[HSMVBTS] Pipeline destroyed.")


class ValidatorController:
    """Hardware-independent pairing/reference state machine for the UI."""
    def __init__(self,pipeline,sensor,recorder,alignment_offset_ms=0.,max_gap_ms=30.,pair_wait_ms=150.,*,clock_ns=time.perf_counter_ns,reference_wait_ms=3000.):
        if not all(np.isfinite(v) for v in (alignment_offset_ms,max_gap_ms,pair_wait_ms)) or max_gap_ms<=0 or pair_wait_ms<0:
            raise ValueError("Invalid timing policy")
        self.pipeline=pipeline;self.sensor=sensor;self.recorder=recorder
        self.offset_ns=round(alignment_offset_ms*1e6);self.max_gap_ns=round(max_gap_ms*1e6)
        self.wait_ns=round(pair_wait_ms*1e6)
        if not np.isfinite(reference_wait_ms) or reference_wait_ms<=0:raise ValueError("Invalid reference timeout")
        self.clock_ns=clock_ns;self.reference_wait_ns=round(reference_wait_ms*1e6);self.reference_deadline_ns=None
        self.pending=collections.deque();self.last_seq=0
        self.reference_id=None;self.reference_pending=False;self.reference_after_ns=0
        self.fatal_error=None
        self.records=0
        self.recorder.event("alignment_policy",offset_ns=self.offset_ns,max_gap_ns=self.max_gap_ns,
                            pair_wait_ns=self.wait_ns,target_definition="frame.grab_end_ns - offset_ns",
                            hardware_delay_calibrated=False)

    def request_tare(self):
        self.check_active()
        self.recorder.event("comparison_paused_for_tare",discarded_pending_seqs=[p.seq for p,_ in self.pending],
                            previous_reference_id=self.reference_id)
        self.pending.clear();self.reference_id=None;self.reference_pending=False
        if not self.sensor.zero(reason="main_match_user_tare"):
            self.recorder.event("reference_not_created",reason="tare_failed")
            return False
        self.reference_after_ns=self.sensor.last_tare_end_ns
        self.reference_deadline_ns=self.clock_ns()+self.reference_wait_ns
        self.reference_pending=True
        return True

    def observe(self,packet,now_ns=None):
        self.check_active()
        if packet.seq==self.last_seq:return False
        if packet.seq<self.last_seq:raise RuntimeError("Camera sequence restarted; begin a new acquisition session")
        if self.last_seq and packet.seq>self.last_seq+1:
            self.recorder.event("camera_consumer_sequence_gap",previous_seq=self.last_seq,current_seq=packet.seq,
                                omitted_successful_grabs=packet.seq-self.last_seq-1)
        self.last_seq=packet.seq
        self.pending.append((packet,self.clock_ns() if now_ns is None else now_ns))
        return True

    def drain(self,now_ns=None,force=False):
        self.check_active()
        now_ns=self.clock_ns() if now_ns is None else now_ns
        if self.reference_pending and now_ns>=self.reference_deadline_ns:
            self.reference_pending=False
            self.recorder.event("reference_failed",reason="fresh_frame_or_force_timeout",tare_epoch=self.sensor.tare_epoch)
        results=[]
        while self.pending:
            packet,enqueued=self.pending[0]
            target=packet.grab_end_ns-self.offset_ns
            match=self.sensor.match(target,max_gap_ns=self.max_gap_ns,expected_tare_epoch=self.sensor.tare_epoch)
            if not match.valid and not force and now_ns-enqueued<self.wait_ns and len(self.pending)<128:
                break
            self.pending.popleft()
            prediction={"valid":False,"ready":False,"status":"REFERENCE_REQUIRED","force":None,"raw_force":None}
            latency_ms=0.
            reference_frame=False
            if self.reference_pending:
                prediction["status"]="REFERENCE_PENDING"
                if packet.grab_start_ns>self.reference_after_ns and match.valid:
                    ref=self.recorder.save_reference(packet,match,reason="operator_unloaded_after_successful_tare")
                    self.recorder.event("sdk_reference_reset_requested",reference_id=ref,frame_seq=packet.seq)
                    code=self.pipeline.reset_reference()
                    self.recorder.event("sdk_reference_reset_result",reference_id=ref,return_code=code)
                    if code!=0:self.fail(f"SDK reference reset failed ({code}); no comparison accepted")
                    self.reference_id=ref;self.reference_pending=False;reference_frame=True
            if self.reference_id is not None:
                start_ns=time.perf_counter_ns();res=self.pipeline.process_frame(packet.image);end_ns=time.perf_counter_ns()
                latency_ms=(end_ns-start_ns)/1e6
                if res is None:
                    prediction={"valid":False,"ready":False,"status":"SDK_PROCESS_FAILED","force":None,"raw_force":None}
                else:
                    valid=bool(res.force_valid) and bool(res.force_ready)
                    values=[float(v) for v in res.force]
                    raw=[float(v) for v in res.raw_force]
                    valid=valid and bool(np.isfinite(values).all())
                    status=res.force_status.decode("utf-8",errors="replace") if isinstance(res.force_status,bytes) else str(res.force_status)
                    prediction={"valid":valid,"ready":bool(res.force_ready),"status":status,"force":values,
                                "raw_force":raw,"sdk_frame_id":int(res.frame_id),"sdk_timestamp_ms":float(res.timestamp_ms),
                                "sdk_timestamp_semantics":"SDK-owned; not assumed camera exposure or host clock",
                                "contact_detected":bool(res.contact_detected)}
                prediction.update(process_start_ns=start_ns,process_end_ns=end_ns)
                if reference_frame:
                    self.recorder.event("sdk_reference_frame_processed",reference_id=self.reference_id,frame_seq=packet.seq,
                                        sdk_result_returned=res is not None)
                    if res is None:self.fail("SDK failed on the reference image; stop instead of claiming a known reference")
            usable=match.valid and prediction["valid"] and self.reference_id is not None
            extra={"alignment_offset_ns":self.offset_ns,"reference_frame":reference_frame,"usable_for_model_comparison":usable,
                   "usable_for_training":match.valid and self.reference_id is not None,
                   "pipeline_latency_ms":latency_ms,"pair_wait_elapsed_ns":now_ns-enqueued,
                   "reference_state":"pending" if self.reference_pending else ("ready" if self.reference_id else "required")}
            row=self.recorder.record_pair(packet,match,reference_id=self.reference_id,prediction=prediction,extra=extra)
            self.records+=1
            results.append({"row":row,"match":match,"prediction":prediction,"latency_ms":latency_ms,"packet":packet})
        return results

    def check_active(self):
        if self.fatal_error is not None:
            raise RuntimeError(self.fatal_error)

    def fail(self,message):
        self.fatal_error=message
        self.reference_id=None;self.reference_pending=False
        self.recorder.event("comparison_controller_failed",reason=message,
                            discarded_pending_seqs=[p.seq for p,_ in self.pending])
        self.pending.clear()
        raise RuntimeError(message)


class SensorValidatorUI:
    def __init__(self,pipeline,force_sensor,recorder,camera_index=0,alignment_offset_ms=0.,max_gap_ms=30.,pair_wait_ms=150.):
        self.pipeline=pipeline;self.force_sensor=force_sensor;self.recorder=recorder;self.camera_index=camera_index
        self.controller=ValidatorController(pipeline,force_sensor,recorder,alignment_offset_ms,max_gap_ms,pair_wait_ms)
        self.window_size=500
        self.model_hist=[collections.deque(maxlen=self.window_size) for _ in range(6)]
        self.sensor_hist=[collections.deque(maxlen=self.window_size) for _ in range(6)]
        self.sync_status="Press T while unloaded"

    def draw_subplot(self, roi, model_line, sensor_line, title):
        roi[:] = (30, 30, 30)
        h, w = roi.shape[:2]
        pad_l, pad_r, pad_t, pad_b = 40, 8, 22, 18
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        if plot_w <= 1 or plot_h <= 1:
            return

        vmin, vmax = 0.0, 0.0
        all_vals = [v for v in list(model_line) + list(sensor_line) if np.isfinite(v)]
        if all_vals:
            vmin = min(all_vals)
            vmax = max(all_vals)

        if vmax - vmin < 1e-5:
            vmax = vmin + 1.0
        pad = (vmax - vmin) * 0.1
        vmin -= pad
        vmax += pad

        cv2.rectangle(roi, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (80, 80, 80), 1)
        if vmin <= 0.0 <= vmax:
            zero_y = int(pad_t + ((vmax - 0.0) / (vmax - vmin)) * plot_h)
            cv2.line(roi, (pad_l, zero_y), (pad_l + plot_w, zero_y), (60, 60, 60), 1)

        def draw_curve(data_deque, color):
            if len(data_deque) < 2:
                return
            pts = []
            N = self.window_size
            base = (N - len(data_deque)) if len(data_deque) < N else 0
            for i, val in enumerate(data_deque):
                x = int(pad_l + ((base + i) / float(N - 1)) * plot_w)
                if not np.isfinite(val):
                    pts.append(None)
                    continue
                y = int(pad_t + ((vmax - val) / (vmax - vmin)) * plot_h)
                pts.append((x, y))
            for i in range(len(pts) - 1):
                if pts[i] is not None and pts[i + 1] is not None:
                    cv2.line(roi, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

        draw_curve(model_line, (255, 160, 0))
        draw_curve(sensor_line, (0, 0, 255))

        cv2.putText(roi, title, (pad_l, pad_t - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(roi, f"{vmax:.2f}", (2, pad_t + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(roi, f"{vmin:.2f}", (2, pad_t + plot_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        last_m = model_line[-1] if model_line else float('nan')
        last_s = sensor_line[-1] if sensor_line else float('nan')
        info_txt = f"Model={last_m:.3f} FT={last_s:.3f}"
        cv2.putText(roi, info_txt, (pad_l, pad_t + plot_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_ui(self, canvas, fps, latency_ms, frame_idx):
        header_h = 36
        canvas[:header_h, :] = (22, 22, 22)
        status_txt = (f"Camera {fps:.1f} FPS | Pipe {latency_ms:.1f} ms | "
                      f"Offset {self.controller.offset_ns/1e6:+.1f} ms | "
                      f"{self.sync_status} | [T] unloaded tare/reference | [ESC] exit")
        cv2.putText(canvas, status_txt, (16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

        titles = ["Fx (N)", "Fy (N)", "Fz (N)", "Mx (Nm)", "My (Nm)", "Mz (Nm)"]
        rows, cols = 2, 3
        sub_w = canvas.shape[1] // cols
        sub_h = (canvas.shape[0] - header_h) // rows

        for i in range(6):
            r = i // cols
            c = i % cols
            y0 = header_h + r * sub_h
            x0 = c * sub_w
            sub = canvas[y0:y0 + sub_h, x0:x0 + sub_w]
            self.draw_subplot(sub, self.model_hist[i], self.sensor_hist[i], titles[i])

    def consume_results(self,results):
        for item in results:
            pred=item["prediction"];match=item["match"]
            model=np.asarray(pred["force"],float) if pred["valid"] else np.full(6,np.nan)
            force=np.asarray(match.label,float) if match.valid else np.full(6,np.nan)
            for k in range(6):self.model_hist[k].append(model[k]);self.sensor_hist[k].append(force[k])
            self.sync_status=f"{match.status}/{pred['status']}"

    def run(self):
        camera=ZeroBufferCamera(self.camera_index,640,480)
        completed=False
        acquisition_error=None
        try:
            if not camera.start():raise RuntimeError(f"Cannot open camera {self.camera_index}")
            self.force_sensor.start()
            win_name="VBTS comparison | T: unloaded tare/reference | ESC: exit"
            cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
            canvas=np.zeros((720,1280,3),np.uint8)
            fps=0.;previous_ns=None;latest_image=None;latency_ms=0.
            print(f"Replayable session: {self.recorder.session_dir}")
            print("请先保持无接触，按 T 置零并建立新参考帧。无效匹配显示缺失，不补零。")
            while True:
                camera.check_error();self.force_sensor.check_error()
                packet=camera.read_latest()
                if packet is not None and self.controller.observe(packet):
                    latest_image=packet.image
                    if previous_ns is not None and packet.grab_end_ns>previous_ns:
                        instant=1e9/(packet.grab_end_ns-previous_ns);fps=.9*fps+.1*instant if fps else instant
                    previous_ns=packet.grab_end_ns
                results=self.controller.drain();self.consume_results(results)
                if results:latency_ms=results[-1]["latency_ms"]
                if self.controller.reference_pending:
                    self.sync_status="Waiting for fresh tare/reference frame"
                elif self.controller.reference_id is None:
                    self.sync_status="Reference required; unloaded T"
                self.draw_ui(canvas,fps,latency_ms,self.controller.records)
                if latest_image is not None:
                    canvas[40:220,-250:-10]=cv2.resize(latest_image,(240,180))
                cv2.imshow(win_name,canvas)
                key=cv2.waitKey(1)&0xff
                if key==27:break
                if key in (ord("t"),ord("T")):
                    ready=self.controller.request_tare()
                    self.sync_status="Reference pending" if ready else "Tare failed; no valid reference"
                    for h in self.model_hist+self.sensor_hist:h.clear()
                if packet is None:time.sleep(.002)
            completed=True
        except BaseException as exc:
            acquisition_error=exc
            discarded=[p.seq for p,_ in self.controller.pending]
            self.controller.pending.clear()
            log_failure(self.recorder,"comparison_aborted",exc,discarded_pending_seqs=discarded)
            raise
        finally:
            def finish_pairs():
                if completed:
                    self.consume_results(self.controller.drain(force=True))
                    self.recorder.event("comparison_stopped",pairs=self.controller.records)
            cleanup_resources(self.recorder,[("camera_release",camera.release),
                              ("comparison_finish",finish_pairs),("window_close",cv2.destroyAllWindows)],acquisition_error)


def default_binary(name):
    candidates=[SCRIPT_DIR/"bin"/name,SCRIPT_DIR.parents[2]/"common/bin"/name]
    return next((p for p in candidates if p.is_file()),candidates[0])

def read_force_profile(path):
    fields={}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line=line.split("#",1)[0].strip()
        if "=" in line:
            key,value=line.split("=",1);fields[key.strip()]=value.strip()
    signs=tuple(float(v) for v in fields["FORCE_AXIS_SIGNS"].split(","))
    if len(signs)!=6 or not all(v in (-1.,1.) for v in signs):raise ValueError("Invalid FORCE_AXIS_SIGNS")
    arm=float(fields["FORCE_MOMENT_ARM"])
    if not np.isfinite(arm):raise ValueError("Invalid FORCE_MOMENT_ARM")
    return signs,arm,fields

def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdk-dll",type=Path,default=default_binary("hsmvbts_sdk.dll"))
    ap.add_argument("--force-dll",type=Path,default=default_binary("hps_ft.dll"))
    ap.add_argument("--profile",type=Path,required=True)
    ap.add_argument("--config-dir",type=Path)
    ap.add_argument("--output-dir",type=Path,default=LOG_DIR)
    ap.add_argument("--camera-index",type=int,default=0)
    ap.add_argument("--baud-rate",type=int,default=115200)
    ap.add_argument("--port",default="")
    ap.add_argument("--alignment-offset-ms","--align-offset-ms",type=float,default=0.,help="Explicit diagnostic offset; target=grab_end-offset. Default 0 is uncalibrated host-time pairing, never a measured hardware delay.")
    ap.add_argument("--max-gap-ms","--max-pair-gap-ms",type=float,default=30.)
    ap.add_argument("--pair-wait-ms",type=float,default=150.)
    ap.add_argument("--check-config",action="store_true",help="Inspect paths and policy without opening hardware")
    args=ap.parse_args(argv)
    if not all(np.isfinite(v) for v in (args.alignment_offset_ms,args.max_gap_ms,args.pair_wait_ms)) or args.max_gap_ms<=0 or args.pair_wait_ms<=0:
        ap.error("Timing arguments must be finite, with positive max gap and pair wait")
    return args

def main(argv=None):
    args=parse_args(argv)
    signs,arm,profile=read_force_profile(args.profile)
    from inventory_device import inventory
    if args.check_config:
        print(json.dumps({"profile":str(args.profile),"sdk_dll":str(args.sdk_dll),"sdk_dll_exists":args.sdk_dll.is_file(),
                          "force_dll":str(args.force_dll),"force_dll_exists":args.force_dll.is_file(),
                          "alignment_offset_ms":args.alignment_offset_ms,"max_gap_ms":args.max_gap_ms,
                          "force_raw_average_count":1,"axis_signs":signs,"moment_arm_m":arm,
                          "profile_validation_latency_ms_not_applied":profile.get("VALIDATION_LATENCY_MS"),
                          "hardware_opened":False},ensure_ascii=False,indent=2))
        return 0
    for path in (args.sdk_dll,args.force_dll,args.profile):
        if not path.is_file():raise FileNotFoundError(f"Required file is missing: {path}. Supply its explicit command-line path; no model will be copied.")
    recorder=SessionRecorder(args.output_dir,profile_path=args.profile,metadata={"tool":"main_match","alignment_offset_ms":args.alignment_offset_ms,
        "force_average_count_applied":1,"profile_force_average_count":profile.get("FORCE_AVERAGE_COUNT"),
        "profile_validation_latency_ms_not_automatically_applied":profile.get("VALIDATION_LATENCY_MS"),
        "axis_signs":signs,"moment_arm_m":arm,"camera_index":args.camera_index,"force_dll":str(args.force_dll),"sdk_dll":str(args.sdk_dll),
        "bridge_inventory":inventory(args.profile,args.force_dll,sdk_dll=args.sdk_dll)})
    pipeline=None;sensor=None;acquisition_error=None
    try:
        pipeline=HsmvbtsPipeline(str(args.sdk_dll.resolve()),str((args.config_dir or args.profile.parent).resolve()),str(args.profile.resolve()),profile.get("PROFILE_ID","HSMVBTS_G01_H20_000011"))
        pipeline.initialize()
        sensor=SixAxisForceSensor(str(args.force_dll.resolve()),baud_rate=args.baud_rate,port=args.port.encode(),
                                  axis_signs=signs,moment_arm=arm,recorder=recorder,average_count=1)
        if not sensor.open():raise RuntimeError("Force sensor unavailable; no synthetic zero ground truth will be logged")
        SensorValidatorUI(pipeline,sensor,recorder,args.camera_index,args.alignment_offset_ms,args.max_gap_ms,args.pair_wait_ms).run()
    except BaseException as exc:
        acquisition_error=exc
        log_failure(recorder,"session_error",exc,stage="main_match")
        raise
    finally:
        actions=[]
        if sensor is not None:actions.append(("force_close",sensor.close))
        if pipeline is not None:actions.append(("pipeline_close",pipeline.close))
        actions.append(("recorder_close",recorder.close))
        cleanup_resources(recorder,actions,acquisition_error)

if __name__=="__main__":main()
