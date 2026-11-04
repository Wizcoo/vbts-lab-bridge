# Planar gripper: portable experimental Fz viewer

This folder runs four frozen CNN models through **ONNX Runtime CPU**, with the same fixed-reference float DIS frontend used for offline research. It produces **signed source Fz in N only**. It is not a six-axis SDK replacement. Physical qualification remains **NOT_READY**: the training data used ball contact, and independent weight steps, flat-contact generalization, drift and physical latency have not been qualified. No camera was opened by the development agents.

For a copyable Chinese lab-PC task covering preflight, operator-confirmed weight steps, raw evidence and handoff, use [the planar weight-step lab prompt](../../prompts/lab-planar-weight-steps.md).

Copy the complete folder, including `models/` and `vendor/`, to another location. No Torch, original research workspace, compiler, native helper DLL or network connection is required to run it after Python dependencies have been installed. Python 3.12 and the exact versions in `requirements.txt` were used for offline tests. Other dependency versions have not been certified for pixel-level frontend equivalence. Launchers do not install dependencies or change execution policy.

## Install and check, without camera access

From this folder in PowerShell (installation is a separate explicit user action):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B planar_force.py offline-check
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

`offline-check` uses synthetic image pixels to check frontend and all four graphs. Its force values are **not** physical measurements or an accuracy test. The unit tests use an injected capture stream and never call a camera. Test artifacts live under ignored `_checks/`.

## Start live viewing manually

Identify the intended camera index yourself, ensure the connected device is `HSMVBTS_G01_H20_000011`, and keep its sensing surface fully unloaded while starting. The sensor ID is your declaration; this program cannot verify a camera's physical serial number. There is no camera enumeration or automatic camera fallback.

The integration host's Windows PowerShell 5 execution policy blocked `.ps1` launchers before Python ran. Use the CMD launcher below when your policy blocks scripts; **do not change or bypass the execution policy**. CMD directly calls Python and returns its exit code. It chooses the executable in this order: explicit `PLANAR_FORCE_PYTHON`, this folder's `.venv\Scripts\python.exe`, then `python` on PATH. An invalid explicit interpreter fails rather than silently switching environments.

From Command Prompt, after creating the local `.venv` above:

```bat
start_live.cmd --camera-index 0 --sensor-id HSMVBTS_G01_H20_000011 --declare-unloaded --duration-seconds 300
```

To select an existing Python environment, set its executable path without embedding extra quotation marks in the variable:

```bat
set "PLANAR_FORCE_PYTHON=D:\Python environment\Scripts\python.exe"
start_live.cmd --help
start_live.cmd --camera-index 0 --sensor-id HSMVBTS_G01_H20_000011 --declare-unloaded --duration-seconds 300
```

The CMD launcher takes Python-style options (`--camera-index`, `--sensor-id`, `--declare-unloaded`). It can also be invoked as `.\start_live.cmd ...` from PowerShell. `--help` does not open a camera. The optional `.ps1` launcher remains available when the existing execution policy permits it, using its PowerShell-style parameters:

```powershell
.\start_live.ps1 -CameraIndex 0 -SensorId HSMVBTS_G01_H20_000011 -DeclareUnloaded -Python .\.venv\Scripts\python.exe -DurationSeconds 300
```

Equivalent direct command (also works without the PowerShell launcher):

```powershell
.\.venv\Scripts\python.exe -B planar_force.py live --camera-index 0 --sensor-id HSMVBTS_G01_H20_000011 --declare-unloaded --duration-seconds 300
```

Choose the actual index; **0 is an example, not an identified device**. The first successfully read exact-size frame becomes the fixed reference only because you explicitly declared it unloaded. A new reference requires stopping and starting a new session. Images must be BGR uint8 640×480; mismatches are recorded as errors, never silently resized for inference. The backend defaults to DirectShow; `--backend msmf` or `--backend any` changes it explicitly. The duration is a host-loop limit; a blocked device-driver read can delay shutdown and has not been hardware tested. `Q` or `Esc` stops; Ctrl+C also preserves recorded rows. Five consecutive read/geometry/quality/inference failures stop with a nonzero exit code.

The default `cnn_uv_formal_seed17` is an experimental startup default, **not a model selected or approved for deployment**. All four models remain available using `--model` (PowerShell `-Model`):

- `cnn_uv_formal_seed17`
- `cnn_xy_formal_seed17`
- `cnn_uv_formal_seed29`
- `cnn_xy_formal_seed29`

The viewer shows raw signed Fz numerically and as an unsmoothed 30-second rolling trace. The horizontal axis is host receive time, not exposure time. Invalid frames leave gaps; there is no hold-last, interpolation, filtering, force clipping, amplitude normalization or automatic zeroing. At least 20% valid grid points are required to predict; 20–90% is explicitly `DEGRADED_RAW_FZ`, while at least 90% is `VALID_RAW_FZ`. These are software quality states, not accuracy guarantees.

For an explicitly chosen display sign use `--normal-sign -1` (PowerShell `-NormalSign -1`). This affects **only `display_fz_n`**, not the raw output. Press `T` once on a valid frame to subtract that frame's raw Fz from subsequent display values. The offset, event time, source frame and effective next frame are recorded. A second `T` is rejected and logged. The plot is labeled `display_fz_n` when a sign or tare is active; raw values remain in every CSV row. No sign is inferred from load direction.

## Replay saved images, without a camera

```powershell
.\.venv\Scripts\python.exe -B planar_force.py replay --sensor-id HSMVBTS_G01_H20_000011 --declare-unloaded --reference "D:\capture\unloaded.png" --images "D:\capture\step 1.png" "D:\capture\step 2.png" --output ".\recordings\replay_001"
```

Each command uses one fixed unloaded reference. Run a separate fresh session when switching reference groups. Images are consumed in the exact order supplied. Missing/bad images remain invalid CSV rows; an invalid reference makes all supplied rows invalid. Saved-image timestamps indicate host file-read completion, **not the original acquisition time**. No GT, force labels or stage coordinates enter this command.

## Recording and traceability

Without `--output`, recordings go into a fresh `recordings/session_<time>_<random>/` directory. An explicit output directory must not already exist. UTF-8 and paths containing spaces are supported. Live original pixels are losslessly saved **before** any UI overlays. Replay references the existing originals and hashes them rather than duplicating them.

- `predictions.csv`: every processed attempt, raw signed Fz, valid/quality/error, raw image and reference paths/SHA256, host monotonic and wall-clock receive times, optional display sign/tare/output, frontend/model/read/save computation times.
- `events.jsonl`: fixed-reference declaration, camera request and explicit one-time tare or rejection.
- `session.json`: model metadata, source hashes, timestamp meaning and user assertions.
- `receipt.json`: completion/failure, row counts, source integrity, `NOT_READY`, and separate requested/opened/capture-provider fields. Injected tests are marked `SIMULATED_TEST`, never hardware access.
- `raw/`: live original PNG files. The initial reference row is `REFERENCE_CAPTURED` with no prediction; it is retained.

CSV latency fields measure software work only, not physical sensor response. The camera may discard frames upstream; the file records every attempted read/processing operation, not a guarantee that every exposure was captured. Recording PNGs can limit throughput and consume disk space. No disk/network upload is performed. `recordings/`, `_checks/`, raw image/CSV formats and training caches are Git-ignored; do not commit private recordings.

The selected ONNX bytes, manifest, frontend contract and vendored dependency hashes are checked. Graph inputs are exactly `raw_uv: float32[N,2,69,82]` in camera pixels and `valid: bool[N,1,69,82]`; output is `signed_fz_n: float32[N]`. TRAIN-only normalization, optional fixed image-grid XY channels, masking and physical Fz denormalization are inside each ONNX. Validity is not an additional output force axis.

`vendor/` files are copied unchanged from the verified offline implementation. Only the `old_dis` numerical path is called. Legacy native-helper and old byte-encoding routines remain in the vendored source for provenance, but they are neither invoked nor required by this application. `frontend_contract.json` binds all six files and the engineering ROI; it does not turn that ROI into a new physical calibration.
