# FaMa low-mass calibration demo — NOT_READY

**CALIBRATION FIT ONLY / target 0–100 g.** This is an experimental fit to the currently observed FaMa data, including near-zero and 10/20/50/100 g records. Those observations have been reused for adaptation; they are **not unseen validation data**. A successful fitted staircase does not establish accuracy at a new contact position, on another sensor, or in a new run.

Two real fixed candidates are included. Both completed the new 1,501-row adaptation-replay ONNX parity checks (3,002 outputs total, zero failed comparisons; maximum absolute difference approximately 3.815e-6 N). This establishes software export consistency only, not physical accuracy or generalization:

- `lowmass_head60`: pretrained UV model with only the regression head fitted for 60 epochs. This is the preregistered default, not a winner selected from fitted curves.
- `lowmass_aug60`: UV model fitted with small joint spatial translations of UV and mask. This numerical augmentation does **not** prove physical contact-position invariance.

Output is raw signed source **Fz in N only**. There are no fabricated other force/torque axes, and these models do not replace an SDK model. No inference clipping, load-bin snapping, smoothing, automatic tare or adaptive reference is applied.

Current reused calibration observations still contain a low-load offset: candidate A has approximately -0.0514 N near-zero output and -0.0901 N at the recorded 10 g plateau, leaving only about -0.0387 N difference (recorded-label difference about -0.0954 N). Candidate B is weaker at this load: about -0.0193 N near zero and -0.0543 N at 10 g. An explicit display tare can therefore leave a smaller apparent 10 g increment. These are observations on the reused data, **not a demonstrated 10 g resolution or independent accuracy result**. Do not compensate by snapping, clipping, increasing gain, selecting only favorable contact locations, or repeatedly re-zeroing during evaluation.

For Chinese instructions see `QUICKSTART_ZH.md`; the copyable independent multi-position collection task is `LAB_CAPTURE_PROMPT_ZH.md`.

## Device identity remains unresolved

The FaMa capture snapshot identifies `HSMVBTS_G01_H20_000015`; the inherited frontend/model profile identifies `HSMVBTS_G01_H20_000011`. A same-batch statement does not resolve that identity mismatch. The runtime requires an explicit `--sensor-id` matching either of these two declarations, records which one you supplied, and **always records `identity_status=UNRESOLVED`**, together with both IDs. This is not automatic hardware serial verification or a claim that either candidate generalizes across sensors.

The inherited frontend files and numerical contract are copied unchanged. In particular, the contract's `000011` identity/engineering ROI is retained as provenance rather than renamed to pretend a new calibration. Only the new demo entrypoint accepts the two explicit device declarations; the actual hardware identity remains unresolved.

## Install and test without hardware

Use Python 3.12 with the versions in `requirements.txt`; Torch and the original research workspace are not runtime dependencies. Installation is an explicit user action. Launchers never install software or change PowerShell execution policy.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B planar_force.py offline-check
```

`offline-check` requires the actual new two-model registry and checks synthetic image inputs only. Missing models or incomplete new parity evidence are rejected. The new schema is `planar_lowmass_demo_candidate_v1`; the original four-model / 42,465-row parity receipt is never reused to qualify the new fit. Actual new model parity covers the 1,501 adaptation-source replay rows, which is software numerical consistency, not independent accuracy validation.

## Manual live demo

Identify the actual camera index and keep the sensor completely unloaded while starting. No camera is opened by the development agents. In Command Prompt, using this folder's `.venv`:

```bat
start_live.cmd --camera-index 0 --sensor-id HSMVBTS_G01_H20_000015 --declare-unloaded --duration-seconds 300
```

**0 is an example index, not an identified camera.** If you explicitly choose the user-declared inherited identity, supply `--sensor-id HSMVBTS_G01_H20_000011`; this does not resolve the discrepancy. Select the other candidate with `--model lowmass_aug60`.

CMD directly calls Python, choosing `PLANAR_FORCE_PYTHON` first, then local `.venv\Scripts\python.exe`, then PATH `python`. To use an existing environment:

```bat
set "PLANAR_FORCE_PYTHON=D:\Python environment\Scripts\python.exe"
start_live.cmd --help
```

The integration host's PowerShell 5 policy blocks `.ps1` execution. Use CMD without changing or bypassing that policy. `start_live.ps1` remains optional on a host whose existing policy permits it:

```powershell
.\start_live.ps1 -CameraIndex 0 -SensorId HSMVBTS_G01_H20_000015 -DeclareUnloaded -Python .\.venv\Scripts\python.exe -Model lowmass_head60 -DurationSeconds 300
```

The first successfully read exact-size frame is used as a fixed reference **only after your explicit unloaded declaration**. It is never replaced during the session. Input must be BGR uint8 640×480. DirectShow is the default; use `--backend msmf` or `--backend any` explicitly if needed. The host loop defaults to 300 seconds (maximum 3,600); a blocked driver read may delay stopping. `Q`/Esc stops. Five consecutive invalid attempts stop and retain the errors.

## What the display and warning mean

The screen prominently states **CALIBRATION FIT ONLY / NOT_READY / target 0–100 g** and the unresolved identity. It displays raw signed Fz numerically and as a 30-second unsmoothed trace, with host receive time on the horizontal axis. Invalid samples leave gaps, with no hold-last or silently removed attempts.

Raw output outside the conservative interval **[-1.15, +0.15] N** triggers `OUTSIDE_EXPECTED_DEMO_OUTPUT_RANGE` in the UI and CSV. Values are **not clamped, snapped or replaced**. The original value is still plotted and logged. This is only an output-value warning: **an output inside that interval does not prove the input is in scope, the applied load is below 100 g, or the prediction is accurate**. The code does not infer true mass from this warning. Quality remains separately recorded (`VALID_RAW_FZ` at ≥90% valid points, `DEGRADED_RAW_FZ` at 20–90%, invalid below 20%).

Optional `--normal-sign -1` changes only `display_fz_n`. Press `T` once on a valid frame for an explicit display tare; its raw offset, time and source attempt are recorded, and subsequent rows retain both raw and display values. A second tare is rejected and logged. The range warning always checks **raw output**, so a display tare cannot hide it. There is no automatic zero fitting or reference update.

## Saved-image replay and recordings

```powershell
.\.venv\Scripts\python.exe -B planar_force.py replay --sensor-id HSMVBTS_G01_H20_000015 --declare-unloaded --reference "D:\capture\unloaded.png" --images "D:\capture\load 1.png" "D:\capture\load 2.png" --output ".\recordings\replay_001"
```

One command uses one fixed reference. Run a fresh session for a different reference group. Missing/bad images remain invalid rows. Supplied image order is preserved. Replay timestamps are host file-read completion, **not original exposure/acquisition time**; no GT or stage data enter inference.

Every session creates a fresh output directory (`recordings/session_*` by default); existing directories are refused. Live original pixels are saved before overlays. `predictions.csv` preserves every processed attempt, raw signed Fz, display output/sign/tare, quality, range warning, declared ID plus both conflicting IDs, original image/reference paths and hashes, and read/frontend/model/save timings. `events.jsonl`, `session.json`, and `receipt.json` retain provenance and completion/failure. Simulated captures are explicitly marked and never reported as hardware access. Timing is software computation/host reception, not physical response latency.

`recordings/`, `_checks/`, raw images, CSVs and training caches are ignored by Git. Keep the entire runtime folder for local portability, including its new `models/` and unchanged `vendor/` files. These newly fitted weights and recordings are **not authorized for public upload**. Existing runtime/SDK/active models remain untouched.

Before any accuracy claim, collect a new independent unloaded→10→20→50→100 g sequence with repeated loads at changed contact positions, preserve raw images and synchronized force truth, and compare against the unchanged original model. Current fit and shared-reference cycle diagnostics do not substitute for that measurement.

## Repository delivery

The user authorized publication of these two candidates and runtime for lab testing. Raw acquisition images, per-frame force labels, training caches and checkpoints are excluded. See [MODEL_CARD.md](MODEL_CARD.md) for aggregate fit/diagnostic evidence and [the lab checkout task](../../prompts/lab-lowmass-test.md) for the PR branch.
