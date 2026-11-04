# FaMa 低载演示：快速启动

**只用于当前设备数据的校准拟合演示，NOT_READY，目标范围 0–100 g。** 不是独立精度验收，也未证明更换接触位置后仍准确。程序只输出有符号 Fz（N），不替换原 SDK 六维模型。

两个真实 ONNX 已包含，分别通过 1,501 帧适配源回放的数值一致性检查（合计 3,002 次比较，无失败，最大差约 3.815e-6 N）。这只证明导出软件一致性，不证明现场精度。

当前 A 在已复用的适配数据上，近零输出约 −0.0514 N，10 g 平台约 −0.0901 N；一次去皮后，两者增量可能只有约 −0.0387 N，而记录标签差约 −0.0954 N。B 的 10 g 更弱：近零约 −0.0193 N、10 g 约 −0.0543 N。不要把拟合曲线或有台阶当成已证明 10 g 分辨力，也不要靠**每档重新 tare**、增益或档位吸附修饰这个现象。请用新落点、新重复数据验证。

## 1. 人工核对设备、相机和无载状态

采集快照写的是 `HSMVBTS_G01_H20_000015`，继承前端/模型写的是 `HSMVBTS_G01_H20_000011`。请核对实物标签、实际相机索引和当前采集配置；记录证据。程序不读取硬件序列号来解决此冲突。即便你明确选择其中一个 ID，记录仍保留双方来源，状态始终是 `UNRESOLVED`。

启动前让触面完全无载。首次成功读取的 640×480 BGR 原图会在人工 `--declare-unloaded` 声明下成为固定参考；运行中不自动更新参考。相机索引 `0` 仅是下方示例，不能据此判断现场设备。

## 2. 无相机的软件检查

使用完整 runtime 文件夹（包含 `models/`、`vendor/`），Python 3.12。以下安装操作由用户明确执行；启动器不会自动安装或修改系统策略：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B planar_force.py offline-check
```

检查会加载两个真实 ONNX，用合成像素测试接口；其输出不是物理测量。模型/清单缺失、哈希不符或适配模型验收未完成时会拒绝运行。

## 3. 用户手动启动

在 runtime 文件夹打开 CMD。先按实物与采集配置人工确认 ID；下面展示**采集快照 ID**的用法：

```bat
start_live.cmd --camera-index 0 --sensor-id HSMVBTS_G01_H20_000015 --declare-unloaded --duration-seconds 300
```

如果你明确采用用户声明的继承 ID，改为 `--sensor-id HSMVBTS_G01_H20_000011`；程序仍保留冲突，不视为已完成硬件身份确认。

默认 `lowmass_head60` 是事先规定的 A 方案，不是按拟合曲线优选的“最佳模型”。要显式运行 B：

```bat
start_live.cmd --camera-index 0 --sensor-id HSMVBTS_G01_H20_000015 --declare-unloaded --model lowmass_aug60 --duration-seconds 300
```

解释器依次使用 `PLANAR_FORCE_PYTHON`、本目录 `.venv\Scripts\python.exe`、PATH 中的 `python`。使用已有环境时：

```bat
set "PLANAR_FORCE_PYTHON=D:\你的Python环境\Scripts\python.exe"
start_live.cmd --help
```

现场 Windows PowerShell 5 曾在进入 Python 前阻止 `.ps1`。请直接用上述 CMD 启动法，**不用修改或绕过执行策略**。本次开发和检查没有开启相机。

## 4. 看什么，保留什么

- 标题必须显示 `CALIBRATION FIT ONLY / NOT_READY / target 0-100g`，并显示双 ID 冲突。
- 默认显示与记录原始有符号 Fz；不取绝对值、不平滑、不截断、不吸附到 10/20/50/100 g 档位。
- 原始输出超出 `[-1.15,+0.15] N` 时提示 `OUTSIDE_EXPECTED_DEMO_OUTPUT_RANGE`，但原值继续记录和绘图。范围内**不代表**输入在分布内、真实负载低于 100 g 或预测可信。
- `T` 仅允许一次显式显示去皮，偏置/时间/来源帧会保存；原始值不变。第二次 `T` 被拒绝。原始输出范围报警不受去皮影响。
- `--normal-sign -1` 只改变显示方向，不改变原始列。不要靠符号拟合使曲线看起来正确。
- 无效帧记录空预测和原因，曲线留断点，不沿用上一帧。连续 5 次无效后停止并返回非零码。
- `Q` / Esc 停止；默认 300 秒。驱动阻塞可能延迟停止，这不是已验证的硬件实时性保证。

每次生成全新的 `recordings/session_*`；`--output` 指定的目录不得已存在。原图在叠加 UI 之前保存。保留 `raw/`、`predictions.csv`、`events.jsonl`、`session.json`、`receipt.json` 整个目录。时间戳为主机接收时间，不是相机曝光时间。不要只留截图或挑选好看的片段。

用户已授权发布本 PR 的两个候选 ONNX 和运行代码；新录制原图、拟合数据及后续新模型不因此获准公开上传。旧 runtime、原 SDK 和原始权重均保持不变。下一步用 `LAB_CAPTURE_PROMPT_ZH.md` 采集独立多落点数据。
