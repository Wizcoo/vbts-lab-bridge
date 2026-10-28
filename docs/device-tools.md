# Windows 设备端采集与导出

## 范围、研究与选择

2026-10-28 核查现有 `collect_sync_dataset.py`、`sync_acquisition.py`、`audit_sync_session.py`、`main_match.py` 与两套离线测试。源 SHA256 见 device/source_provenance.json；原工程只读，未修改现役模型、DLL 或数据。

方案 A：重新写采集器与标签同步，工作量大，容易丢失已测试的 tare/参考/失败语义。方案 B：复制成熟采集核心到独立目录，显式配置路径，新增不删行的可移植导出。选择 B。有限计划：复制与依赖隔离 -> 离线假设备测试 -> 导出正负测试 -> PR/CI -> 实验室检查真实驱动与 ABI -> 人员确认后采集。当前没有执行真实硬件采集。

## 提供的入口

| 脚本 | 用途 | 是否打开硬件 |
|---|---|---|
| inventory_device.py | 指定文件的哈希、环境、主机时钟锚点 | 否 |
| collect_sync_dataset.py --check-config | 检查路径/符号/配对设置 | 否 |
| collect_sync_dataset.py | OpenCV DirectShow 相机 + HPS FT 原始采集 | 是 |
| main_match.py --check-config | 可选 SDK 比较配置检查 | 否 |
| main_match.py | 按现有 ABI 调用现场 HSM SDK 并保存比较记录 | 是，先核现场头文件/ABI |
| audit_sync_session.py | schema-v2 原始日志、PNG 哈希、标签与双侧配对重建 | 否 |
| export_session.py | 所有原始文件复制、通用 CSV、校验和、可选 ZIP | 否 |

HPS ABI 为 32 位 int + 264 字节 hps_ft_info，必须对照现场 SDK；代码默认 accepted_codes=10000、115200 baud，不能未经核查当所有力计适用。Windows 64 位 Python 与 DLL 必须匹配。相机要求 640x480 uint8 BGR，错误尺寸不隐式缩放。

本地离线环境：Python 3.12.13，NumPy 2.5.3，opencv-python 5.0.0.93；版本与 PyPI 对应：[NumPy](https://pypi.org/project/numpy/2.5.3/)、[OpenCV](https://pypi.org/project/opencv-python/5.0.0.93/)。设备端使用新 venv，不升级原生产环境。

## 安装与预检

在仓库根目录，用现场实际路径替换下列示例；不要使用 device/test_force_profile.txt，它仅供离线测试。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r device/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m unittest discover -s device -p 'test_*.py' -v
New-Item -ItemType Directory -Force local_data | Out-Null
$profilePath = 'C:\ACTUAL_SDK\profiles\HSMVBTS_G01_H20_000011.txt'
$forceDllPath = 'C:\ACTUAL_SDK\bin\hps_ft.dll'
.\.venv\Scripts\python.exe device/inventory_device.py --profile "$profilePath" --ft-dll "$forceDllPath" --output local_data/preflight_001.json
.\.venv\Scripts\python.exe device/collect_sync_dataset.py --profile "$profilePath" --ft-dll "$forceDllPath" --camera-index 0 --port COM3 --check-config
```

替换 camera-index、COM3、baud 等为现场已确认值；不能通过检查返回值推断设备已经连接。预检文件不覆盖，重跑换名字。inventory 中的路径仅是声明的文件身份，实际加载仍需设备日志/运行入口证据。

## 人员确认后采集

```powershell
.\.venv\Scripts\python.exe device/collect_sync_dataset.py --profile "$profilePath" --ft-dll "$forceDllPath" --camera-index 0 --port COM3 --output-dir local_data/captures --align-offset-ms 0 --max-pair-gap-ms 30
```

30 ms 是软件默认双侧 bracket 上限，不是同步精度或合格结论；不要按测试误差调它。默认 0 偏移表示未校准主机时间配对，不应用 profile 的 VALIDATION_LATENCY_MS。

窗口操作：人员确认完全无接触后按 T（会实际 tare 并等待新的参考及双侧力样本），按 R 开始/暂停，ESC 关闭并落盘。力流在暂停时仍记录。按 T 是物理零点操作，不应由 agent 在带载状态自动执行。

源 schema-v2 文件：session.json、force_raw.csv、capture_sync.csv、events.jsonl、sensor_profile_snapshot.txt、frame_*.png、reference_*.png/json。异常和无效标签保留为空，不补 0。用 session.json 的 status 判定 closed/failed，不把有文件当成功。

## 审计与无删行导出

```powershell
$sessionPath = 'C:\ACTUAL_REPO\local_data\captures\session_ACTUAL_ID'
.\.venv\Scripts\python.exe device/audit_sync_session.py "$sessionPath" --report local_data/audit_001.json
.\.venv\Scripts\python.exe device/export_session.py "$sessionPath" --output local_data/exports/session_ACTUAL_ID --identity local_data/preflight_001.json --origin new_capture --zip
.\.venv\Scripts\python.exe tools/session_integrity.py verify local_data/exports/session_ACTUAL_ID
```

有人工动作记录时加 `--actions <实际CSV>`。没有则缺失，不自动生成动作标签。`--origin` 必须如实选 new_capture / historical / synthetic / unverified，审计通过不自动推断来源。

输出包含字节不变的 raw_source/，另有 camera_frames.csv、force_samples_raw.csv、force_samples_transformed.csv、alignment.csv、predictions.csv、source_inventory.json、quality_report.json、session_manifest.json、checksums.sha256。原始采集与便携 schema 的映射见 exporter；schema-v2 是基础合同的受限实现，不冒称所有要求已具备。

ZIP 与 receipt 在导出目录旁，receipt 记录包大小/SHA256。原始数据与导出目录不可重叠；已有输出不覆盖。审计失败默认阻止导出；为保留失败证据可显式加 --allow-audit-fail，manifest/quality_report 保持失败和 NOT_READY，不能用于训练资格放行。

audit_sync_session.py 的 --manifest 是另一个会排除无效行的训练适配器；本任务不要使用它来代替完整诊断导出。

## 已知缺口必须交接

- 相机采用 latest-frame：只保存记录器选中的帧，可能跳号；暂停期也不保存全部图像。不能声称全部曝光/全部已接收帧落盘。需要逐曝光全保存时另开适配任务。
- 时间戳是主机 grab 调用与 FT DLL 调用时间；硬件曝光/采样时间不可得，物理同步未验证。
- raw_force 是 DLL 在当前 tare epoch 返回的值，不能恢复硬件清零前信号。
- 单位、坐标、力矩原点和标定必须现场补证；软件 N/N*m 字段不是物理标定证明。
- 纯采集器不加载模型；main_match 可保存 SDK raw_force/force 与状态，但 raw_force 不是已证明的图内 ONNX 原始输出。两者均不提供最终网络输入张量/位移场 dump；不伪造。
- 对比程序依赖现场 C ABI，检查 --check-config 不验证结构体布局；先核头文件/版本再考虑运行。首轮原始双流不需要 SDK/模型。

## 发布与数据交接

工具和文档通过 PR；小清单、已审查的报告提交 sessions/<id>/。完整 ZIP 留本地，待用户批准的私有 Release/存储路线；不把图片/模型/驱动整体 force-add，不上传秘密。提交访问位置、receipt、质量限制和 REVIEW.md，接收端校验后再分析。代码合并不授权自动换模型。

## 复制后的最小适配

--profile 改为必填；默认数据目录位于忽略的 local_data/captures。采集元数据增加实际传入 profile/DLL 和脚本文件的声明哈希。测试使用明确标注的合成 fixture；生产入口不会默认读取该 fixture。其余同步/变换/采集核心沿用源代码。审计规则保持不变。

## 已执行的离线验证

2026-10-28 Windows/Python 3.12.13：设备相关 36 项测试全部通过（包括原32项与新增4项全量导出/拒绝覆盖/失败审计/清单语义）；仓库完整性 8 项测试通过。五个 CLI --help 与两个 --check-config 成功，未打开硬件。导出测试验证2条保存帧（其中1条无效配对）都保留，原始字节/参考不变、ZIP CRC与包SHA256一致。实际采集、SDK ABI、最终输入张量、硬件同步与物理精度均未验证。
