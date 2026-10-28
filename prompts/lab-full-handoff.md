# 实验室 PC agent 完整接续任务

请接续 VBTS 与六维力同步诊断，仓库为 https://github.com/Wizcoo/vbts-lab-bridge ，私有。目标是保留可追溯数据交给分析端定位多个模型实测不佳的首个失配环节；设备端不需要训练 dataset，不做新训练、不覆盖现役模型，不自动驱动机械加载。

## 1. 拉取、确认和分支

如果尚未克隆，git clone 上述仓库；已克隆则先 git status，保留本地改动，再按现有分支正常更新，禁止 reset/clean/force push。从最新 main 建立 codex/lab-session-<唯一时间> 分支。先读 AGENTS.md、docs/session-contract.md、docs/device-tools.md 和本文件。记录实际 commit。

仓库包含 device/ 下的采集、预检、审计、无删行导出和可选 SDK 比较脚本；不包含力计 DLL、相机驱动、SDK DLL、模型或生产 profile。先检查现场已有工程，使用已确认的实际文件路径，不用测试 fixture 当设备配置，不为了运行去改现役 profile。共享合同调整需通过 PR。

## 2. 无硬件动作的准备

确认 Windows x64、Python 3.12 x64、现场 hps_ft.dll 架构/版本/函数/264字节 info ABI、相机索引/分辨率、力计串口/波特率、真实 profile。不要盲目枚举并打开所有设备。记录型号/序列号、现有 tare/轴符号/力臂、标定来源；未知写 UNKNOWN。

在新 venv 安装 device/requirements.txt，运行两组 unittest 和 --help。完整命令见 docs/device-tools.md。为测试问题使用隔离临时目录，不污染生产。保留版本、日志和退出码；所有测试都是离线假设备，不能称实测。

用 device/inventory_device.py --profile <实际TXT> --ft-dll <实际DLL> --output <新报告.json> 生成只读文件身份报告，可加 --sdk-dll、--model 和重复 --asset <文件>。该报告不证明实际加载，只保存声明文件的身份；运行日志另核。

用 device/collect_sync_dataset.py --profile <实际TXT> --ft-dll <实际DLL> --camera-index <已确认索引> --port <实际串口> --check-config 检查。缺文件、设备未连接、权限或 ABI 不明时，完成可做的离线工作并报告 BLOCKED，不伪造数据，不默默改用模拟源。不要再次要求传训练 dataset。

## 3. 等我确认设备和开始采集

准备好后告诉我具体设备/端口/路径及准备执行的命令，等我确认“已连接，可以开始”。不要因脚本存在就自行开始硬件采集或 tare。确认后运行纯采集器，输出到仓库忽略的 local_data/captures 或独立本地盘；参数必须引用实际路径。默认 --align-offset-ms 0，不应用 profile 的 VALIDATION_LATENCY_MS，不按测试误差搜索时间偏移。

窗口交给现场人员：确认完全无载/无接触后按 T（实际 tare+建立新参考），按 R 开始/暂停，ESC 结束。不要由 agent 在未知载荷状态自动按 T。由人员在安全范围内执行：起始无载、缓慢加载、保持、卸载、重复按压和结束无载；其他位置/切向/扭转仅在人员批准范围内进行。记录工具、位置、动作起止与无载确认，不要求探索载荷极限。

保存每个实际记录的原始 PNG、参考及事件、所有 FT DLL 读数/失败/重复IPOC、主机时间和双侧配对。原始数据不改、不删难例，不强制无载标签为零。程序异常保留 failed session，不为了审计通过删错误。

必须说明当前 latest-frame 策略可能跳过中间帧、暂停时不保存全部图像；主机接收/调用时间不等于硬件采样/曝光时间。最终 SDK 输入张量和位移 dump 尚缺失，不能拿 Python 重算或零数组冒充。先完成可用原始双流；若为当前诊断必须补这些能力，列为下一有限适配任务并保留首轮结果。

首轮不要求运行 main_match.py。若确需保存现场模型预测，先核现场 HSM SDK 头文件/ABI 与脚本结构体及实际 ONNX/TXT 配套身份，得到人员许可后再用可选比较入口；保留 SDK raw_force/force，不把前者直接称 ONNX 图原始输出。不要替换模型。

## 4. 采集结束后审计、导出

确认 session.json 状态；用 device/audit_sync_session.py <session> --report <原session外的新路径.json>。只保存帧完整性与主机配对通过不等于硬件同步/标定/精度通过。

用 device/export_session.py <session> --output <新的导出目录> --identity <本轮inventory.json> --origin new_capture --zip；有实际人工动作CSV时加 --actions <文件>。历史数据选 historical，测试选 synthetic，不确定选 unverified。最后用 tools/session_integrity.py verify <导出目录>。

exporter 保留 raw_source 字节和全部行，包括无效标签，再输出通用CSV、质量报告、manifest、checksums、ZIP/receipt。不要用 audit 的训练适配器 --manifest 代替完整诊断数据。审计失败时先报告失败原因；确需交接失败证据可加 --allow-audit-fail，但保留失败和 NOT_READY。已有输出不覆盖。

## 5. PR 与交接

在 sessions/<session_id>/ 提交已审查的小型 manifest、quality_report、receipt、REVIEW.md、实际命令/环境和未解决问题。大包先留本地，使用我批准的私有附件位置，不自动公开链接、上传专有驱动/模型/完整dataset或批量 force-add 图像。若暂无批准的可访问传输位置，清楚报告“包已生成，上传待定”，不要声称另一端可下载。正常 git push 当前分支并开 PR；没有创建权限时提供分支和可点击比较链接，不制造 PR 成功。

最终给我：PR/分支链接、实际是否接硬件、来源类别、采集时间和帧/FT样本数、原session/导出/ZIP本地路径、包SHA256和大小、批准的下载位置、审计/完整性各自结果、同步/标定/坐标/张量缺口和下一步。别只发“脚本运行成功”。
