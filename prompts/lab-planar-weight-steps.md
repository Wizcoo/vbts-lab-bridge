# 实验室 PC：平面夹爪 ONNX 与砝码阶梯验证

以下正文可直接交给实验室 PC 上的 agent。这是待执行任务，不是已完成实验报告。

---

请在实验室 PC 接续平面夹爪法向力验证，仓库为 https://github.com/Wizcoo/vbts-lab-bridge ，工具位于 PR #4：https://github.com/Wizcoo/vbts-lab-bridge/pull/4 。目标是在现场展示并记录真实砝码加载产生的 Fz 阶梯，先检查软件，再进行经人员确认的物理测试。不要重新训练、修改模型或覆盖现役 SDK 资产。

## 1. 已知事实与执行边界

- 设备：平面夹爪 `HSMVBTS_G01_H20_000011`，不要混用指尖工具或其他传感器 profile。
- 候选模型由球头数据训练；原采集 Fx/Fy/Fz 单位 N、before_contact 无载已由人员确认。这不证明本轮启动时已经无载，也不证明平面砝码泛化。
- 本工具仅输出原始有符号 Fz，未监督其余五轴。输入 `raw_uv float32[N,2,69,82]` 与 `valid bool[N,1,69,82]`，输出 `signed_fz_n float32[N]`；归一化和输出反归一化在 ONNX 图内。不能替换现役六轴 SDK 的 ONNX/TXT。
- 可用砝码标称质量为 10、50、100、500、1000 g。可用不等于全部获准加载；开始前由现场人员确认固定方式、受力面、加载路线及允许上限。没有额定量程/机械承载证据时，不自行试探极限。
- 仓库当前公开。用户已批准本次代码和四个 ONNX 公开，不代表批准公开新采集图像、力计数据、设备私有配置或实验结果。新 session 留本地；上传前逐项确认内容和目标位置。
- 不关闭现场正在工作的客户端，不抢占相机，不控制机器人/位移台，不自动 tare 或按键。已有明确的现场授权可沿用其范围；缺少时只在首次打开硬件前向人员确认，不反复询问已授权动作。

## 2. 准备正确 checkout 和隔离环境（无需打开相机）

先读 `AGENTS.md`、`README.md`、`docs/session-contract.md`、`tools/planar_force_live/README.md` 和本文件。保留其他任务的未提交改动，不 reset/clean/force push。当前 main 可能还没有该工具，应获取 PR 分支，在新的工作目录/独立 worktree 运行：

```powershell
git fetch origin codex/analysis-planar-onnx-live-fz
# WORKTREE_PATH 换成新的绝对目录；SESSION_ID 换成唯一标识，不能原样执行占位符。
git worktree add -b codex/lab-SESSION_ID WORKTREE_PATH origin/codex/analysis-planar-onnx-live-fz
```

未克隆时可先 `git clone https://github.com/Wizcoo/vbts-lab-bridge.git`。记录仓库 URL、`git rev-parse HEAD`、`git status --short`、OS、Python 架构/版本和实际执行路径；冻结本轮 checkout 和模型身份后不自动更新。首个运行时提交是 `76508fb031a2fad2a037bde0a0c2550df8e38bed`，后续说明文件提交可在其上追加，以实际 checkout 为准。

进入 `tools/planar_force_live`，使用 Windows x64 / Python 3.12 的独立环境。已有匹配环境可复用；否则在该新工作目录创建 `.venv` 并安装 `requirements.txt`，不升级或修改生产环境。保存安装/检查日志及退出码：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B planar_force.py offline-check
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\start_live.cmd --help
```

若 `py` 不存在，先找到真实 Python 3.12 路径，不能凭空填写。网络或依赖失败则保留日志并报告 BLOCKED；不得静默换版本。预期是 4 个 ONNX 加载及 14 项单测通过，状态包含 `PASS_OFFLINE_IMPORT_GRAPH_AND_FRONTEND_ONLY`、`camera_accessed=false`、`synthetic=true`。这些数值是模拟检查，不是测力结果。

读取并保存 `models/manifest.json`、逐模型 manifest、`frontend_contract.json` 和实际 SHA256；保留原始六个 vendor 文件，不因哈希失败去修改 manifest 放行。记录相机候选索引来自何处、设备标签与连接方式；程序不能验证物理序列号。不要通过逐个打开所有相机来猜索引。

## 3. 现场开始确认与首轮命令

准备好后向现场人员一次说明实际设备、相机索引、允许载荷、摆放方式和将执行的命令。若未获相应授权，等待“设备已连接、触面无载、可开始”的明确确认。不要因本 prompt 或 PR 存在就自动打开硬件。人员负责手动加载。

试验时触面保持水平，砝码载荷完整通过已确认的触面传递；手、外壳、支架或夹具不得分担载荷。若使用托盘/垫片，记录其质量及是否保留在参考中。托盘已接触时不是“完全无载”；本轮应从真实卸载参考启动，并把后续托盘列为独立载荷事件。若接触状态无法满足此要求，停止并说明。

在已确认的 640×480 BGR 相机上启动，首次成功读到的帧会成为整轮固定参考。填写实际索引；下例不会运行，直到人员提供索引并确认后才执行：

```powershell
# 在 tools/planar_force_live 目录；.venv 已通过上述检查。
$cameraIndex = [int](Read-Host '现场确认的相机索引')
$sessionId = 'planar_weights_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$sessionPath = Join-Path $PWD ('recordings\' + $sessionId)
# 不提前创建 sessionPath；程序要求它不存在。
.\start_live.cmd --camera-index $cameraIndex --sensor-id HSMVBTS_G01_H20_000011 --declare-unloaded --model cnn_uv_formal_seed17 --duration-seconds 300 --output $sessionPath
```

默认 DirectShow；若失败，保存失败 session，查明原因后在新 session 显式尝试 `--backend msmf`，不得静默切换其他相机。CMD 直接调用 Python，不需放宽 PowerShell 执行策略。若复用其他环境，先将 `$env:PLANAR_FORCE_PYTHON` 设为已核实的解释器绝对路径。

第一轮不按 T，不设 `--normal-sign`，观察原始有符号 Fz。`Q`/`Esc` 退出，Ctrl+C 也保留记录。连续五帧错误会结束；保留所有失败行，不删帧或循环重试至看似成功。300 秒只是主机循环期限，驱动阻塞时不保证能及时退出；不要自动杀死无关客户端。

## 4. 冻结实验方案，优先完成 A

在看曲线之前固定组别、质量顺序、重复次数和 plateau 统计窗口。每个载荷级别记录“放置开始、手离开、保持结束、取下结束”等人员确认事件，时刻使用同一主机 Python `time.perf_counter_ns()`；单独注明人工反应/记录延迟。不能从 Fz 阈值反推动作，再称其为人工真值。

**A：中心法向阶梯，首轮约 3–5 分钟。** 无载 15 秒；对获准的标称质量按升序各放置并保持 15 秒、完全卸载 10 秒。它们是逐个砝码的独立绝对载荷，不是 10+50+100+500+1000 g 累加。1000 g 未获准则不做，明确记为 NOT_RUN。若人员操作使计划超过 300 秒，应正常结束当前轮，在新的无载 session 继续，保留所有部分结果。

**B：重复性与卸载回零。** 独立 session 固定同一位置、相同已批准砝码（优先 100 g 或现场选定载荷），加载 15 秒/卸载 10 秒，重复 3 次；固定参考，不在每次卸载重新 tare。保留初始和最终无载段，用于计算回零变化。需要更长保持时单独安排 60 秒平台，报告观测时长，不外推长期漂移。

**C：位置/接触泛化。** 在 A 完成后，以同一已批准质量比较中心和现场确认可用的两个偏置位置，每位置重复 3 次，记录位置、接触面积、朝向和工具。不得让砝码接触边框或支架。该组只能表征已测试位置，不代表整个触面的泛化。

**D：四模型对同一原始数据复算，无新增硬件操作。** 先封存 A 的原始 session，再基于同一固定参考、同一图像序列，分别用四个模型的 `replay` 命令生成新输出目录；按原始 `attempt_index` 的整数顺序，不能对文件路径做字典序猜测。参考对应 `REFERENCE_CAPTURED` 行，不把后续无载图替换为新参考。原始失败/缺图行保留在源记录和映射中，不能因只重放成功保存图像而报告全帧成功。

`replay --reference PATH --images PATH1 PATH2 ...` 只适用于命令行能容纳的小批诊断。完整 session 不要一次展开数千路径、截断、无记录地抽样或分块重置参考；如果需完整四模型回放，交给分析端使用保留每个源 attempt、参考和失败行的离线入口。未完成则写 NOT_RUN，并提供源数据清单。运行时 replay 的时间是本次读文件时间，回映原始时间必须使用源 frame_id/attempt_index，不能当成原采集时间。

若时间紧，仅完成 A 和少量 B；其余明确 NOT_RUN，不为展示好看而平滑、裁剪、强制归零或按测试数据拟合比例。

标称质量换算仅作参考：`F = mass_g / 1000 * 9.80665`，并非经过校准的六轴真值：

| 标称质量 g | 参考力 N |
| --- | --- |
| 10 | 0.0980665 |
| 50 | 0.4903325 |
| 100 | 0.980665 |
| 500 | 4.903325 |
| 1000 | 9.80665 |

记录质量来源、校准/不确定度及载荷传递条件；未知写 UNKNOWN。不要因为软件导出误差很小而声称可分辨 10 g。

## 5. 动作与力计证据

本工具不会自动生成砝码动作标签。将辅助记录写到 `sessionPath` 外的独立 `sessionId_evidence` 目录，至少保存 `actions.csv`、`LAB_REPORT.md` 和命令日志。`actions.csv` 建议列：

```text
action_id,group,repeat_index,start_monotonic_ns,end_monotonic_ns,description,mass_g,mass_source,tool,contact_position,operator_confirmation,timing_method,timing_uncertainty
```

`timing_method` 写实际记录方式，延迟未知写 UNKNOWN；不要制造纳秒精度的物理动作时刻。若无动作记录渠道，先向人员安排实际记录方式；只采到了曲线时，禁止补造精确 plateau 边界。准备时创建表头不等于记录完成。

有校准六维力计且已获授权时，应保存独立原始力流、轴/单位/符号/力矩原点、tare 事件及同一主机时间，按 `docs/session-contract.md` 留下配对证据。没有可用的独立力计入口时，先完成砝码实验，力计项写 NOT_RUN；不要同时启动另一个会抢占同一相机的采集器。无力计时只能做砝码参考比较，不能称完成六轴标定。

源 Fz 的压缩符号若无证据，写 UNKNOWN；先报告原始有符号趋势和无载/加载差值。不要取绝对值后称有符号精度通过。需要演示翻转显示时，人员明确选择后在新 session 设置 `--normal-sign -1`，记录理由。一次 T 仅影响显示且需实际无载确认；绝不能为每个台阶调整零点。所有原始值继续保留。

## 6. 停止、核对、汇报

几何/设备不符、明显滑移、接触边框、过载、持续掉帧、磁盘不足或人员要求停止时立即结束当前测试并记录原因。若在无载或获准小载荷下已明显异常，先定位相机/参考/前端/模型输入链路，不继续盲目加重。

退出后核对 `session.json`、`predictions.csv`、`events.jsonl`、`receipt.json` 和 `raw/`。记录全部 attempt 数、参考行数、有效/DEGRADED/无效/失败数、原图存在及哈希、实际起止时间和退出码。程序原列 `timestamp_ns` 是 `perf_counter_ns` 接收时间，`host_wall_time_ns` 是主机墙钟，二者都不是曝光时间。不要声称保存了相机所有物理曝光。

在每个已人工标注的稳定保持段内，预先规定“手离开后 5–10 秒”的固定统计窗口；未覆盖完整窗口则记为不足，不凭曲线挑最平片段。保留窗口内全部有效值，同时报告无效比例；少于 10 个有效样本时不形成该段统计结论。统计原始 Fz 均值/标准差/P5–P95 区间、同质量跨轮变化、初末无载差和各载荷相对紧邻之前无载段的增量。零点差分是分析量，另存字段，不能回写原始预测。若有证据确定符号和力参考，再报告 Bias/MAE/RMSE/P95绝对误差/最大绝对误差；否则相关精度项 N/A。

计算时间分位数用 CSV 的 frontend/model/processing 等字段；实际主机吞吐量用源时间跨度与 attempt 数说明计算方法。物理延迟、静态精度、%FS 无独立证据时写未验证/N/A；低激励 Pearson 为 N/A，不用相关系数替代精度。

结束写入后可在仓库根目录运行：

```powershell
# SESSION_ABSOLUTE_PATH 换成本轮真实 sessionPath；封存后不再改目录内文件。
python tools/session_integrity.py seal SESSION_ABSOLUTE_PATH
python tools/session_integrity.py verify SESSION_ABSOLUTE_PATH
```

该工具只证明字节与清单完整。原运行时目录不等同完整六轴 session-contract；缺少 SDK 张量、六轴原流、动作或同步项要逐项列明，不生成空文件冒充。完整性通过与物理资格分开报告。

最终给我一份可直接交回分析端的摘要：实际 checkout/模型/前端哈希、解释器与依赖、现场授权范围、设备确认依据、原始 session 与辅助证据绝对路径、执行 A/B/C/D 哪些组、全部行数/错误/退出码、力计是否真实接入、阶梯原始图与统计、未解决问题、校验清单与包 SHA256/大小（仅打包后填写）。

新实验数据与结果默认留本地。若尚无批准且接收端可访问的传输位置，写“本地已封存，上传待授权”，不说已交付。公开仓库的旧 prompt 若仍称它私有，以当前实际可见性为准。不要将新 session、截图、日志或设备隐私自动推到 PR #4；本次公开授权只覆盖既有代码/四个 ONNX 和这份通用测试说明。
