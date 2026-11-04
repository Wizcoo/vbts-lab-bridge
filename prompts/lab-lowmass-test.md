# 低载候选：实验室 PC 接续任务

请从 `https://github.com/Wizcoo/vbts-lab-bridge.git` 的 `codex/analysis-lowmass-demo` 分支测试 `tools/planar_lowmass_demo`。本分支独立包含两模型与运行代码，不需要先合并或复制另一份四模型工具。只做冻结候选的新位置验证，不重新训练或替换现役 SDK。

如果没有本地 checkout，可在新的目录运行：

```powershell
git clone --branch codex/analysis-lowmass-demo --single-branch https://github.com/Wizcoo/vbts-lab-bridge.git vbts-lowmass-lab
cd vbts-lowmass-lab
git rev-parse HEAD
git status --short
```

已有 checkout 时先检查并保留本地工作，fetch 后用独立 worktree/新分支，不 reset、clean、强制切换或关闭正在工作的客户端。先读 `AGENTS.md`、`tools/planar_lowmass_demo/QUICKSTART_ZH.md`、`MODEL_CARD.md` 与 `LAB_CAPTURE_PROMPT_ZH.md`，记录实际提交及模型哈希。

进入 `tools/planar_lowmass_demo`，使用独立 Python 3.12 环境。环境准备及下列检查不打开相机：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B planar_force.py offline-check
```

保留命令/版本/退出码。缺少解释器或依赖时报告实际问题，不静默换版本。预计26项测试与两真实ONNX加载通过；合成检查值不是物理测量。

现场人员确认实际设备ID、相机索引、可用触面、完全无载与本轮采集授权后，再按快速说明手动启动 `start_live.cmd`。已有明确授权可沿用；不要由 agent 自动开未知相机、tare、加载或更换模型。ID冲突（采集快照000015/继承000011）仍需留痕，不能自动改写为已确认。

默认 `lowmass_head60`；`--model lowmass_aug60` 为另一固定候选。先完成同一新数据上的比较，不根据曲线临场选模型后称独立验收。保持 `CALIBRATION FIT ONLY / NOT_READY` 标识；不平滑、吸附档位、强制归零或裁掉错误。10g增量偏弱是当前已知限制。

具体多位置与重复加载步骤、时长参数、动作/六维力记录及完整交接要求见 [独立采集 prompt](../tools/planar_lowmass_demo/LAB_CAPTURE_PROMPT_ZH.md)。原始记录在本地，现有公开授权只覆盖本候选代码/两ONNX及说明，不覆盖新采数据、截图、私有配置或未来模型。测试结束后先报告本地完整会话路径和概况，另行确认数据交接位置。
