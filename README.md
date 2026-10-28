# VBTS Lab Bridge

本机分析端与实验室 PC 的协作仓库。设备端不需要训练 dataset；通过 PR 交换工具、诊断 session 清单、日志和审核意见。

## 第一轮目标

完成一次可追溯的 `真实采集 -> 数据交接 -> 本机检查 -> 离线复算`。初始化仓库没有采集数据、模型、硬件驱动，也没有证明物理精度。

1. 两端克隆本仓库，先读 [AGENTS.md](AGENTS.md)。
2. 实验室 agent 按 [首轮任务](prompts/lab-first-session.md) 清点设备、现有 session 和采集入口。
3. 现场人员确认连接和安全动作后，记录同一次加载中的相机与六维力数据。
4. 按 [session 合同](docs/session-contract.md) 提交清单和完整性结果。
5. 本机 agent 按 [审核任务](prompts/analysis-review.md) 检查，再判断下一步。

## 内容边界

- `tools/`：离线完整性工具，Python 3.10+，仅标准库。
- `docs/`：两端流程、数据合同、初始化决策。
- `prompts/`：可直接给两端 agent 的任务。
- `sessions/`：每次交接一个目录，保存元数据与审查结果。
- `configs/`：非敏感配置模板；实际配置须绑定 SHA256。
- `experiments/`：冻结的实验方案、模型身份和结论。
- `.github/`：PR 模板、问题模板及基础检查。

大数据留在本地或明确批准的私有附件位置。Git 默认忽略图像、张量、权重、数据压缩包、DLL/EXE 和私有配置。小型真实数据确需入 Git 时，先确认目标是私有仓库并逐个审查、显式添加；不要整体强制添加目录。

## 可立即运行

```powershell
python tools/session_integrity.py --help
python -m unittest discover -s tests -v
```

在实际 session 根目录生成和检查完整性清单：

```powershell
python tools/session_integrity.py seal C:/capture/session_YYYYMMDD_HHMMSS
python tools/session_integrity.py verify C:/capture/session_YYYYMMDD_HHMMSS
```

seal 只读取文件并新建 `checksums.sha256`，不覆盖已有清单。verify 要求全部文件与清单一致；它只验证字节完整性，不验证时间同步、GT、指标或 SDK 一致性。

现阶段没有跨平台采集器；不把旧工程依赖硬件 DLL 的程序冒充本仓库开箱即用的采集功能。
