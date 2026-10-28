# Session 数据合同 v1

## 文件与身份

实际数据包采用唯一 session ID，包含：

```text
session_manifest.json
config/                       实际配置/程序身份
reference/                    参考帧
images/                       所有已接收并保存的原始帧
camera_frames.csv
force_samples_raw.csv
force_samples_transformed.csv
alignment.csv
actions.csv
tensors/                      实际 SDK/推理入口张量
predictions.csv
quality_report.json
logs/
checksums.sha256
```

缺失文件/字段写明原因及恢复办法，禁止用空文件宣称已完成。原始流和变换流分开。原始图像优先无损；若已有采集用 JPEG，保留原字节并披露压缩，不转成 PNG 后声称原本无损。

manifest 最小字段见 configs/session_manifest.template.json。模板不是实测数据，必须在采集后填入事实。所有相对文件路径限制在 session 根目录内。

## 字段语义

camera_frames.csv 至少有：frame_id、image_path、host_receive_monotonic_ns、device_timestamp_raw、device_timestamp_unit、exposure_timestamp_raw、save_status、quality_status。设备时间不可得时用空值并解释；不能把接收时间重命名为曝光时间。

force_samples_raw.csv 至少有：force_sample_id、host_receive_monotonic_ns、device_timestamp_raw、device_timestamp_unit、Fx_raw、Fy_raw、Fz_raw、Mx_raw、My_raw、Mz_raw、status、overload。原始轴顺序/单位在 manifest 声明。能取得通信报文时一并保存。

force_samples_transformed.csv 关联原 force_sample_id，保存 tare 后和坐标/力臂变换后的六值；明确公式、顺序、单位与 tare 事件。不要假定目标原点已知。

alignment.csv 至少有：frame_id、time_basis、method、force_before_id、force_after_id、weight_after、nearest_delta_ns、bracket_gap_ns、valid、reason。最近邻填写实际样本；插值填写两端及权重。记录允许时间差及依据，不从测试误差反选阈值。

actions.csv 至少有：action_id、start_monotonic_ns、end_monotonic_ns、description、tool、contact_position、operator_confirmation。无载由现场确认，不能把标签强制置零。

predictions.csv 至少有：frame_id、model_sha256、profile_sha256、input_tensor_path、input_tensor_sha256、raw_Fx/raw_Fy/raw_Fz/raw_Mx/raw_My/raw_Mz、post_Fx/post_Fy/post_Fz/post_Mx/post_My/post_Mz、quality_status、error。未推理也保留对应失败记录。

## 时间与质量

相机和力计线程尽可能共用同一主机单调时钟；记录 API、单位、原点及 UTC 锚点/时区。保留整数精度，禁止经 Excel 舍入。设备计时器记录频率、回绕/重启、时钟映射、缓冲批次；没有硬件同步证据写“主机时间对齐，物理同步精度未验证”。

quality_report 记录接收/保存/失败数、力采样数、时间倒退/重复/间隙、缺图、NaN、过载、配对时间差分布、参考更新和断流事件。原始帧与保存行数不等于独立曝光数。未知写 UNKNOWN，不用 0 代替。

完整性工具仅检查文件路径/字节/清单覆盖；字段、同步、标定和物理精度必须另外审核。
