# FoG-STAR 冻结步态检测最终技术方案与确认性实验报告

## 1. 结论摘要

本项目基于 FoG-STAR v3.0 真实数据完成了数据审计、公开 1D-CNN 强基线复现、跨部位信息探针、时序上下文实验、5 次随机种子稳定性实验以及患者级 LOSO 内部确认性实验。综合结果后，最终模型冻结为 **ACTE-FoG（Ankle-Coordination Temporal Ensemble）**。

ACTE-FoG 的核心原则是：以双踝信号作为主要判别信息，以全身多部位协调特征提供上下文补充，再利用短时序概率解码处理 FoG 的局部持续性和转变特征。实验不支持“24 路原始信号全部直接进入更复杂深度网络”作为当前数据规模下的首选路线。

固定 11/5/6 患者划分下，5 个随机种子得到测试 AUROC **0.9318±0.0003**、AUPRC **0.7735±0.0021**、F1 **0.7336±0.0027**。但该固定测试集在探索阶段被反复观察，因此该数字应视为探索性结果，不能作为无偏泛化性能的唯一依据。

冻结模型后进行患者级 LOSO 内部确认，完整实验得到 pooled AUROC 约 **0.9080**、AUPRC 约 **0.6878**，16 位同时存在正负样本的患者 macro AUROC 约 **0.8879**。快速复核实验（30 trees）得到 pooled AUROC **0.9046**，与完整 LOSO 结论一致。模型在 Walk、Turn-left、Turn-right 场景表现较好，Standing 是当前主要剩余难点。

## 2. 数据与实验协议

- 数据集：FoG-STAR v3.0
- 患者：22 例
- 采样率：60 Hz
- 传感器位置：左踝、右踝、腰背、腕部
- 每个位置：三轴加速度 + 三轴陀螺仪，共 24 路惯性信号
- 标签：FoG、activity、FoG severity、subject/session/task 标识
- 预处理：0.5–20 Hz 四阶零相位 Butterworth 带通滤波
- 窗口：2 s，即 120 个采样点
- 滑动：0.5 s，即 75% overlap
- 标签：窗口内 FoG 多数投票
- 去均值：窗口内逐通道去均值
- 分组原则：任何窗口均不得跨 subject/session/task 边界
- 缺失：显式保留 missingness 比例，滤波仅在连续有限值区间内执行

固定复现划分：Train=1–11，Validation=12–16，Test=17–22。该划分与公开论文所报告的主要临床统计和非冻结患者 4/1/1 分布高度吻合。

最终评价以患者独立为原则。由于 6、7、8、10、14、18 号患者没有 FoG 阳性窗口，单患者 AUROC 对这 6 位患者不可定义。因此 LOSO 同时报告：

1. 全部 LOSO 预测汇总后的 pooled AUROC/AUPRC；
2. 仅在同时含正负样本的患者上计算 subject-macro AUROC。

## 3. 公开强基线复现

公开 1D-CNN 的单传感器输入为 120×3，即 2 秒单个三轴陀螺仪信号。网络采用多层因果 Conv1D、MaxPooling、Dropout、Global Average Pooling 和 Dense 分类头。

在当前 FoG-STAR v3.0 数据版本、严格患者划分和公开训练逻辑下，本地完成的受控运行测试 AUROC 约为 **0.897–0.900**。其中一个严格复现实验得到：

- Validation AUROC：0.8557
- Test AUROC：0.8967
- Test AUPRC：0.6875
- Test F1：0.6441（验证阈值）

公开论文左踝 gyro 参考结果约为 Validation AUROC 0.920±0.008、Test AUROC 0.909。当前 v3.0 与论文时期数据版本、窗口总量及临床/标注信息存在差异，因此不要求数值完全一致，但本地结果已经处于同一性能量级。

## 4. 最终模型：ACTE-FoG

### 4.1 总体结构

ACTE-FoG = **双踝主导表征 + 全身协调摘要 + 缺失感知 + 集成分类 + 短时序概率解码**。

模型不把四个部位的 24 路原始信号等权处理。研究发现踝部对 FoG 判别最稳定，因此采用“ankle-first”的层级结构，将全身信息限制为紧凑的协调上下文，以降低小样本条件下的跨患者过拟合。

### 4.2 双踝局部表征

双踝共 12 路原始通道。每个 2 秒窗口提取：

- 标准差 STD；
- 均方根 RMS；
- 一阶差分平均绝对幅值 MAD-like dynamic feature；
- 冻结指数 Freeze Index：3–8 Hz 功率相对 0.5–3 Hz locomotor band 的比值，经 log1p 变换；
- 通道缺失比例。

这些特征同时刻画运动幅度、动态变化、FoG 高频成分及数据可靠性。

### 4.3 全身协调表征

对左踝、右踝、腰背、腕部四个节点分别计算加速度模长和陀螺仪模长，在一个窗口内计算四节点两两相关性。

四个节点共有 C(4,2)=6 对，Acc 和 Gyro 两种模态共形成 **12 个协调特征**。该设计保留“上下肢和躯干是否失去同步”的信息，但避免将全部原始 24 路高维时序直接交给模型。

### 4.4 分类器

最终分类器采用 ExtraTrees：

- n_estimators=500（正式固定划分实验）；
- min_samples_leaf=3；
- max_features=0.5；
- class_weight=balanced。

ExtraTrees 在当前样本规模下表现出较好的非线性表达能力和随机种子稳定性，同时便于后续特征重要性和通路贡献分析。

### 4.5 短时序概率解码

FoG 并非独立的单窗口事件，尤其在转身和启动阶段具有明显的前后时序依赖。因此对窗口级预测概率进行 7-window 时序聚合。

窗口步长为 0.5 s，7 个相邻预测对应约 5 s 的有效原始信号覆盖范围。离线研究采用 centered decoder；实时部署应使用 causal decoder，仅聚合当前及历史窗口，避免未来信息。

## 5. 固定 11/5/6 划分结果

5 个随机种子：

| 指标 | ACTE-FoG |
|---|---:|
| Validation AUROC | 0.8760 ± 0.0019 |
| Validation AUPRC | 0.4362 ± 0.0031 |
| Test AUROC | **0.9318 ± 0.0003** |
| Test AUPRC | **0.7735 ± 0.0021** |
| Test F1 | **0.7336 ± 0.0027** |
| Sensitivity | **0.8378 ± 0.0059** |
| Specificity | **0.8853 ± 0.0039** |
| Accuracy | **0.8756 ± 0.0022** |

该结果证明双踝主导、全身协调和短时序上下文是有效组合。但由于测试集参与了探索阶段的模型选择，该结果定位为 exploratory performance。

测试集活动分层（seed 2）：

- Walk AUROC：0.944
- Turn-right AUROC：0.931，Sensitivity 88.5%，FPR 8.3%
- Turn-left AUROC：0.926，Sensitivity 84.7%，FPR 10.7%
- Standing 是主要残余假阳性来源。

## 6. LOSO 内部确认性实验

冻结 ACTE-FoG 后，对 22 位患者逐一留一：每次用 21 位训练，对剩余 1 位预测；预测仅在同一 subject/task 内进行 7-window 时序解码。

完整 LOSO 结果：

- pooled AUROC：约 **0.9080**
- pooled AUPRC：约 **0.6878**
- subject-macro AUROC：约 **0.8879**（16 位可定义 AUROC 的患者）

活动分层结果显示：

- Walk AUROC：约 **0.973**
- Turn-left AUROC：约 **0.902**
- Turn-right AUROC：约 **0.854**
- Stand AUROC：约 **0.612**

快速工程复核（30 trees, seed 2）得到 pooled AUROC 0.9046、AUPRC 0.6886、macro AUROC 0.8807，与完整 LOSO 方向一致，说明确认性结论稳定。

## 7. Hard-negative 实验结论

针对 Standing/Turning 负样本增加训练权重，不在推理时使用 activity 真值。快速 LOSO 对照：

| 指标 | 原 ACTE-FoG | Hard-negative |
|---|---:|---:|
| pooled AUROC | 0.9046 | 0.9060 |
| pooled AUPRC | 0.6886 | 0.6928 |
| macro AUROC | 0.8807 | 0.8901 |
| Turn-right AUROC | 0.8465 | 0.8565 |
| Turn-left AUROC | 0.8988 | 0.9000 |
| Stand AUROC | 0.6147 | 0.5988 |

Hard-negative 对总体和转身场景有小幅帮助，但没有解决 Standing，甚至略有下降。因此：

- 可以保留为可选训练策略；
- 不作为最终模型核心创新点；
- 不据此继续修改主模型结构。

## 8. 最终有效结论

### 8.1 已被实验支持的部分

1. **踝部是主要信息源**：单踝公开强基线已经较高，双踝进一步提供左右步态互补信息。
2. **全身协调信息有增量价值**：四节点 Acc/Gyro 模长相关性作为紧凑上下文，可以补充仅踝部信号无法表达的全身运动失协调。
3. **短时序上下文有效**：FoG 的判断需要相邻窗口，而非仅依赖一个孤立 2 秒片段。
4. **简单堆叠 24 路原始信号并非最优**：高维全身输入在固定测试集上可以很高，但验证集和 LOSO 稳定性不足。
5. **模型存在明显患者异质性**：固定测试集 0.9318，而 LOSO pooled 约 0.908，说明 split sensitivity 必须被正式报告。

### 8.2 当前不能过度宣称的部分

1. 不能将固定 Test AUROC 0.9318 直接称为“无偏泛化性能”；
2. 不能声称 Standing 问题已经解决；
3. 不能仅根据 22 例内部数据声称临床普适性；
4. centered 7-window decoder 是离线算法，实时系统必须使用 causal 版本重新报告性能和延迟；
5. 需要独立外部数据集或前瞻性队列完成最终外部验证。

## 9. 推荐论文主线

建议论文围绕以下科学问题组织：

> 强单踝 FoG 检测器已经可以获得较高性能，但 FoG 尤其在转身和运动转换阶段表现为跨身体部位的协调失稳，并具有短时持续和演化特征。能否在保持踝部强判别能力的基础上，用紧凑的全身协调表征和短时序上下文提高患者独立 FoG 检测的鲁棒性？

方法贡献可概括为：

- ankle-dominant hierarchical representation；
- compact cross-body coordination modeling；
- short-context temporal decoding；
- missingness-aware processing；
- subject-independent confirmatory evaluation。

主结果建议以 LOSO 为核心，固定 11/5/6 结果作为与原文协议可比的 secondary analysis。

## 10. 后续研究优先级

### 第一优先级：外部验证
在另一 FoG 公共数据集或独立临床队列冻结参数直接测试，避免继续在 FoG-STAR 上调参。

### 第二优先级：Standing 专项机制
不要直接用 activity 标签修正预测。应研究可观测运动学信号，如低动态能量、姿态稳定性、启动意图、踝-躯干相位变化，构造可推理的 stand/transition context gate。

### 第三优先级：事件级评价
增加 episode-level sensitivity、false alarms/hour、onset latency、minimum detected episode duration。对于临床和可穿戴实时告警，这些指标比单纯窗口 AUROC 更重要。

### 第四优先级：连续疾病状态量化
在完成二分类检测后，再将 FoG probability、episode burden、duration、transition instability 和 severity 组合为患者级连续 FoG burden score，并与临床量表、病程及治疗前后变化验证。

## 11. 最终结论

本轮研究已经从“寻找复杂模型”收敛为一套可以解释、可以复现、与公开强基线公平比较、并经过患者级内部确认的 FoG 检测方案。最终推荐模型为 ACTE-FoG。固定协议下其测试 AUROC 约 0.932；更严格的患者级 LOSO pooled AUROC 约 0.908。后者更适合作为当前阶段对真实跨患者泛化能力的估计。

当前模型的优势主要来自踝部强局部表征、全身协调补充和短时序上下文；主要剩余问题是 Standing/transition hard negatives 与患者异质性。工程主线已经可以冻结，后续工作应转向外部验证、事件级临床指标和连续疾病状态量化，而不是继续堆叠模型复杂度。
