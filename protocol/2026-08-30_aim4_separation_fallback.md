# 2026-08-30 Aim 4 完全/准完全分离处理：预设 SAP 修正案

**状态：FROZEN（在本文件 SHA-256 登记至阶段 7 状态包后生效）**  
**适用对象：** Aim 4，医院组织与住院死亡关联  
**证据等级：** associational/supportive；本修正案不改变或加强因果解释。  
**（预设）时间与结果盲态声明：** 本修正案在修正 `state_provider` 语义错误后的 Aim 4 正式分析表重建、正式 volume–outcome 效应估计及其置信区间生成之前制定。先前使用错误州字段的 Aim 4 模型、表和效应数值均已作废，不得用作本修正案的依据或比较对象。

## 1. 原 estimand 与分析总体保持不变

1. 主要总体仍为 2021–2025 年严格成人胆总管结石队列（Cohort B）中可与 RD 匹配、死亡结局有效、具有完整前 12 个月 volume 历史、且满足冻结 context-linkage 规则的 AIH。住院死亡定义为 `MORTE == 1`。
2. 主要暴露仍为医院层面 trailing-12-month、全适应证、唯一 Cohort-A ERCP AIH 量，以既有冻结规则构造，并以 4-knot restricted cubic spline 建模。
3. 主要 estimand 仍为：在同一冻结分析人群的经验协变量分布上标准化后，trailing-12-month ERCP volume 第 90 百分位与第 10 百分位的住院死亡**边际风险差（RD）**。边际风险比（RR）为支持性量；条件 odds ratio（OR）必须标记为条件量，不得替代 RD。
4. 既有协变量、参照类别、非线性项、数据可用性规则和禁止调整项完全保留：年龄 RCS、性别、race/color（含显式缺失类别）、急诊/非择期与择期二元入院类型、K80.3/K80.4/K80.5 诊断层、预入院诊断字段可得的合并症负担、同月医院类型/SUS 床位/ICU 能力/相关服务能力、municipal IVS/ANS context、provider UF 固定效应及 calendar-month 固定效应。ICU 使用、住院日、报销额、任何后暴露中介和数据驱动变量筛选继续禁止进入死亡主模型。
5. 本修正案不改变 Aim 4 在 `config/estimands.yaml` 与 `reports/SAP_v1.md` 中的关联性定位；不得以本模型、bootstrap 或任何敏感性分析声称 volume 的因果效应、医院排名或可操作的反事实政策效应。

## 2. 为什么需要此预设 fallback

`state_provider` 已改为从 SIH `MUNIC_MOV` 的前两位推导的真实 provider UF。某些正确 UF 或其他分类协变量可能出现零死亡或极稀疏死亡，导致常规最大似然（MLE）logistic 模型完全或准完全分离。普通 MLE 的有限表面收敛、Wald P 值或有利的 volume 结果均不能否定分离。本修正案只处理这种已知数值与估计问题，不基于方向、显著性或拟合优劣选择模型。

## 3. 固定的模型路径与触发规则

### 3.1 先行的 MLE 诊断（仅诊断，不作为选择性结果报告）

对更正后的、冻结 Aim 4 分析表，以完整预设设计矩阵拟合常规 binomial-logit MLE；模型须保留 provider-UF 和 calendar-month fixed effects。执行并保存以下诊断：

- `detectseparation`（或版本锁定的等价线性规划分离检测）对完整设计矩阵的 complete/quasi-complete separation 判定；
- 迭代是否在 120 次内收敛、任何系数/标准误/拟合风险是否非有限；
- 是否出现 GLM/IRLS 收敛警告，或概率被数值截断到 0/1 的分离警告；
- 设计矩阵秩、condition number、最小单元格与每参数事件数；
- 对每个分类水平报告样本数与死亡数，仅以审计和隐私抑制后的汇总形式发布。

以下任一条件触发 fallback，且不允许根据 volume 效应的方向、P 值或置信区间取消 fallback：

1. separation detector 返回 complete 或 quasi-complete separation；
2. MLE 不收敛、产生非有限系数/协方差/预测值，或出现已记录的分离数值警告；
3. 在相同预设设计下，任一 MLE 系数绝对值大于 15，或任一标准误大于 10，作为近边界数值不稳定的预设保护阈值；
4. 完整设计矩阵秩亏。

秩亏不由惩罚法“修复”：因为该参数在保留固定效应和协变量的既定 estimand 下不可识别。此时不得删除州或月份固定效应、合并类别、改变调整集或改用结果驱动的简化模型；Aim 4 直接进入第 7 节的降级路径。

若上述条件均不满足，常规 MLE 与 hospital-clustered robust covariance 为主要 Aim 4 估计器，既有模型规格不变。若任一条件满足，MLE 仅保留为失败诊断，正式主要估计器切换到第 3.2 节，不报告 MLE 的 volume 效应量作为竞争性“更有利”结果。

### 3.2 分离时的主要估计器

分离 fallback 固定为 R `brglm2` 的 mean bias-reduced binomial-logit：`glm(..., family = binomial(link = "logit"), method = "brglmFit", type = "AS_mean")`。它使用同一批分析行、同一显式设计矩阵、同一 spline knots、同一协变量、provider-UF 与 calendar-month fixed effects；不使用 stepwise 选择、层级随机效应替换或结果驱动合并。

术语必须准确：`AS_mean` 是 mean bias reduction，不称为普通 MLE，也不称为未限定的 Firth 估计。作为预设**方法敏感性**，仅在主要 fallback 成功后，以同一设计运行 `brglm2` 的 Jeffreys-prior/Firth-style penalized likelihood（`type = "MPL_Jeffreys"`）。该敏感性不改变主要效应定义；与 `AS_mean` 的方向、量级和 95% CI 重叠情况必须完整报告。不能以任一版本的较小 P 值选定主结果。

对 `AS_mean` 的主拟合，以原冻结分析总体（而非仅 bootstrap sample）的经验协变量分布进行 g-computation 标准化。volume P10、P90 与四个 spline knots 都由原冻结分析表一次性计算、写入 manifest 后固定；每个 bootstrap replicate 不重新估计这些量。输出固定为 P10/P90 的标准化风险、RD 和 RR。条件 OR 可作为附表支持性结果，但不用于主张或 RD/RR 的不确定性推断。

## 4. 医院簇 bootstrap：不确定性与可重复性

由于 penalized logistic 下常规 hospital-cluster sandwich 方差不作为主要 CI，95% CI 预先固定为**分层医院簇 percentile bootstrap**：

1. 主随机种子固定为 `20260830`，使用记录在环境锁中的 R 随机数生成器和并行 RNG（`L'Ecuyer-CMRG`）。开始前保存 `.Random.seed`、R 版本、`brglm2` 版本、脚本 SHA-256、输入表 SHA-256 和设计矩阵/knots manifest。
2. 以 corrected `state_provider` 分层，在每个 provider UF 内对医院 `cnes7` 有放回抽样，抽取数等于该州原始医院数。被抽中的医院带入其全部符合主要总体定义的 AIH；重复抽中时赋予独立的 bootstrap pseudo-cluster ID。此设计保留州 fixed-effect 支持，同时在医院层面重抽样，不把同一医院的患者误当作独立抽样单位。
3. 固定执行 2,000 个编号为 1–2,000 的 replicate，不根据中途结果停止、扩充或更换种子。每个 replicate 按第 3.2 节拟合 `AS_mean` 模型，随后在原冻结总体上用固定 P10/P90 与 knots 标准化，保存 RD、RR、收敛状态、失败原因和 runtime。不得删除、winsorize 或以更有利的设置替换极端但有效的 replicate。
4. 95% CI 固定为有效 replicate 的 2.5th 和 97.5th percentile；不使用 BCa。选择 percentile CI 的理由是预先避免 rare-event penalized estimator 下 jackknife acceleration 与有限 cluster 删除的额外不稳定性。Monte Carlo 误差（至少报告 2.5th/97.5th percentile 的 binomial-order-statistic 近似 SE）必须列入 supplement。
5. 最大并行工作进程为 8，每个进程强制 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1` 和任何 R/BLAS 等价线程变量为 1；不得嵌套并行。replicate 输出按固定编号原子写入，主进程按编号排序汇总，以保证可恢复和可审计。

## 5. 有效重复、收敛和失败门槛

每个 replicate 只有在以下全部成立时才是有效 replicate：设计矩阵满秩；`AS_mean` 明确收敛；所有估计/预测/标准化风险/RD/RR 有限；风险在闭区间 [0,1]；且固定 P10/P90 contrast 可计算。失败必须记录为下列互斥原因之一：`rank_deficient`、`nonconvergence`、`nonfinite_estimate`、`invalid_prediction`、`contrast_not_estimable` 或 `runtime_error`。

- **通过：** 至少 1,900/2,000（95%）有效 replicate，且没有系统性单州、单月份或单类医院的失败模式。主文可报告 `AS_mean` 点估计与 percentile 95% CI，仍只使用 associational/supportive 语言。
- **警告但可报告：** 1,800–1,899 有效 replicate（90.0%–94.9%），仅当失败审计表显示无集中于单一州/月份/医院类型的结构性模式时，可在 supplement 报告；Aim 4 必须在主文中明确标为 exploratory/supportive，不能用于标题、摘要或核心结论。
- **降级：** 少于 1,800 有效 replicate、任何系统性失败模式、主点估计不收敛、原始设计秩亏，或 `AS_mean` 与预设 Jeffreys sensitivity 在方向上相反且二者 CI 均提示实质性不确定性。此时不报告精确主 RD/RR 95% CI 作为稳定关联；Aim 4 从主文结果移至“不可稳健估计/探索性补充材料”，Aim 1–3 不受影响。

失败后只允许修复显式的软件或 I/O bug，并在修复记录中说明该 bug 与任何效应数值无关；修复后必须从 replicate 1 使用相同种子和同一 frozen input 全量重跑。不得因有效重复数不足而更换模型、放宽协变量/固定效应、增大并行数、增大/缩小 bootstrap 次数或改用另一种 CI。

## 6. 必须保存的诊断、敏感性与报告项目

（预设）正式运行必须生成 machine-readable QC 与中文审计摘要，至少包括：

- MLE trigger audit、separation detector 原始判定、设计秩与 condition number；
- `AS_mean`、Jeffreys/Firth-style sensitivity 及（若无 trigger）MLE 的模型状态。任何 MLE 结果仅以诊断或预设主要路径呈现，不得选择性挑选；
- 原冻结总体的 N、死亡事件数、医院数、每参数事件数，以及每个分类水平隐私抑制后的稀疏性审计；
- 固定 P10/P90、knots、完整协变量清单、显式参考水平和实际可用性/缺失规则；
- 2,000 个 bootstrap replicate 的状态表、有效数、失败原因、各州/月份/医院类型失败分布、运行时间和随机数/环境 manifest；
- RD、RR、标准化风险以及 95% percentile CI；若有条件 OR，明确标为 conditional；
- spline 的联合关联检验仅作支持性描述，不能以单一 P 值替代效应量与 CI；
- 原 SAP 规定的 COVID 期间处理、context coverage、缺失处理和次级结局/次级暴露敏感性，均不得因分离 fallback 而取消或重定义。

结果叙述示例仅限于："在预设调整后，较高与较低医院 ERCP volume 的标准化院内死亡风险差呈现[方向]关联（associational）；该观察性比较可能仍受未测量 case-mix、转诊选择和测量误差影响。" 禁止使用 "导致"、"降低/提高风险"、"因果效应"、"最佳医院" 或对个体患者的预测性语言。

## 7. 预设降级与阶段 7 验收规则

Aim 4 的通过不影响论文最小核心（Aim 1–3）。若任何第 5 节降级条件成立，或更正后的 provider-UF 语义/AIH linkage/analysis-row provenance 未能通过 QC，主论文保留 Aim 1–3 的服务扩散、公平与网络叙事；Aim 4 只在补充材料中说明其不稳定或不可估计性，不以缺失或失败模型推断临床结局。

本修正案必须与更新后的 `config/estimands.yaml`、`reports/SAP_v1.md`、Aim 4 代码、环境锁、输入 manifest 和 Stage 7 result registry 一并版本化。任何改变本修正案所列 estimand、触发阈值、估计器、bootstrap 设计、CI 类型、种子、重复数或降级门槛的后续修改，都必须另建 dated amendment，说明是否已经查看任何正确 Aim 4 效应结果，并使旧 QA 失效后重跑受影响的全链路。

**（预设）冻结确认：** 本文仅制定方法和失败规则；未记录、推断或嵌入任何更正后 Aim 4 volume–outcome 效应数值。
