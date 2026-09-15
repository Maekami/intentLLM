# V4 Static：R20 Format 起点上的 Actor–Validator 实施与实验任务书

更新：2026-09-15。状态：V3 已归档，活动仓库已核验为 R20 Format，V4 启动快照已准备；新架构尚未实现，真实实验启动数为 0。Evo 暂停。

**本文件是后续 working agent 的主要执行任务书。任务是实现、测试并评估静态系统；唯一可写范围为 `intentLLM/runs_batch_qwen_full/gp_v4/static/`。真实运行总上限 30 次完整 target_32 batch，允许用 `OPENROUTER_API_KEY_1` 和 `OPENROUTER_API_KEY_2` 同时运行两批。** 失败、取消和工程重启均计入 30 次，不要求花完。

本任务落实 [review_v4.md](review_v4.md) 的双 Actor、完整历史、精简 Validator 输出、原因归因、独立重试与最后候选交付方案；以本文件明确的工程边界、实验矩阵和并发规则执行。旧 C1/C2/C3 提示词路线、A2 必跑、九次独立确认和单 batch 互斥安排已被替换。历史计划没有续跑额度。

## 1. 执行入口、起点与工作边界

### 1.1 先核验起点，再初始化工作副本

仓库根目录为 `/home/jiaying/zihang/intent/intentLLM`；`V4_WS` 指其下 `runs_batch_qwen_full/gp_v4/static`。先读本任务书与 [CURRENT_STATE](../CURRENT_STATE.md)，核对 [启动清单](../runs_batch_qwen_full/gp_v4/static/bootstrap/START_STATE.json) 和 [R20 Format 核验](../runs_batch_qwen_full/gp_v4/static/bootstrap/baseline_verification.json)，再按阶段推进。

本次重算四子项目 `src/**/*.py`、`configs/**/*.yaml/yml`、`pyproject.toml` 共 **187 文件**，与 V3 原启动清单、v2.6 R20 Format 快照均逐字节一致：

`4ea8194887c8622b35f994075234e54cf1d64565de4a9eb82d0f5caaa45bb6a8`

算法为按仓库相对 POSIX 路径排序，依次将“路径 UTF-8＋NUL＋原始文件字节”送入 SHA256。不能用 Git HEAD 替代当前工作树；四份 Format prompt 是已核验起点的一部分。**当前就是 R20 Format，因此本次未回退或覆盖运行源码。**

`V4_WS/bootstrap/r20_format/` 是本次从当前活动仓库复制、逐文件核验的启动材料，包含四子项目运行文件、测试、四个通用脚本和 target_32。它是当前仓库的冻结副本，不是任何 V3 候选。working agent 将其独立复制到 `V4_WS/repo/`，所有导入和执行均从自己的副本开始；不要直接在 bootstrap 中开发。版本快照另存 `V4_WS/versions/<arm>/repo/`，每次真实 batch 从对应冻结版本启动。

初始化时重核主目录、bootstrap 和初始副本的 187 文件锚点及数据 hash。若活动目录后来漂移，先定位并记录；本轮唯一 A0 仍由已核验 bootstrap 定义，不能无声吸收他人的新改动。bootstrap 损坏时只可从匹配清单的活动文件重建到工作区，不能把不匹配版本当 R20。

187 文件只定义初始基线。后续每版 manifest 必须覆盖该版实际使用的全部源码、新增角色、prompt/profile/schema、resolved 参数和数据，不能沿用初始路径集合而漏记新增文件；冻结后三次之间保持一致。

### 1.2 working agent 的全部写入必须位于 V4_WS

```text
runs_batch_qwen_full/gp_v4/static/
  bootstrap/                 本次交接的 R20 Format 输入与核验；工作期间保持冻结
  repo/                      实现和无生成测试副本
  versions/<arm>/repo/       每个实际参测版本的完整独立冻结运行副本
  runs/<arm>/seed_*/launch_*/ 每次启动唯一输出目录
  artifacts/                 协议、来源、版本/输入清单、账本、统计与审阅
  scripts/                   本任务调度、复算、边界检查脚本
  tmp/ cache/                临时文件和所有工具/模型缓存
  STATUS.md FINAL_REPORT.md  恢复入口及最终交付
```

允许只读主项目与历史档案以查证；**任何新增、修改、删除、缓存、测试临时文件、下载及日志都必须落在 V4_WS**。禁止回写主源码、根 CURRENT_STATE、本任务书、review、主数据或 v1–v3 历史；禁止写其他 batch 目录、用户 HOME、共享虚拟环境或 `/tmp`。本次归档与交接文档更新由当前整理任务完成，不是授予 working agent 的例外权限。

运行前设置任务内 TMPDIR/TEMP/TMP、XDG_CACHE_HOME、HF_HOME/HF_HUB_CACHE、TRANSFORMERS_CACHE、TOKENIZERS 缓存（如使用）、PYTEST 临时目录及工具实际使用的其他写入路径；不重定义 HOME/CODEX_HOME。启用 `PYTHONDONTWRITEBYTECODE=1`，pytest 禁用外部缓存并指定任务内 `--basetemp`。需要额外 Python 依赖时只在任务内建环境/缓存，不能 `pip install -e` 到共享环境。对实际导入 `__file__` 和所有输出路径做 realpath 边界检查；不用指向主项目的可写软链接或硬链接。

不复制 `.git`、`.env`、带凭据的 run.sh/run2.sh/run3.sh、历史 runs 或模型权重；不 reset/clean/checkout 主工作树、不提交推送、不修改或重启共享模型服务、不操作他人作业。不创建跨项目全局锁；本任务自己的调度锁和账本在 V4_WS 内。

### 1.3 修改许可与历史参考

这次**允许在副本内实现新 GP 架构**，不再受 V3“只能改 prompt”的限制：可改 GP runtime/schema/context/assembly/renderer/session、必要的 `assistant/config.py`、factory/角色注册、GP prompt/profile 及相应测试；可在任务 scripts 中实现调度、报告和离线审阅工具。保留 `v2_contracts/full` 的旧行为用于 A0/P；新架构使用独立配置入口（建议 `architecture_version=v4_actor_validator`），不能让 A0 静默走新链路。

`interaction_pipeline`、`user_simulator`、`metrics` 的源码及 Base 行为、公共 LLM 客户端保持冻结。任务 simulator 配置仅覆盖 dataset_path 到本版本的 target_32；不改 DAG、满意度/终止/评分规则。新角色的必要适配应在 assistant 的 GP 边界完成，不把局部改造扩散为全项目重构。生成参数按第 6 节固定；推进参数按矩阵单独比较。

历史只读入口：[v2.6 计划](../v2_6/assistant/plan_v2_6.md)、[v2.6 结果](../v2_6/runs_batch_qwen_full/gp_v2_6/RESULTS.md)、[V3 归档](../v3/README.md)、[V3 核验与分析](../v3/assistant/v3_results_analysis.md)、[四次 R20 全量](../v2_6/runs_batch_qwen_full/r20_format_full292/RESULTS.md)。V3 Static 的 A1/A2/A3 均未通过综合接受条件，A2 的共同成功路径信号保留为背景；这轮不再移植 A2 或增加一个 A2 实验臂。Evo 不采集、不构建、不运行、不移植技能。


## 2. 已制作的数据、选择规则与目标

### 2.1 固定样本

文件：[target_32.jsonl](../user_simulator/dataset/target_32.jsonl)。来源为 [DAG_fixed.jsonl](../user_simulator/dataset/DAG_fixed.jsonl)，与四次全量冻结的数据逐字节一致；每条 JSONL 保留完整原始行，不改图、节点、元数据或答案。

**用户已确认筛选口径：排除出现普通错误的题，保留超限。从四次均为 SUCCESS 或 FAILURE_TURN_LIMIT 的 275 题中，按四次实际助手回合数的样本方差取前 32。** 这是合格集合内的前 32，不是全部 292 题的无条件前 32；实验中不得重新筛选或替换样本。

这样处理是因为普通错误可能在第 1 轮中断，短错误轨迹不是高推进效率。若不排除，前 32 中有 6 题不同，含一题四次从未成功；其最佳 E/S=5.09375/6.37500，也不是本文件的目标。三种口径的完整敏感性结果保存在 [target_statistics.json](proposal_artifacts/v4_static/target_statistics.json)，没有悄悄删除异常后沿用旧排名。

选择公式与顺序固定如下：

```text
T_i,r = repeat r 的 batch_summary.runs 中记录的实际助手回合数
v_i   = Σ(T_i,r − mean_r T_i,r)² / 3，r=1,2,3,4
排名  = 方差降序；相同方差按 DAG_fixed 原始位置升序
取样  = 合格集合的前 32；写入 JSONL 时保留所选题的原始相对顺序
```

超限回合数为实际的 **20**，不能换成 S 的未达成标记 21；不用 E/S 方差、极差或最大 gap 替换回合数方差。ddof=0 与 ddof=1 在均有四条记录时排名相同，本任务统一报告 ddof=1。边界第 32 题方差为 8.0000。

- 源数据 SHA256：`44accd8edfadd290b223018e50181f953ec72422616903b603d9848e62cc4770`。
- target_32 SHA256：`9ef1009c6f2f2f19c3929dc4f32b0282dd40624f39114e21c47ddc8c5cdf45b7`。
- [筛选清单](proposal_artifacts/v4_static/selection_manifest.json)：32 个 ID、文件位置、原位置、方差、四条原始路径/哈希及每题最佳；[完整排名](proposal_artifacts/v4_static/variance_ranking.tsv)：全部 292 题及排除标记。
- 已通过现有 DatasetLoader/Validator 及 430 个前缀/END 离线检查；32 条源行字节一致；128 条入选轨迹的 E/S 已用 `TraceLoader` 与 `compute_dag_turn_metrics` 从原始记录重算一致。[独立复核](proposal_artifacts/v4_static/independent_validation.json) 另核验了排名、整轨迹择优及全部 128 份 events 哈希。

选择逻辑参考 [build_target_32.py](proposal_artifacts/v4_static/build_target_32.py) 与 [筛选说明](proposal_artifacts/v4_static/README.md)。这些是已完成的制作证据；working agent 不直接运行会回写主数据的原脚本。需要复算时复制到 V4_WS 并显式重定位输出，V3 专用样本包现位于 v3/user_simulator/dataset/self-evo-test/。

这 32 题历史上共 **104 成功、24 超限、0 普通错误**；22 题至少一次超限，仅 10 题四次都成功。故它主要是高波动与长尾压力集，不能沿用“这轮主要研究普通成功路径”的叙事。历史 first16 与它重叠 5 题，Evo test16 重叠 3 题，清单已列出；它是明确用于开发的集合，不是独立盲测。

### 2.2 已计算的观察目标

每题按 `(S,E,repeat)` 升序选**同一条完整有效轨迹**，再对 32 题等权平均。不能把某次的 E 与另一次的 S 拼接，也不能把不同回合拼成一条新交互。本集合择优后 32 题均成功，所选轨迹的 E 合计 135、S 合计 177。

| 历史口径 | N | E ↓ | S ↓ | 成功／超限 |
| --- | --- | --- | --- | --- |
| repeat 1，seed 42 | 32 | 7.56250 | 11.21875 | 24／8 |
| repeat 2，seed 142 | 32 | 7.87500 | 10.40625 | 26／6 |
| repeat 3，seed 242 | 32 | 6.50000 | 9.03125 | 29／3 |
| repeat 4，seed 342 | 32 | 7.71875 | 11.00000 | 25／7 |
| 四次批均值 mean±sample STD | 各 32 | **7.41406±0.62259** | **10.41406±0.98371** | 合计 104／24 |
| 逐题最佳完整轨迹：观察目标 | 32 | **4.21875** | **5.53125** | 32／0 |

目标相对历史平均的距离为 ΔE=3.1953125、ΔS=4.8828125。为便于跟踪，历史 gap 收回一半对应 **E=5.81640625、S=7.97265625**；它只是中间观察点，不是额外硬门槛或保证可达到的收益。四次分别排除超限后，成功条件均值为 E=5.86408、S=7.97824，不能与全 32 题目标直接相减作同集合收益。

**V4 核心目标：在新交互中使 E/S 稳定向 4.21875/5.53125 靠近，同时减少超限，且不靠新增普通错误或虚构完成压低均值。** 最终必须同时报告“是否优于同期 A0/Base”和“距离历史观察目标多远”。没有达到历史最佳不等于毫无改进；达到了某个历史数值也不自动证明稳定超过 Base。

高方差选样存在均值回归，重新跑 A0 自身也可能变好。子集位置变化还会改变实际 sample seed。因此旧四次结果只提供诊断和目标，**不替代 V4 的同期 A0/Base，也不充当第五次重复**。最佳值是四条已观测联合路径的事后选择，既不是理论下界，也不是每种用户路径都能实现的承诺。

## 3. 实施目标与机制归因

主问题是：**目标和已知约束是否正确保留，行动与完整产物是否落实这些要求，本轮回复是否有合理的推进价值。** 合法 JSON、更多 Goal、更多文字、一次模型自称完成均不是目标推进真值。

主要证据见 [review 的归因部分](review_v4.md)、[诊断证据](proposal_artifacts/v4_static/diagnostic_evidence.json) 与 [全量案例索引](../v3/assistant/proposal_artifacts/v3_results_analysis/full_r20_case_index.json)：有 Tracker 将已知范围维护错误的情况，也有原文/target 已正确传递而正文仍不满足限制的情况；重复输出有新模型请求，不能直接认定为跨轮缓存错误。另有材料确实未到达、用户对已改正文继续投诉等情况，不能统一归为助手失误。

实现前完成 32 题各一条历史最佳和一条最长路径的可见历史审阅，共 64 条路径；与机制有关的位置再查角色原始输入/输出。现有结构化统计覆盖 128 条轨迹、1309 回合，**不替代这项尚未完整完成的语义审阅**。在任务内记录原文位置、当时已知信息、最早可证实偏差及反例，不把后来才出现的信息回填成早已知道。

本轮依次识别：P 的推进参数作用，M 的角色合并和组装作用，V0/V2 的判断、状态提交及反馈修复作用，F1–F3 的 prompt 优化作用。没有旧 P4 的独立真实实验臂；不能宣称新架构已经在 E/S 上胜过 P4。

## 4. 首版架构与最小输出协议

### 4.1 三阶段主路径

```text
阶段 1：Tracker
阶段 2：Intra Actor 与 Inter Actor 并行 → 系统组装完整候选
阶段 3：一次联合模型 Validator → 系统按 ②→④→③ 裁决
通过则交付；失败且有校验额度则按原因重执行；额度耗尽则交付最后完整候选。
```

Tracker 维护 current/adjacent goals、约束、Need、证据和依赖；不代写答案，不预先标记尚未发送的产物已满足用户。Intra Actor 选择当前 advance/revise/clarify/真实阻塞并生成正文或问题；Inter Actor 在合法相邻候选中选择 none/elicit/anticipate 并生成自己的部分。两位 Actor 互补组成**一份**回复，不是两份全文择优。

首版取消独立 Planner→Generator 的内容转述，但保留动作、目标与来源结构。不得再加模型统稿器、Observer、历史摘要器或把两个 Actor 实际串行。M 消融关闭 Validator，正常路径为两个模型阶段、三次请求；V0/V2 正常为三个阶段、四次请求。将来多个并行 Validator 是备选，本轮不实施；②/④/③ 逐个串行调用不符合本方案。

一个不可拆成品由一个 Actor 完整生成。两位 Actor 看同一 Tracker 版本及共同适用约束，Inter 不能依赖 Intra 本轮尚未生成的酒店/方案等选择；首版延后这类依赖性交付。提问从独立字段经当前优先、去重和整轮预算筛选，未选问题不得留在正文；丢弃区块不得留下悬空引用。代码渲染后得到用户实际将看到的完整候选，Validator 审查该候选及其区块来源，不能只逐段验收。

保留正式实体身份、请求资格、基础 schema/引用/依赖检查和原样问题渲染。新实体继续由模型提局部身份、系统分配正式 ID，已有实体复用真实引用；不自动清洗非法 ID，不让换目标重置已问次数。结构合法不能替代语义检查。

### 4.2 ④的输入是完整可见历史

首版固定 `validator_history_mode=full_visible`。最近两轮不足以覆盖早期预算、限制变更/撤销、曾经给出的材料、跨目标回指及多轮循环。输入至少分为：

| 分区 | 内容 |
| --- | --- |
| HISTORY | 按时间保留 U1,A1,…,U(t−1),A(t−1) 的完整可见原文，包括上一轮完整用户消息和助手回复 |
| LATEST_USER | 本轮 Ut 原文，不在 HISTORY 重复复制 |
| STATE | 稳定前态、本次 Tracker 提案、实际已问事实与证据引用 |
| ACTOR_ACTIONS | 两位 Actor 的行动、依赖、问题与正文引用 |
| CANDIDATE | 最终完整组装回复、实际展示问题及系统添加的来源/区块索引 |
| REPAIR_FEEDBACK | 本回合此前的具体失败理由及必要失败草稿，单独标为未交付 |

相关原文尽量只放一次，再用系统引用关联；上一轮通过兜底发送的回复也是实际 HISTORY。内部失败草稿不冒充已发生的对话，审计事件不整包塞进输入。隐藏 DAG、队列、满足标签/理由、终局 E/S、任务答案不进入 Tracker/Actor/Validator。

调用前检查 prompt、完整历史、材料、候选、修复区和输出预留的总容量；为 Validator 注册明确的角色预算及扩展上限，总容量不超过 262144。先按既有上下文恢复机制扩展允许预算，技术恢复独立计账；仍放不下时记录上下文故障，不静默截为两轮、不新开摘要模型。若以后需要选择性历史，属于另行登记的机制变化，不能混在 F 的 prompt 改版中。

④比较真实已交付历史与当前义务，不仅比较当前稿与上一份内部失败稿。首次回应、用户换任务、必要澄清、确实缺少材料或用户要求原样重发可合理通过，reason 说明依据；通过不必然代表正的实际增益。文字变长、道歉、承诺、改 ID 均不构成推进。还没有用户对候选的实际反馈，因此判定只是事前判断。

### 4.3 门控顺序与精简模型字段

| 顺序 | 检查 | 失败后的起点 |
| --- | --- | --- |
| ② | Tracker 是否忠实保留当前目标、约束、Need、已知材料及状态 | Tracker |
| ④ | 候选是否对当前义务有有效推进，或属于有依据的合理例外 | 按 cause：tracker_state→Tracker，actor_behavior→两位 Actors |
| ①，首版关闭 | 清晰计算定义的内容约束 | 将来启用时归 Actors；本轮不实现 |
| ③ | 行动、内容及组装整体是否符合原文限制、是否一致 | 两位 Actors |

首版联合调用输出 `checks` 数组，按位置对应 `[2,4,3]`，到第一个 No 就停止；全部通过需三项。字段只有：每项 `pass` 布尔值和简短 `reason`；失败项增加 `refs`；**仅④失败增加 `cause=tracker_state|actor_behavior`**。失败理由指出具体问题及修复方向，引用选择系统输入中已有的少量位置。

```json
{
  "checks": [
    {"pass": true, "reason": "目标和当前有效限制已正确保留。"},
    {
      "pass": false,
      "reason": "用户仍要求缩短正文，候选却原样重复上轮内容，需要实际压缩。",
      "refs": ["history:m11", "history:m20", "candidate:intra:0"],
      "cause": "actor_behavior"
    }
  ]
}
```

引用为示例，实际只使用本候选输入提供的引用域。系统验证合法前缀、条件字段和版本一致性；无效 JSON、缺项、非法 cause/ref 为技术/协议错误，不伪装为内容门控 No。不能靠 reason 关键词猜 cause，也不要求模型重复给 owner、gate_id、restart_from、overall_pass、置信度或评分。

②已判通过、④再指出 Tracker 状态错误时允许据具体新证据归 Tracker，并记录判定不一致供审计；不能为了流程整齐强迫归 Actor。真实材料未提供可能是合法阻塞；系统拥有却未送给裁判是上下文问题，不能硬归角色语义错误。

### 4.4 全链路字段职责

| 层 | 模型必须产生或选择 | 系统补齐 | 最终审计 |
| --- | --- | --- | --- |
| Tracker | 目标、约束、Need、状态与依据、该复用的实体引用 | 正式新 ID、版本、结构差异及可由既有关系导出的字段 | 原提案、规范化状态、证据及稳定/未确认标识 |
| Actor | goal_id、action、必要 dependencies/blocked_by、deliveries 的 body、可选 request 的 need_id/question_text | actor/source、unit_id、整轮预算筛选、排序、渲染 | 原响应、被选/丢弃区块及最终候选 |
| Validator | pass/reason；失败 refs；仅④失败 cause | gate_id、first_failed_gate、跳过项、路由、次数和交付分支 | 原响应、解析结果、失败原因、修复上下文 |
| 运行 | 不要求模型输出身份或统计 | episode/turn/candidate/attempt ID、prompt/profile/schema hash、token、延迟 | 完整输入引用/快照、覆盖范围、调用依赖、两类重试、最终交付、原 episode 结果 |

模型不再输出给 Generator 的冗余 target，不重复生成已由数组容器或关系确定的 scope/constraint_ids。Tracker 的事实识别仍属语义责任，不因为输出精简就删除必要信息。结构字段变化不等于语义增益；人工根因、误判/漏检、推进类别和离线 E/S 属于审计层，不增加为在线必填字段。大对象用引用与 hash 去重，保留可复原的实际输入。

## 5. 重试、归因、交付与状态

### 5.1 两套独立且有限的恢复账本

| 账本 | 触发 | 额度 |
| --- | --- | --- |
| validation_reexecutions | 合法内容门控 No 触发从 Tracker 或 Actors 重执行 | V2 每个用户回合共 2 次；V0 为 0；多缺陷一次重执行只计一次 |
| runtime_retries | 超时、服务故障、空响应、解析/schema/基础引用、上下文等 | 沿用每 turn/owner/error_code 默认最多 3 次额外恢复；新角色明确注册/映射，底层 GP retry.max_attempts=1 |

这两套账本独立，不因回滚或新的校验尝试清零。Validator JSON 错误先在技术额度内恢复，不占两次语义修复；合法正文违反语义限制也不能改称协议错来借技术额度重做。所有恢复仍受 turn timeout 和请求并发限制，不新增无限循环。

从 Tracker 重执行使用回合开始的稳定状态、同一 Ut 和 validator reason/refs，重做 Tracker→双 Actor→Validator；从 Actors 重执行保留当前 Tracker 版本，重做双 Actor→Validator。两位 Actor 首版一起重做，不让模型多输出一个细粒度 owner。修复上下文带具体原因、引用原文及必要失败草稿，不把裁判意见当新的用户事实。

令 nT/nA 为本回合两类校验重执行次数，`nT+nA≤2`：

| 校验主路径 | 模型串行阶段 | 联合 Validator 主路径请求数 |
| --- | --- | --- |
| 首次 | 3 | 4 |
| 两次 Actors | 7 | 10 |
| Tracker＋Actors | 8 | 11 |
| 两次 Tracker | 9 | 12 |

`D_validation=3+3*nT+2*nA≤9`。**9 只是校验机制主路径的最大串行长度；其他错误恢复可使实际路径超过 9。** 实际深度按调用依赖 DAG 计算，不能把并行分支重试直接相加。分别报告主路径、实际深度、总调用及两类恢复。校验内部重做不算新的用户交互回合。

### 5.2 耗尽后交付最后完整候选

```text
取得完整组装候选和合法 Validator 结果
通过 → 交付当前候选
未通过且 validation_reexecutions < 配置额度 → 按失败门/cause 重做并计数
未通过且额度用尽 → 交付最后一次完整组装回复，记录 fallback_last
```

最后一次仍做 Validator 并保留 No；不选更早的“最佳稿”，不追加第四次语义修复，不替换模板拒答，不在用户正文中附加内部错误标签。完整候选经过既定基础协议检查和 Renderer；不能交付 Actor 片段、异常信息或残缺 JSON。技术故障导致无法取得完整候选/合法判断时仍按独立技术故障规则记录，不能假装通过内容门控或已经执行语义兜底。

这里的 Validator 是有界修复机制，不保证所有判错内容均被阻止交付。正常或兜底都只向用户提交一次，原子保存实际正文、真正发送的问题及回执；废稿与未选问题不算已问或已交付。

②通过且后续没有 tracker_state 否决时可保存 Tracker 更新；②失败或④归 Tracker 时，保留上次稳定语义快照，把当前提案标为未确认审计证据。**实际已发送的正文、Need 问题及其身份仍登记**，下轮从真实历史重建，不能因语义回滚把已问的问题当成没问过。系统生成 `delivery_mode=normal|fallback_last`、`validation_status=passed|failed`、失败门、状态/候选版本和账本；交付、裁判通过、用户满足三者分开。

V0 与 V2 使用同一状态提交规则。V0 没有内容修复机会，但失败状态的处理仍可能影响下一轮，因此它是“无修复校验”消融，不是纯记录型影子校验。M 不调用 Validator，使用自身的既有提交流程。

## 6. 参数、Prompt 与固定条件

### 6.1 推进参数的含义和本轮对照

| 参数 | 含义 | 本轮 |
| --- | --- | --- |
| request_budget | 整轮可选不同 Need 的问题数量，当前优先，再相邻 | A0/M/V0/V2 固定 1；P 仅将 A0 的该值改为 2 |
| adjacent_candidate_limit | Tracker/投影中的活跃相邻候选上限 | 固定 2；不是一轮必须交付两目标 |
| max_adjacent_deliveries | 同一个被选相邻目标的正文单元上限 | 固定 1；不是多个相邻 Goal 的数量 |

已有使用量审计见 [policy_utilization.json](proposal_artifacts/v4_review/policy_utilization.json)，既定上限下的使用量不代表放宽后的需求。其余推进参数可作为未来方向，但本轮不做笛卡尔网格搜索；未登记的新参数组合不能算 P 的重复。

### 6.2 生成与运行参数冻结

B 使用原 `prompt_base + qwen_3_6_27b_vllm_non_thinking`，A0 使用原 `goal_progression + qwen_3_6_27b_gp`，P 从 A0 只改上述策略参数。生成温度等不作搜索，也不为“统一”而改变 Base。

新架构在首次实验前冻结角色映射：Tracker 沿用原 Tracker；两位 Actor 都承担实际成品生成，使用原 Generator 的 **resolved generation 配置及正文输出预算**，不套用仅够 Planner 计划的短输出预算；Validator 采用原 Tracker 的 generation 配置，另登记足以容纳裁判输入的 context 预算。明确记录字段继承与角色映射，M/V0/V2 和 F 全部一致；不称新旧角色职责/预算完全相同。查看实际 resolver，不能仅凭 YAML 某一层推测最终参数。新增角色仍使用本地同一 non-thinking 模型，不自动引入更强模型。

| 项目 | 固定条件 |
| --- | --- |
| assistant / simulator | 本地 Qwen3.6-27B non-thinking / deepseek_v4_flash_0731 |
| 数据与交互 | target_32 原顺序；hard；20 回合；sample_retries=0；no-update-memory |
| 满意度/前缀 | monotonic_satisfaction=false；enforce_controller_prefix_closure=true |
| 每批并发 | episode 16；GP 每个进程最大在途请求 32 |
| 批次并发 | 最多 2，两个 key 各至多一批；GP 合计可达 64，不是共享 32 |
| 上下文/时间 | hard_context_tokens=262144；既有恢复规则与 turn timeout=900 秒 |
| 重复 | 每个冻结设置三次，seed=42/142/242；实际 sample seed=基础 seed＋子集零基位置 |

所有参数、原文输入模式、实际 prompt、schema、代码、数据和模块导入路径进入每版 manifest。并发耗时和服务时段可能影响延迟/故障，原样记录；不要为加速扩大单批并发或改变共享服务器设置。

### 6.3 结构化、可审阅的 Prompt

采用 YAML 多行字符串与 Markdown 章节：Role、Inputs and evidence、Decision procedure、Repair feedback、Output、Examples。Tracker 清楚区分事实与推测；Actor 直接落实行动与内容；Validator 清楚定义②→④→③、④例外和 cause 边界。系统注入真实字段 schema、有效预算及引用域，避免 prompt 里另写一套数字。

失败反馈简短且能定位修改。每版直接改相应章节、合并重复规则、删除过时条款，提供条款映射与通用正反例；不在末尾堆特殊补丁。不写 target_32 ID、原问句触发器、最快答案或隐藏意图，不从成功轨迹抽固定回复。

## 7. 30 次真实运行预算与迭代顺序

一次真实运行是**启动一个完整 32 题 batch**，最多 960 episode 槽。所有真实 assistant/simulator 生成都来自预登记 batch；不单题探路、不额外真实角色回放、不借离线测试启动模型、无付费学习或额外裁判审阅。解析、hash、fake tests 和人工审阅不占 batch 额度。第一次正式 batch 同时承担真实 smoke，若失败依然计数。

| 设置 | 定义 | seed | batch 上限 |
| --- | --- | --- | --- |
| B | 原 Prompted Base | 42/142/242 | 3 |
| A0 | 冻结 R20 Format | 42/142/242 | 3 |
| P | A0，仅 request_budget 1→2 | 42/142/242 | 3 |
| M | Tracker＋双 Actor＋代码组装，无 Validator | 42/142/242 | 3 |
| V0 | M＋联合 Validator，校验重执行额度 0，失败仍交付 | 42/142/242 | 3 |
| V2 | 同 V0，校验重执行额度 2，耗尽交付最后候选 | 42/142/242 | 3 |
| F1/F2/F3 | 最多三版结构化 prompt 优化，各版三次 | 42/142/242 | 9 |
| 工程预留 | 可证实配置/实现/运行故障的必要完整组 | 启动前登记 | 3 |
| **总计** | **18＋9＋3，不另加确认** | — | **30** |

先完成 B/A0/P 的同期对照；完成归因、实现与无生成检查后，冻结 M/V0/V2 的共同实现和首版 prompt，再运行三组各三次。V0 与 V2 仅校验重执行额度不同；V2−V0 识别反馈重做增量，V2−M 评估整个校验机制，M−A0 评估合并和组装。所有结果连同 Base 比较，不只选弱父版。

F1/F2/F3 使用原九次确认预算，按当前可见轨迹提出具体问题和父版，一版冻结后完成三次，再决定下一版；可改 Tracker、Actor 或 Validator 的结构化 prompt。父版的代码、角色映射、schema、历史模式、策略参数和恢复/交付规则保持不变；多角色联改记为改动包，不冒称单条规则的效果。没有依据可跳过，不强行用完。

本轮不再运行旧 A2/C1/C2/C3、不比较 U/R 归因路线、不叠加①/三个裁判、不额外运行 442/542/642 确认。普通超限、低分或校验误判不算工程故障；需要语义改版使用 F 槽。工程预留必须记录可复现缺陷、修改和受影响版本；行为变化另冻版本并完整三次，不能只补失败样本或拼不同版本。

每版一旦启动且没有明确运行完整性故障，应完成冻结的三次，不因第一批差就换版。组中断则如实标不完整，不突破 30、不把未完成组藏掉。未用额度不自动变成新机制/新样本搜索；执行完基础组及有依据的 F，或达到预算上限即交付。没有合格新方案可保留 A0。

## 8. 双 key 并发、账本与运行模板

### 8.1 两条进程独立凭据，共享任务内额度账本

用户已允许真实实验使用 `OPENROUTER_API_KEY_1` 和 `OPENROUTER_API_KEY_2` 并行启动两批；不再要求单 batch 互斥，也不需要为第二条队列另问许可。每个子进程只将自己选择的变量值映射为现有客户端读取的 `OPENROUTER_API_KEY`，并从子进程环境移除另外两项带编号变量。凭据仅从当前已注入环境获取，不读取历史 launcher 或他人进程环境；不打印值、不写 `.env`/配置/命令日志、不输出整个环境。

优先用 Python `subprocess.Popen(argv, env=child_env, cwd=frozen_repo, ...)`，用进程环境传值，避免把密钥放在命令行参数。manifest/账本只存 `key_slot=1|2`。只有一项凭据存在时可单队列继续；都缺失时仍完成实现与无生成检查，保留待运行状态，不寻找文件中的其他密钥。

必须由**一个调度器**维护 `artifacts/trial_ledger.jsonl` 与任务内锁，原子预留 launch_id、arm/version hash、seed、key_slot、唯一输出目录后再启动。临界区同时检查：已启动＋仍占用的预留≤30、各组剩余槽、总活动数≤2、每个 key_slot≤1；不能两个 worker 各自看“还剩一次”后都启动。预留尚未创建子进程且可证实未调用时才可取消释放；状态不明则按已用保守计账。任何实际进程启动，包括失败、取消、重启均计一次。

启动成功立即记录 PID、进程开始时间及输出目录；恢复时先核对存活进程身份/账本，不重复启动旧批，不仅凭 PID 数值认定归属。不要让两个 batch 写同一个目录、共用可变源码或覆盖同一汇总文件。冻结版本可只读共享，派生配置/运行输出分开；在其他版本实验进行时，仅编辑开发 repo，不能改已参测 versions。

每个阶段在首批前登记 seed 分块与组别交错队列；确定可执行队列前两项占两个 key 槽，空闲槽接续已登记的独立作业。用任务内 round-robin 在两槽间分配组别，避免一个方案总落在某一凭据/时段。父版分析或 F 修订有依赖时等待对应组完成再冻结，不让并发跳过版本选择顺序。记录两槽实际时段、重叠作业、退出码与故障；不保证并发恰好提速两倍。

### 8.2 子进程命令模板

下面是调度器预登记后的**单个子进程参数模板**，不是绕过账本的直接启动脚本。`V4_STATIC_REPO` 指该 arm 冻结副本；父调度器已在 child_env 设置该槽 `OPENROUTER_API_KEY`、任务内缓存/临时目录及 PYTHONPATH，不在命令中传密钥。

```bash
python -m interaction_pipeline.cli.batch \
  --all --limit 32 --baseline goal_progression \
  --assistant-model-profile "$V4_STATIC_PROFILE" \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --simulator-config "$V4_STATIC_REPO/user_simulator/configs/v4_static_target32.yaml" \
  --difficulty hard --max-turns 20 --seed "$V4_STATIC_SEED" \
  --concurrency 16 --sample-retries 0 --no-update-memory \
  --output-dir "$V4_STATIC_WS/runs/$V4_STATIC_ARM/seed_$V4_STATIC_SEED/launch_$V4_STATIC_LAUNCH"
```

B 改用 `--baseline prompt_base --assistant-model-profile qwen_3_6_27b_vllm_non_thinking`。每版 simulator 配置只覆盖 dataset_path 为该副本 target_32 绝对路径；从真实子进程环境移除 `REASON_DAG_DATASET_PATH`，防止覆盖配置。`--limit 32` 不代替核对 resolved 数据 hash、32 个 ID 与顺序。PYTHONPATH 必须全部指向对应冻结副本的四个 src；从该副本 cwd 启动，不使用归档执行器。

## 9. 实施里程碑与必要核验

| 阶段 | 工作与完成条件 |
| --- | --- |
| M0 起点与协议 | 初始化任务内 STATUS、源码/输入 manifest、protocol、双槽账本；核对 R20/数据/导入/输出边界，三项关键参数与原恢复规则；只检查 key 是否存在 |
| M1 归因与对照 | 完成 64 条历史快慢路径审阅，保留具体原文及反例；冻结并运行 B/A0/P 三次，基础对照用量共 9 |
| M2 架构实现 | 在 GP 副本实现 Tracker→并行 Actors→联合 Validator、最小 wire、完整历史、按原因恢复、独立账本与最后候选；保留 A0/P 旧分支 |
| M3 无生成验收与机制组 | 完成下列测试、冻结共同 M/V0/V2；各三次，共 9 次；记录审判准确性、修复和兜底，不把裁判判断当结果 |
| M4 Prompt 迭代 | 最多 F1–F3，每版三次；逐版分析再修改，每版提供 CHANGE_NOTE、父版、条款映射、正反例与实际 hash |
| M5 交付 | 全部启动与原始结果复核、配对分析、推荐实际完整测试过的版本；写 FINAL_REPORT/STATUS，停止，不向主项目合并 |

无生成检查使用 mock/fake 模型，至少覆盖：

1. bootstrap/副本锚点、target_32 原行/顺序、原 DatasetLoader/前缀/END 与指标口径；实际 imports、输出/缓存/临时目录没有越界。历史脚本若硬编码主目录，只在任务内复制适配，**不直接运行会重写主数据的 build_target_32.py**。
2. 两个 Actor 确实并行，共享同版状态；Inter 不读尚未完成的 Intra 输出；整轮问题预算、已问 Need、单元筛选与最终组装符合协议。
3. 早期约束原文在完整历史中可见，上一轮用户/助手均完整；候选、失败草稿和已交付回复不混淆；上下文溢出显式恢复/失败。
4. checks 合法前缀、首失败优先、④条件 cause/refs 与版本引用；②→④→③ 顺序正确，①确实关闭，语义 No 与协议故障分别记账。
5. Tracker/Actors 重执行范围与原因注入正确；技术重试不占校验次数，校验重做不清技术账本；两次校验上限及实际超过九阶段的技术恢复路径均有 fake 反例。
6. 耗尽交付最后完整候选且只发一次；日志仍为 failed；被否决 Tracker 提案不变权威状态，但真实已问与实际正文保留，下轮历史可读。
7. V0 零修复、V2 两次修复，其余同配置；M 无模型 Validator；旧 A0/P 与 Base 行为不受新入口影响。
8. 双槽并发预留竞态、总预算边界、每槽单 batch、重启识别、输出唯一、密钥不进 argv/日志；可用无模型短子进程演练。

先跑冻结副本原 assistant 测试及有关 GP 回归，再跑新增机制/调度/统计测试；不要扩大为无关重构。三次冻结期间不变更代码或 prompt。测试证明控制流和输入输出规则，不证明模型一定判对、完整历史必然有用或 E/S 已改善。

## 10. 评价、选优与完成标准

### 10.1 正式指标及配对

沿用原实现：SUCCESS＋FAILURE_TURN_LIMIT 计 E/S，普通错误为 N/A 并单列；未达成记 21，已经达成的 E 不因后来超限抹去。每批均值后汇报三次 mean±样本 STD；同 seed 共同有效 ID 配对，公布分母和逐题差。所有已启动/中止/失败组完整报告，不仅汇报裁判通过样本。

并列成功条件配对、固定双方六次均成功 ID、全部尝试的成功/超限/普通错误数，分列历史 22 个曾超限 ID 和 10 个四次成功 ID。这些分组只供离线分析，不驱动线上路由。低方差若来自最好路径退化、都超限或删除普通错误，不算稳定改善；另报逐题实际 T 方差、E/S 极差、最坏 S 和三次全成功率。

fallback_last 不自动记 SUCCESS、不排除于指标；正常交付也不等于目标满足。分别报告各门误否决/漏检的可复核样本、④例外、cause 路由准确性、修复新增缺陷、兜底率及下一轮结果。每条证据追到真实候选和历史，reason 不是正确性证明。

可见 token 为每 episode 用户实际看到的全部助手正文之和，再跨 episode 平均；不除以回合。内部候选、校验、两类重试和废稿成本另列，同时报全请求、调用依赖深度、延迟与并发时段；缺失 usage/金额不填零。

### 10.2 开发选优与结论边界

候选相对实际父版及 A0：三次平均配对 ΔE≤0、ΔS<0，至少两次 ΔS<0；成功不减少、超限/普通错误不增加；共同成功路径不退化，实际产物没有系统性虚构/越权。始终直接报告相对 B 的结果，V0 不是唯一接受门槛；V2/F 同时看 M、A0、B 和真实父版。只在正常成功路径有收益但整体失败更多的版本可作为后续 F 的分析对象，不能直接宣称稳定替换。

多个合格候选在固定共同有效集合上按平均 S、E、失败数排序，再考虑改动和代价；集合过小如实标证据不足，不换有利分母。最终 W 必须是一个实际完整三次测试过的冻结版本；不得临时拼装 P＋F 或其他未经测试的组合。没有合格版本保留 A0，任务执行完成不等于研究效果成功。

报告到历史目标 **E=4.21875、S=5.53125** 的距离。出现普通错误时同时按该版本有效 ID 重算对应历史目标并报告覆盖率，不能把较容易的剩余子集与全 32 题目标混比。达成全 32 题观察目标的表述需完整覆盖，并列超限情况。

九次已用于开发 prompt 迭代，**本轮没有独立确认阶段**。最终可称“开发集迭代后的三次重复表现”，不能称独立确认、全 292 稳定或未见题泛化；多次尝试选优和高方差选样的偏差都需说明。不额外补跑确认、不扩大样本、不恢复 Evo。

### 10.3 交付文件

主要交付 `V4_WS/FINAL_REPORT.md`：实际完成矩阵/启动数/两槽使用、全结果 E/S 和失败表、配对及目标距离、各改动的机制证据与代价、推荐冻结版本及局限。`STATUS.md` 保留阶段、活跃进程身份/输出、各组和总账本已用/预留/剩余，以及下一步或停止理由。

同时保存 protocol、bootstrap/版本/输入 manifest、trial_ledger、逐样本 TSV 与配对成员、真实 events/transcript/config_snapshot/final_state、模型最小输出及系统审计、token/调用/延迟表、测试结果、CHANGE_NOTE 和可在独立副本应用的相对 patch。核验最终版本实际请求中的 prompt/profile/hash 与冻结清单一致；审阅所有新增失败、同集合 S 改善/恶化最大的各五题和两题近似不变案例，数量不足则全部审阅。

完成后所有产物留在 V4_WS。不得更新根 CURRENT_STATE、主计划/源码或历史档案，也不自动移植推荐版本。下一轮自进化的共同底座和预算待静态结果审定后另定。
