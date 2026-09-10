# CURRENT_STATE — 项目现状与后续任务交接

更新：2026-09-10，v2 设计文档、历史 R1–R20 试运行及诊断材料已归档至 [v2/](v2/README.md)；原诊断清单见 [archive_manifest.tsv](v2/archive_manifest.tsv)，补充迁移清单见 [remaining_archive_manifest.json](v2/remaining_archive_manifest.json)。源码、测试、prompt 和模型配置保持原位，当前运行实现仍是 R20。

**最新待实施规格为 [plan_v2_5_updated.md](assistant/plan_v2_5_updated.md)**，原 [plan_v2_5.md](assistant/plan_v2_5.md) 仅作讨论稿。用户允许在 GP v2 源码和配置上直接覆盖；prompt 要保留有效信息并重组为清晰章节，不能把精炼理解成删掉关键边界；需保留 Runtime 分配 ID、角色动态引用域和独立 0–3 次恢复。试运行按单个设置固定版本运行 3 次再统计，具体协议见新计划。本轮只完成文档和忽略规则，未实现 v2.5 或启动新实验。

`v2/` 按用户最新要求**整体忽略、不做 Git 备份**，现用代码的 Git 备份由用户执行；实现前确认旧版可还原。P0–P6 离线诊断和最小六组在线矩阵已完成；下文冻结 R20 的说明及第 1–8 节是历史状态/事实，不是禁止按新规格原位覆盖，也不是待重新执行的任务。

## 0. 本阶段已完成：冻结 R20，离线诊断与六组在线矩阵

用户在本次会话明确批准最多 **20 次**真实 `python -m interaction_pipeline.cli.batch`，最低六次；已执行 **6/20 次**，其余 **14 次未使用**。这与此前 R1–R20 的历史 20 次额度分别记账。本轮不再追加运行，最小诊断验收完成；不把“尚未稳定优于 Base”当成未完成当前诊断任务。

先完成并冻结 [architecture_diagnosis.md](v2/architecture_diagnosis.md)，再执行在线矩阵；最终在线结果、成本、失败与建议见 [online_architecture_diagnosis.md](v2/online_architecture_diagnosis.md)。两份报告分别保留，归档仅调整路径；归档前原件及冻结 hash 的对应关系见 `v2/archive_manifest.json`。[v2/diagnostics/README.md](v2/diagnostics/README.md) 提供证据索引与不调用模型的复算命令。

### 0.1 交付与验证

| 阶段 | 已完成范围 | 主要证据 |
|---|---|---|
| P0 | 525 项输入清单；R20 源码/schema/prompt/六份 GP profile 冻结 | `v2/diagnostics/manifest.json`、`v2/diagnostics/validation.json` |
| P1 | simulator 实际执行顺序、历史 system 差异及当前配置核验 | `v2/architecture_diagnosis.md`、`v2/diagnostics/environment_audit.json` |
| P2–P3 | 16 样本、67 intent、156 边、60 条 skip；表达审计 36 可表达 / 0 已证明全局不可表达 / 24 不确定 | `v2/diagnostics/dag_profile.tsv`、`skip_edges.tsv`、`expressivity_audit.tsv` |
| P4 | 历史 77 episode、691 轮重建；58 条纳入 GP 指标与既有 TSV 一致 | `v2/diagnostics/turn_diagnostics.tsv`、`episode_diagnostics.json` |
| P5 | R20 全 16 条及其余 14 条历史超限，共 30 个审阅 case；覆盖历史全部 15 条超限 | `v2/diagnostics/earliest_divergence.tsv` |
| P6 | H1–H5、Q1–Q6、反证与最小判别实验 | `v2/architecture_diagnosis.md` |
| 在线 | 六组各 16 条，共 96 次 episode 尝试；90 完成、3 超限、3 组件失败 | `v2/diagnostics/online_round1/`、`v2/diagnostics/online_environment.json` |

新增分析脚本位于 `v2/assistant/scripts/`；19 项诊断回归已归档至 `v2/assistant/tests/integration/test_gp_diagnosis.py`，最新 assistant 全套结果为 **371 passed（15.72 秒）**。验证只覆盖 assistant 测试与分析一致性，不代表全仓库测试通过或模型稳定胜出。

### 0.2 当前在线结论与运行条件

| 设置 | 完成 / 超限 / 其他失败 | 正式 N | E / S | 同样本当前 Base E / S |
|---|---:|---:|---:|---:|
| base | 16 / 0 / 0 | 16 | 4.0625 / 5.1875 | 4.0625 / 5.1875 |
| full | 14 / 2 / 0 | 16 | 3.5000 / 7.3750 | 4.0625 / 5.1875 |
| no_inter | 16 / 0 / 0 | 16 | 5.3750 / 6.6250 | 4.0625 / 5.1875 |
| joint | 14 / 1 / 1 | 15 | 3.7333 / 5.8000 | 4.2000 / 5.3333 |
| no_intra | 16 / 0 / 0 | 16 | 4.3125 / 6.1875 | 4.0625 / 5.1875 |
| no_tracker | 14 / 0 / 2 | 14 | 4.0714 / 5.0714 | 4.2143 / 5.2857 |

正式 E/S 纳入完成及超限，其他失败排除并单列；共同有效集合为 13 条，详见在线报告。Full 相对当前 Base 是 E 改善、S 退化。no_tracker 在有效配对集合上 E/S 略好，但有 2 条组件失败、110 次恢复决策及 13 次 optional Inter 撤销，不能据此认定可靠优势。其余简化也有指标取舍。

**Recommendation: KEEP_V2。** 保持 R20 冻结，保留结构简化候选，优先核验 simulator 关键限定/用户材料披露，以及当前与相邻 Goal/Need 的归属和分区协调；现有证据不足以证明必须启动 v3，也不代表 Full 最优。各消融涉及职责迁移，不能作为纯单因素因果实验。

本轮共用 first_16（与 DAG.jsonl 前 16 条顺序一致）、hard、max_turns=20、seed=42–57、concurrency=16、sample_retries=0、no-update-memory。用户已将 `monotonic_satisfaction` 改回 **false**；96 份快照和实际 simulator system 模板一致，初始可见用户文本仍存在生成差异。R20 联合指纹保持 `ba2a3c02c2972adf09028104b45d66cbdbae72235226e3854982f175dbb3a4f1`，数据 SHA256 保持 `8177c7bae6955ada9f91e7201f296895464d6d7020ef67291f0eeb5d0442dd84`。

原始日志和已有分析产物保留。新批次位于 `v2/runs_batch_qwen_full/gp_v2/diagnosis_20260910/`，实际启动/结束记录见 `v2/diagnostics/online_trial_ledger.jsonl`，没有复用 try_1…try_20 或自动重跑失败样本。当前在线矩阵为 Base / Full / no_inter / joint / no_intra / no_tracker；冻结的六份 GP profile 另外包含 no_anticipate，但本轮没有运行该设置。

### 0.3 对 TODO 的必要校正（执行时以此避免误判）

1. **边 reason 是分析注释，不是当前 Controller 的逐边执行条件。** [GraphNavigator](user_simulator/src/user_simulator/graph/navigator.py) 用边的 source/target 建候选；[controller_context](user_simulator/src/user_simulator/controller/llm_controller.py) 传入候选 DagNode 和可见历史，没有传 `DagEdge.reason`。节点 `reason_text` 与边 reason 也不是同一字段。可做 reason 粗分类，但“可表达 X/60”只是标注审计，不能写成实际可触发率。
2. **Hard 一轮可有多个 gain，但受前缀和事件顺序约束。** [transitions.py](user_simulator/src/user_simulator/engine/transitions.py) 在较早候选 false 后阻止全部后续候选；有 N1→N_last 不意味着忽略前面条件直接跳到最后。Controller 新暴露的节点可在同轮接受 Satisfaction Updater 评估；但 [episode.py](user_simulator/src/user_simulator/engine/episode.py) 的 backbone 自动暴露发生在满足更新之后，要单独记 `exposure_source=backbone_auto`。Hard 最终只选未满足队列中节点序号最小的一项，以 abstract 模式生成用户消息；这不限制前面的多节点暴露/满足。初始 N1 记在 t0，END 不计 intent gain，END 已暴露后 Controller 会跳过。图论最短 hop 数不能直接当成 E/S 的最少回合数。
3. **当前 satisfaction 不是单调的。** 当前 YAML（用户在 P0 后改回 false）以及 R18/R19/R20、历史 Base 的快照均为 `monotonic_satisfaction=false`；不能依据 Episode 构造函数默认 true 推断实际行为。保留 TODO 的“首次满足 gain”，同时记录重新满足、满足回退和净变化；累计曾经满足的节点并集不等于某轮全部同时满足。No-progress 只表示没有新增首次暴露/首次满足，不能当成没有部分进展、没有回退或没有价值。
4. **一个 Need / 一个 delivery 不等于一个隐藏 intent。** `RequestProposal.need_id` 和每个 IntraItem 的 request 数量有结构约束；question_text/text 本身可含多个内容，但通过解析不表示符合“一项独立信息”的提示语义，多个真实 Need 偷装进一个问题还会漏记请求历史。同样，一个合法且连贯的交付可能自然覆盖多个隐藏节点。表达审计要区分硬结构限制、prompt 语义限制、台账无法表示和模型没有选择；先寻找合法表达或反例，不直接从 request_budget=1、max_adjacent_deliveries=1 推出 H1 成立。
5. **离线相关性不等于可归因的额外回合。** 历史 Base 环境不匹配时记录 `environment_match=false`，按实际配置/prompt 差异解释；同 sample_id 或同 turn 数不代表两条轨迹处于同一信息状态。先找各轨迹中可观测的最早偏差，缺乏证据允许 NO_CLEAR_ERROR；“42%”必须说明是哪些审计 case/独立问题的比例，不能伪称已证明 42% 额外 turns 由某组件造成。未登记某个隐藏 future intent 也不能直接算 Tracker 错误，须有当时可见上下文支持。
6. **指标需要防止误归属。** Skip Utilization 基于 controller_requested 的 frontier/candidates 和最终 prefix-normalized 决策，不凭节点编号跳变或 Inter=anticipate 猜测；首次暴露与自动暴露分别记录。一次回复的 gain 可能来自当前正文、相邻正文、问题、用户纠错触发，甚至满足评估对早期助手内容的重新认可，不能全部归给 Inter 或 request。请求数为 0 时 ratio 标 N/A，并同时报告分子、分母；Base 没有 GP 请求台账，不等于它没有提问，问题块数量须独立人工标注或记为未知。S=21 是失败标记，不能将 S−E 当作真实完成尾段，未满足样本需报告观察到的尾段/截尾状态。
7. **归因与日志完整性要分开。** 七种主类别沿用 TODO，但用 error_kind/error_owner 单列结构、运行时和 simulator 错误；不能把 runtime.io 等硬塞成语义判断失败。缺失/中断阶段标 unknown，不补成 gain=0。区分促成 assistant t 的输入用户选择（通常记录在 t−1）与 assistant t 后生成的下一条用户选择（记录在 t）。组件重试和旧契约不当成多个已交付回合，取成功交付对应的版本与回执。

## 1. 先掌握这几件事

- **当前运行的是 GP v2，最终保留版本为 R20。** v2 已直接覆盖原 GP 源码、prompt 和配置，baseline 名仍是 `goal_progression`，Full profile 仍是 `qwen_3_6_27b_gp`。不要另建平行 v2 工程，也不要恢复旧 v1 schema 或兜底机制。
- **已经完成 v2 的 20 次真实试运行，不是“尚未联调”。** 输出位于 `v2/runs_batch_qwen_full/gp_v2/try_1` 至 `try_20`，每次 16 条；因达到用户授权的次数上限停止，尚未达到 E/S 稳定超过 Prompted Base 的目标。此前额度已用完；诊断阶段另获 20 次上限并使用 6 次，具体进度以第 0 节和在线账本为准，不自动启动 try_21。
- **历史 R1–R20 汇总口径：只纳入正常完成和回合超限样本，排除其他失败。** 20 次共 284 条完成、15 条超限、21 条其他失败，即 299/320 条纳入。起点 test2 不计入 20 次。
- **最近一次 assistant 全套离线回归为 371 passed。** R20 结束时为 340 项，后续统计工具增加 12 项，本次诊断增加 19 项；不代表全仓库所有测试通过，也不代表模型语义或效果已达标。
- 本次新增离线分析脚本、测试、诊断产物及六批新日志；GP 运行时源码/schema/prompt/profile 未改。simulator YAML 的用户修改单列记录。运行成功只证明执行时服务可用。

### 推荐阅读顺序与文档时效

1. 本文件第 0 节、[离线报告](v2/architecture_diagnosis.md)、[在线报告](v2/online_architecture_diagnosis.md)：了解完成状态及证据；[todo.md](v2/todo.md) 保留为原任务说明。
2. [plan_v2_updated.md](v2/assistant/plan_v2_updated.md)：理解设计基础，但开头“尚未实现”和部分初始契约是旧状态；原讨论稿为 [plan_v2.md](v2/assistant/plan_v2.md)，不直接当作 R20 代码事实。
3. [v2 iteration_summary.md](v2/runs_batch_qwen_full/gp_v2/iteration_summary.md) 和 [implementation_v2.md](v2/assistant/implementation_v2.md)：核对 R1–R20 保留/撤回项、最新 E/S、实际接口与参数；实现事实以当前源码/config 和后续修订为准。
4. `user_simulator/dataset/first_16.jsonl`、simulator 当前代码/prompt 与各批次快照：按第 0 节入口核实机制，不从 TODO 的图示推断未观察到的能力。
5. [report_v2.md](v2/assistant/report_v2.md) 仅作首次实现的历史验收参考，其中“未真实联调”和 206 项 assistant 测试不是最新进度。最新用户要求优先，冲突需显式说明。

旧 `assistant/plan.md`、`implementation.md`、`report.md` 和 v1 GP 运行产物已移动至 [v1/](v1/README.md)，该目录在根 `.gitignore` 中。归档只迁移了非代码文件，后来原位代码被 v2 覆盖。`assistant/proposal_artifacts/` 是其他提案/原型材料，`assistant/result.md` 是历史 baseline 汇总，均不是当前 GP 实现或胜出证据。

## 2. 项目内容与研究目标

项目研究隐藏后续需求逐步显现的多轮交互。Simulator 持有 Reason-DAG 和评测状态；assistant 只能读取可见自然语言对话，不能读取隐藏节点、满足状态或评测答案。GP 的主线是 goal progression：完成当前目标，同时在有依据、有新增价值时推进相邻目标，减少暴露全部需求、满足全部需求所需的回合。

| 目录 | 职责 / 使用方式 |
|---|---|
| `assistant/` | 模型配置、baseline、普通/GP/记忆会话、GP 组件及测试。 |
| `user_simulator/` | 隐藏意图图、用户生成、交互难度及满足状态更新；GP 迭代不修改。 |
| `interaction_pipeline/` | 连接两端，提供 batch/demo/evo/trace2skill 等入口，落盘 episode 与 batch 产物；GP 迭代不修改。 |
| `metrics/` | 离线读取日志，计算 E/S、可见 token、可选 AITR；不向 GP 提供隐藏评测信号。 |
| `runs_batch_{qwen,gemini,luna}_full/` | 历史 baseline 原始结果；历史 GP v2 结果已移至 `v2/runs_batch_qwen_full/gp_v2/`。 |
| `v1/` | v1 文档和试验产物归档，不是可独立安装的运行包。 |
| `v2/` | v2 设计、R1–R20 与诊断归档；现用运行代码仍在 `assistant/`。整个目录被 Git 忽略，仅作本地历史参考。 |

四个子项目均采用 Python ≥3.11 和 `src` 布局。现有注册项包括 `base`、`prompt_base`、`interactcomp_react`、`trace2skill`、`goal_progression`；ExpRAG/ReMem 通过 memory profile 与专用 Session 接入，不是直接同名的普通 baseline。

主目标只要求超过已有 **Prompted Base**，不是超过所有 baseline。静态系统以推进效率为主，token 效率作为观察项，后续自进化再重点优化。临时偏好及恢复应自然作为 Tracker 的正常状态更新处理；人为扰动集属于后续讨论设置，当前没有为其增加专用分支或改造 simulator。

## 3. R20 当前架构与分工（诊断阶段冻结）

```text
可见历史、上一轮状态、已交付回执
                ↓
             Tracker
                ↓
       并行 Intra / Inter
                ↓
       Assembler（代码授权）
                ↓
            Generator
                ↓
 Renderer（代码拼接）→ Session 原子提交
```

Full 正常有 3 个串行模型阶段、4 次生成。无待生成正文时（例如只有已批准的成品问题）跳过 Generator，程序直接交付并记录 `assistant_generator_skipped`；重试会增加实际调用，但不增设常规 Reviewer/Checker。

| 组件 | 唯一职责 / 主要边界 |
|---|---|
| Tracker | 识别当前/相邻目标、约束、缺失信息 Need 和必要原文，维护语义身份；不选动作、不写答案、不让模型统计询问次数。 |
| Intra Planner | 对当前目标选择 `advance/revise/clarify`，提出交付与请求。只有真实缺失信息阻塞时才允许 `action=null`，不是额外通用动作。 |
| Inter Planner | 从已识别候选中至多选一个目标，决定 `none/elicit/anticipate`；不重新识别完整用户状态，不把当前缺失信息包装成相邻目标。 |
| Assembler | 确定性校验请求资格、当前优先、去重、预算和授权单元；不以关键词或另一次模型调用裁决自由文本语义。 |
| Generator | 根据获批契约与必要材料完成实质回答，输出 `units/issues` 薄 JSON；Full 不重新仲裁动作，不生成请求通道。正文完整性与事实质量仍依赖模型。 |
| Renderer / Runtime / Session | 原样追加冻结问题，记录文本位置/hash、ID/次数/版本/重试事实；成功返回才提交历史和状态，失败不伪造交付。 |

### 实测后已经确定的关键契约

- **一份关系只有一个权威来源。** Tracker 对外输出 `SemanticProjection.current_goals/adjacent_goals`，程序仅转成内部 `SemanticSnapshot.goals`；约束关系只由 `Constraint.applies_to` 表达，`Goal.constraint_ids` 与 `changed_goal_ids` 由代码派生，不要求模型重复填写。
- **Need 身份与当前适用目标分开。** 稳定 `ref` 保留询问历史；`goal_id` 可按证据更新到本轮相关目标；注册表 `origin_goal_id` 仅是不可变来源。不能换目标就清零次数，也不能按同名自动合并信息。早期“永久固定父目标”的实现已在 R10 修正。
- **已完成结果也能支持下一步。** 合法锚点为 active current 或 addressed 结果；暂停目标、未完成的相邻候选不能作为该类锚点。
- **上下文按职责选取。** Full 中只有 Tracker 默认读取完整可见历史；下游读取所属目标、相关约束/Need/原文及最新用户消息。旧助手回复仅在相关材料/证据被选中时传递，不自动广播上一条整段回复。必要原文不擅自截断。
- **需求与答案分开。** 状态描述尚需完成的结果，Planner 的 `target` 描述交付要求，Generator 写实际内容；不能把上游猜测或用户已经否定的答案固化为必须复述的指令。`question_text` 是直接展示的成品问题。
- **身份规则统一。** 旧实体复用已注册 g/c/n ID；新实体声明 `new:<local_name>`，运行时分配规范 ID。统一新 ID 正则为 `^new:[A-Za-z0-9_./-]+$`，避免原 `\S` 在 XGrammar 中触发非 ASCII 否定字符类警告；中文描述和正文不受限制。schema、本地校验、反馈和 prompt 保持一致，不自动清洗非法身份。
- **同一个错误协议负责恢复。** 校验失败交给字段的原 owner，反馈具体路径、允许引用及有界局部旧输出；上游改变只使相关下游失效。optional 相邻工作可撤销，required 路径耗尽则明确失败；没有 Prompted Base 隐式兜底。

用户的设计原则仍适用：统一机制而非按样本打补丁；照顾 27B 的能力和稳定性；prompt 精炼可读，可用通用合成例子或清晰条件规则，但不得放入真实评测例子、答案、样本 ID 分支或隐藏 DAG；避免职责重叠和无关上下文，不增加额外常规串行模型阶段。

## 4. 六个设置、部署与配置陷阱

配置均在 [assistant/configs/models/](assistant/configs/models)，角色 prompt 在 [assistant/configs/prompts/](assistant/configs/prompts) 的 5 份 `gp_*.yaml`。六个设置**包括 Full**，所有设置都使用 `--baseline goal_progression`。

| 设置 | assistant profile | 与 Full 的职责差别 | 正常生成次数 |
|---|---|---|---:|
| Full | `qwen_3_6_27b_gp` | Tracker + 两 Planner + Generator | 4 |
| no_tracker | `qwen_3_6_27b_gp_no_tracker` | 两 Planner 各自产出本分区语义+计划包；不隐藏调用 Tracker | 3 |
| no_intra | `qwen_3_6_27b_gp_no_intra` | Generator 显式承担当前规划与正文，统一编译请求 | 3 |
| no_inter | `qwen_3_6_27b_gp_no_inter` | 删除主动相邻规划，不由其他角色代做 | 3 |
| joint | `qwen_3_6_27b_gp_joint` | 一次联合 Planner 输出两份计划，是结构对照 | 3 |
| no_anticipate | `qwen_3_6_27b_gp_no_anticipate` | Inter 只允许 none/elicit | 4 |

次数是无重试且有正文的通常路径，不是每轮固定收费次数。历史 R1–R20 的 20 次在线结果均为 Full；本次另完成第 0 节的六组诊断矩阵。已有 `prompt_base` 不新建实现，也不占本表六个 GP 设置的名额；没有额外 GP Prior、Reviewer、无跨轮状态或 Scaffold 设置。

当前 Full 配置要点（完整参数说明以 implementation_v2.md 为准）：

- Qwen3.6-27B，`provider=vllm`、`model_id=qwen3.6-27b`、`base_url=http://127.0.0.1:8001/v1`，thinking 关闭。不要误用同目录的 OpenRouter thinking profile。
- Simulator 使用已有 `deepseek_v4_flash_0731`，hard，max-turns=20。默认数据集为 `user_simulator/dataset/DAG.jsonl`；20 次试验用指定 `--all --limit 16` 的样本及顺序，复现时核对 batch 快照的路径、hash 和样本列表。
- episode 并发为 16；GP 同进程/事件循环、同 provider/URL/model 的共享请求上限为 **32**，不是每条样本 32，也不控制 simulator；多进程不共享这把限流器。
- 实际总上下文 **262144 token**。各角色正常输入软限 16384；Tracker 扩容上限 **65536**，其余角色 32768。软限不等于模型总容量，必要原文不能因超软限被静默删除。
- 初始输出上限：Tracker 4096、Intra 2048、Inter 1024、Joint 3072、Generator 32768；截断恢复上限分别为 8192/4096/4096/4096/32768。
- Generator 采样仍为 temperature=0.7、top_p=0.8、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1，与指定 non-thinking baseline 参数对齐。
- Generator 当前 `structured_decoding=prompt`：不向服务端强制 JSON schema，但提示内 schema、本地 JSON/契约校验和恢复仍存在。其他启用角色默认 `schema`。它不是自由文本放行，也未证明对所有后端都更优。
- 当前每轮请求预算 1、活跃相邻候选上限 2、anticipate 交付单元上限 1；单请求时限 180 秒，轮级总时限 900 秒。参数可在合法范围内调节，不把推荐值另写死进逻辑。

### 重试是三层，不是“组件失败后系统整体重跑”

| 层级 | 当前设置 | 实际含义 |
|---|---|---|
| 客户端单次请求 | `retry.max_attempts=1` | 包含首次，因此不自动重发；GP 当前要求该值为 1，SDK 自动重试也关闭，避免配额相乘。 |
| GP 组件恢复 | `recovery.default_max_retries=3` | 首次之外最多 3 次，按 episode/turn/owner/error_code 独立计数；不同错误不共享，新输入版本也不清零。 |
| 整 episode | `--sample-retries 0` | 不从头重跑失败样本。设大于 0 会触发外围交互重试，当前管线会清理被替代尝试目录。 |

GP 的可配置错误配额范围是 0–3，见 `by_role_and_code`；旧 `format_retries` 只兼容映射 parse/schema，不是新一层重试。总 deadline、取消或不可恢复错误可提前终止，不承诺任何错误都用满四次尝试。

## 5. 历史 R1–R20 结论与沿用的指标口径

本节只记录历史 R1–R20；当前冻结矩阵见第 0 节及在线报告。历史主对照是 non-thinking Prompted Base：[20260826T083442Z_ddf4e7b6](runs_batch_qwen_full/20260826T083442Z_ddf4e7b6)，对应完整 16 条 E=3.2500、S=4.5000。不要误用 `20260826T083500Z_2820c6c9` thinking 批次，也不要把 `assistant/result.md` 中全量结果混作这 16 条的对照。

最新用户指定的计算方法：

1. 根据最终状态纳入 `status=completed` 或 `outcome=FAILURE_TURN_LIMIT`；其他失败排除并单独报告数量，不靠 `turns=20` 判断。
2. 使用现有 `intent_metrics` 计算每个 episode 的 all-node exposure/satisfaction turn，再对纳入 episode 求算术平均；未达标指标仍记 21，已经达标的 E 不因后续超限而重写。
3. 同样本对照按当次纳入的 sample_id 集合重算 baseline。样本集合不同，不能不加说明直接比较两个子集分数；原始日志和单样本算法不改。
4. token 独立于 E/S 的可用性；token 缺失不得把本来可算 E/S 的样本剔除。符合纳入条件却读取失败的指标应显式报错，不静默缩小分母。

| 批次 / 版本 | 完成 / 超限 / 其他失败 | 纳入 N | GP E / S | 同样本 Base E / S |
|---|---:|---:|---:|---:|
| try_10 / R10，最低 E 子集 | 15 / 0 / 1 | 15 | 3.4667 / 5.0667 | 3.1333 / 4.4000 |
| try_14 / R14，全员完成时最低 S | 16 / 0 / 0 | 16 | 3.8125 / 5.3125 | 3.2500 / 4.5000 |
| try_17 / R17，最低 S 子集 | 14 / 0 / 2 | 14 | 3.7143 / 5.0000 | 2.6429 / 3.9286 |
| try_18 / R18，全员完成时最低 E | 16 / 0 / 0 | 16 | 3.7500 / 5.5625 | 3.2500 / 4.5000 |
| try_19 / 同一 R18 复跑 | 13 / 1 / 2 | 14 | 4.3571 / 7.1429 | 3.3571 / 4.7143 |
| try_20 / R20，最终保留 | 14 / 1 / 1 | 15 | 3.6667 / 5.6667 | 3.4000 / 4.5333 |

全部 20 次都没有 E 或 S 优于各自同样本对照；最小 E 和最小 S 也不来自同一批。try_17 排除了两条在历史 baseline 中较长的样本，不能用其子集 S 宣称超越全 16 条版本。上述样本被反复用于开发，且当前 simulator 部分 prompt/routing、可见交互与历史不同；尚无严格冻结同条件、独立数据上的优势证据。

完整 21 批表（含起点 test2）和逐次说明在 iteration_summary.md；[逐样本 TSV](v2/runs_batch_qwen_full/gp_v2/iteration_metrics_completed_or_turn_limit.tsv) 保存 336 条记录的纳入标记及双方 E/S。旧笔记中的整批 N/A/“运行中”不再代表最新结果。

可见 token 口径保持：`mean_over_episodes(sum(tokens(用户可见 assistant 回复)))`。不计用户输入、组件草稿、JSON 包装、隐藏 reasoning、内部重试和未交付输出，也不除以对话轮数。GP 对 Renderer 最终文本单独计数；`visible_response_tokens` 未知时不回退冒充内部 completion 用量，E/S 仍保留。历史 token 比较应核对实际计数来源，必要时用同一 tokenizer 离线重算，不改旧日志。

## 6. R20 保留了什么，什么仍未解决

版本演进不需要从头重做；详细原因看迭代汇总：

- R1–R8 主要修复多余 Generator 调用、动作/状态引用、单一关系来源、错误定位和证据消息/段落配对；同时启动了相邻候选。
- R10–R13 调整 Need 的适用目标、已完成进展锚点、需求/答案职责，以及真实用户材料和演示的边界。
- R9 的“自动给下游上一条整段回复”在 R14 撤回。R15–R17 新增语义提示及 Generator 重写也未保留。R18 回到 R14 的语义基础，但保留 R17 的 Tracker 容量配置与预检审计；try_18/19 是同一版本复跑，不是两个新方案。
- R20 只进一步修正错误 JSON 字段路径，以及客户端截断异常中已有原输出的提取、局部修复和审计。重复身份仍拒绝，模型自己修复，不自动改名/合并；没有提高输出上限或重试配额。

仍需区分工程稳定性和语义效果：

- 合法 JSON/ID 不保证目标识别、目标与动作匹配或答案正确；旧目标被移出快照后又被引用、Need 语义漂移等问题仍可能出现。
- 相邻动作数量不代表推进收益。当前与相邻可能重复比较、过早询问，或没有把相关工作合并推进。
- 用户只有提交意愿、没有真实材料时，仍会发生承诺/询问/猜测循环；必要原文未选入下游时，评价也可能退化为泛化反馈。
- Generator 仍可能产出不完整交付、正文隐式重复提问、未核实事实；不存在能够可靠保证语义正确的额外审核器。

最后 try_20 唯一其他失败是 user22_task2 的 simulator 首次用户生成返回空结构，GP 尚未被调用，未越界修复或补跑；user20_task1 回合超限，E=5/S=21。该批无 GP 组件终止错误，但 R20 新修复针对的重复 ID/截断路径没有在线触发，仅有合成回归验证，不能据此宣称恢复机制已经稳定。

## 7. 代码入口与审计方式

GP 源码集中在 [assistant/src/assistant/goal_progression/](assistant/src/assistant/goal_progression)。按任务定位：

| 要处理的内容 | 主要文件 |
|---|---|
| 会话接口、原子提交、恢复 checkpoint / events | `session.py`；每轮执行顺序与依赖失效看 `engine.py` |
| 状态与组件输出字段 | `schemas.py`、`tracker.py`、`state_schema.py` |
| ID、事实计数、状态差分 | `identity.py`、`events.py`、`state_delta.py` |
| 各角色上下文、材料和锚点 | `context.py`、`anchors.py` |
| 规划调用、合法动作组合、契约校验 | `planners.py`、`action_schema.py`、`contracts.py` |
| 授权、生成与最终文本 | `assembly.py`、`generator.py`、`rendering.py` |
| 配额、超时、原始响应、错误反馈 | `runtime.py`、`errors.py`、`repair_context.py` |
| 配置接线与角色提示 | `assistant/src/assistant/config.py` 的 GP 命名空间、GP YAML；加载看 `prompts.py` |

已有 `factory.py` 会选择 `GoalProgressionSession`，对外仍是 `respond(...) -> str`、history/reset/replace_history/finalize_task 和 `last_call_metadata`；GP 多调用不应塞进普通 baseline 的 `build_messages`。响应级 metadata 用于归属并行调用，不能依赖共享 client 的“最近一次响应”字段。完整 checkpoint/events 可精确恢复；只有可见历史时请求历史次数为未知，不能伪造为零。v1 状态不可直接导入 v2。

每个样本沿用 `events.jsonl`、`transcript.jsonl`、`config_snapshot.yaml`、`final_state.json`，批次保存 `batch_config.yaml` / `batch_summary.json`。内部输出必须继续独立落盘，不能只保留最终回复：

- `assistant_{role}_requested/raw_result/validated/validation_failed/failed/cancelled`：请求上下文、原始/结构化输出、错误、用量及耗时。
- `assistant_gp_snapshot_projected/contract_compiled`：语义状态和最终授权；`assistant_gp_context_preflight`：输入估算、输出预留、软/扩容/硬限制。
- `assistant_gp_retry_scheduled/recovery_exhausted/dependencies_invalidated/optional_pruned`：恢复配额及影响范围。
- `assistant_gp_reply_composed/commit_prepared/state_committed/turn_failed`：正文、冻结问题、回执和事务。

审计标识为 `architecture_version=v2_contracts`，不能仅凭 profile 名判断 v1/v2。native 事件可能因 I/O 重试重复，按 episode_id + event_id 去重；最终交付以 pipeline 的 `assistant_generation_completed` 确认为准，孤立写前日志不证明交付。聚合调用账本仍在 `payload.llm_call.goal_progression.calls`，错误字段是 `validation_error`；旧截断原文可能需查 `usage.transport_attempts[-1].raw_output`，R20 已把它接入组件原始输出与修复。内部内容不进入用户可见 transcript 或 simulator 对话。

## 8. 可复用命令与验证记录

所有命令从 `intentLLM/` 仓库根目录执行，不要再套一层 `intentLLM/`。先准备源码路径：

```bash
export INTENTLLM_ROOT="$PWD"
export PYTHONPATH="${INTENTLLM_ROOT}/interaction_pipeline/src:${INTENTLLM_ROOT}/assistant/src:${INTENTLLM_ROOT}/user_simulator/src:${INTENTLLM_ROOT}/metrics/src${PYTHONPATH:+:${PYTHONPATH}}"
```

只读复算（不调用模型，务必显式选择最新口径）：

```bash
python assistant/scripts/gp_iteration_inspect.py \
  v2/runs_batch_qwen_full/gp_v2/try_20 \
  --metric-population completed-or-turn-limit
python assistant/scripts/gp_iteration_inspect.py --fingerprint
```

检查工具还支持 `--sample`、`--turns`、`--progress`。不加新口径选项仍用旧 `all` 模式，不能据其 N/A 覆盖当前总结。`gp_analyze_iteration.py` 和早期 notes 保留旧报告逻辑，重新统计优先用上述入口。移动过的历史目录以实际 batch 路径加 `Path(run['run_dir']).name` 定位，不修改原记录中的绝对路径。

离线回归（最近结果 371 passed；日志 `v2/diagnostics/final_assistant_tests.log`）：

```bash
python -m pytest -q --tb=short -c assistant/pyproject.toml assistant/tests v2/assistant/tests/integration/test_gp_diagnosis.py
python -m ruff check assistant/src/assistant/goal_progression \
  assistant/scripts/gp_iteration_inspect.py assistant/tests/unit/test_gp_*.py \
  assistant/tests/integration/test_gp_iteration_inspect.py
```

注意：`interaction_pipeline/tests/test_goal_progression_integration.py` 仍是 v1 状态/回退 fixtures，位于此前禁止修改范围，不能为使其通过而恢复旧协议。初次 v2 报告的 pipeline 回归明确排除了它；v2 接线和审计已由 `assistant/tests/integration/test_gp_native_audit.py` 等覆盖。不要把 assistant 测试通过扩大表述成全仓库测试通过。

以下是原有 16 条运行方式的后续复用模板，**执行须计入明确授权且尚未使用的额度，本轮不再追加**；将输出目录替换为新的唯一名称，不能复用已完成的 try_1…try_20，也不自动进行全量或 baseline 重跑：

```bash
python -m interaction_pipeline.cli.batch \
  --all \
  --limit 16 \
  --baseline goal_progression \
  --assistant-model-profile qwen_3_6_27b_gp \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --difficulty hard \
  --max-turns 20 \
  --seed 42 \
  --concurrency 16 \
  --sample-retries 0 \
  --no-update-memory \
  --output-dir runs_batch_qwen_full/gp_v2/next_run_YYYYMMDD_HHMMSS
```

此命令真实调用 Qwen 与 DeepSeek，需服务和 simulator 凭据可用。pipeline 默认 `sample_retries=3`、`update_memory=true`，所以这两个关闭选项不可省略；`--no-update-memory` 不关闭轮间 Tracker。单轮回放 `replay_gp_turns.py` 和 Generator 探针 `probe_gp_generator.py` 也是真实模型调用，不是只读工具。更改配置/prompt 后须新启动进程，不假设运行中的批次会热更新。

R20 运行时源码 + 5 份 GP prompt + 6 份 GP profile 的联合 SHA256 为 `ba2a3c02c2972adf09028104b45d66cbdbae72235226e3854982f175dbb3a4f1`，本次已核对一致；不包含文档、测试和统计脚本。版本快照见 [revisions/](v2/runs_batch_qwen_full/gp_v2/revisions)，R20 为 [R20.tar.gz](v2/runs_batch_qwen_full/gp_v2/revisions/R20.tar.gz)。这些 R 编号是实验文件版本，不是 Git commit。

## 9. 后续任务的边界与起点

后续实施以 [plan_v2_5_updated.md](assistant/plan_v2_5_updated.md) 为准：允许原位覆盖 GP，保留审计、身份/引用防线和统一有界恢复；不维护双套生产架构。旧对照从用户备份或已核验的冻结副本执行，不能将新代码搭配旧 profile 名当作 R20。新计划三次重复协议与下面的历史诊断额度分开记账。

此前诊断及最小在线矩阵已验收；当时新增分析脚本、合成测试、诊断产物、说明和六批新日志，GP runtime/schema/prompt/profile 未改。未来实施仍须遵守相应任务范围，**不要修改其他 baseline、共享模型客户端、simulator、interaction 或 metrics 来改善 GP 分数，也不要改历史日志或隐藏评测。** 用户自行调整 simulator YAML 的事实已记录，不能将它算成 GP 架构改进。

后续从在线报告的未决问题和最新 v2.5 计划出发，无需重新执行 P0–P6。旧诊断剩余 14 次是历史余额，不是必须耗尽的任务，也不是在本版三次重复协议外自动追加运行的许可。当前没有确认能够稳定胜出的架构；结构简化候选的条件均值、测试通过和吞吐变化都应与工程可靠性及语义推进分开解释。

工作区存在大量既有 tracked 修改和 untracked 文件，它们是用户的项目内容。先看 `git status`，只编辑本次范围；不 reset/clean、不删除运行目录、不回滚无关修改，不因“未跟踪”就认为可覆盖。历史交接记录的受保护源码/config 联合校验值为 `2171f5068299b2263d30cd694156807da101e428eca9fb767c887e62df301a34`，范围是 `interaction_pipeline/src,configs`、`user_simulator/src,configs`、`metrics/src`、`assistant/src/assistant/baselines,llm`；它是此前的历史核验值，不用于覆盖本次用户对 simulator 配置的修改记录；本阶段以 `v2/diagnostics/manifest.json` 与 `v2/diagnostics/validation.json` 的逐文件比较为准。

每次后续交付分清四件事：代码已接入、离线测试通过、真实模型跑过、相对 baseline 稳定提升。当前前三项已有记录，第四项尚未达到。继续更新本文件和 iteration_summary.md 时，应保留这个区分，并明确新增验证覆盖了什么。
