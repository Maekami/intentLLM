
# Goal-Progression Agent v2.5：Joint Policy + Minimal Prompts

> 状态：v2.5 实施方案。
> 基线：继承 R20 / `plan_v2_updated.md` 已验证的 Events、Semantic Projection、Turn Contract、Renderer、Receipt、事务与审计基础设施。
> 核心变化：将 Full 的并行 Intra/Inter Planner 简化为一个真正联合决策的 Joint Policy，并从第一原则重写 Tracker / Joint / Generator 三个生产 prompt。
> 目标：减少语义职责切分和提示词干扰，同时保留 v2 已建立的事实所有权、信息边界和可审计执行闭环。
> 本版不修改 simulator、DAG、Hard difficulty、metrics 或其他 baseline。

## 1. 为什么需要 v2.5

R20 的在线结构诊断给出以下信号：

- Prompted Base：E=4.0625，S=5.1875。
- Full R20：E=3.5000，S=7.3750。
- Joint：有效样本 E=3.7333，S=5.8000。
- Full 能更早推进 exposure，但存在少数严重 satisfaction 长尾。
- Joint 基本保留 Full 的 E 优势，同时显著降低 S、调用数、内部 token 和 latency。
- no_inter / no_intra 没有整体恢复到 Base。
- no_tracker 有性能信号，但失败、恢复和上下文成本明显更高，不适合作为当前主方案。

因此下一步不继续对 Full R20 做 R21/R22 式局部修补，也不直接启动 v3。

v2.5 的问题定义是：

> 保留 v2 的状态与执行基础设施，但减少 policy decomposition 和 prompt complexity，检验一个更简单的三调用架构能否稳定获得 progression 收益。

## 2. Prompt complexity 是一个需要独立处理的架构变量

当前生产 prompt 的主段规模约为：

| Prompt    | 主段行数 | 英文词数 |
| --------- | -------: | -------: |
| Tracker   |      101 |      934 |
| Joint     |       65 |      537 |
| Intra     |       57 |      530 |
| Inter     |       54 |      463 |
| Generator |       62 |      561 |

R20 Full 每轮涉及的四个生产 prompt 主段合计约 2,488 词。
现有 Joint 路径 Tracker + Joint + Generator 仍约 2,032 词。

这些数字本身不是性能结论；问题主要是规则密度和职责交叉：

1. 同一个语义边界在多个角色重复描述。
2. schema/runtime 已能检查的结构约束仍大量写进自然语言 prompt。
3. R1–R20 暴露的失败模式逐渐以禁止句、例子和特殊边界形式积累。
4. 生产 prompt 多次重复“用户承诺但未提交”“拒绝旧假设”“disputed draft”“外部验证”等情景。
5. Synthetic tile example 在多个角色重复出现，可能形成不必要的任务锚定。
6. Planner prompt 同时包含 policy、schema 说明、异常处理、事实边界和 realization 建议。
7. Generator 作为 realizer 仍承担较多 verification / recovery 解释，增加再次决策的诱因。

对 Qwen3.6-27B non-thinking，本版将 prompt complexity 视为需要控制的模型输入设计问题，但不预设“更短一定更好”。

目标是：

> 删除不属于该角色的决策，而不是机械删除必要信息。

## 3. v2.5 设计原则

### 3.1 三次正常 LLM 调用

```text
Visible history + runtime facts
            |
         Tracker                 LLM #1
            |
     Semantic Snapshot
            |
     deterministic Context Builder
            |
       Joint Policy              LLM #2
            |
       Turn Proposal
            |
 deterministic Assembler
            |
       Turn Contract
            |
        Generator                LLM #3
            |
         Renderer
            |
 Visible reply + ExecutionReceipt
```

正常路径不增加 Reviewer、Checker、Router 或第二次 policy call。

### 3.2 保留唯一所有权

| 内容                                             | owner     |
| ------------------------------------------------ | --------- |
| 当前用户语义、Goal/Constraint/Need identity      | Tracker   |
| 历史 request 次数和 receipt                      | Runtime   |
| 本轮整体 progression policy                      | Joint     |
| 最终授权、priority、dependency 和 request budget | Assembler |
| 已授权正文的具体实现                             | Generator |
| 最终文本拼装与执行回执                           | Renderer  |
| satisfaction / hidden intent completion          | benchmark |

消费者只引用，不重新裁定上游 owner 已经拥有的事实。

### 3.3 Prompt 只表达不可由代码替代的语义规则

以下内容优先由 schema/runtime 实现，不在生产 prompt 中重复展开：

- JSON 类型和枚举合法性。
- ID 是否存在。
- ID scope / version 合法性。
- request eligibility 的已知计数部分。
- request budget。
- unit 数量上限。
- required Need 是否 available。
- required current coverage。
- duplicate unit / request。
- retry count。
- transaction / receipt / hash。
- Renderer 最终拼装顺序。

Prompt 只需要告诉模型完成其语义任务所需的最少边界。

### 3.4 Production prompt 不携带历史故障案例

默认移除生产 prompt 中的 synthetic tile example。

这些例子继续保留在：

```text
assistant/tests/
diagnostics/
prompt_regression/
```

用于回归和人工审阅，而不是每轮注入模型。

如果未来证明某个抽象边界仅靠规则无法稳定理解，才允许重新加入一个最小、领域无关示例；必须作为独立 prompt ablation 验证。

### 3.5 不再通过追加禁止句迭代

若出现新失败，先归类：

```text
semantic projection
policy decision
runtime contract
realization
benchmark variance
```

只有当失败暴露现有抽象缺口时才修改对应 owner。

不得：

```text
看到一个 sample
→ 在 prompt 末尾加一句 "Do not ..."
→ 再跑同一 sample
```

## 4. 保留的 v2 基础设施

v2.5 不重写以下部分：

- VisibleEvent / AttemptEvent / ExecutionReceipt。
- append-only audit。
- Semantic Snapshot 的 current projection 思想。
- full visible history 默认只给 Tracker。
- Context Builder 的 scoped material/evidence projection。
- Need identity 与 runtime request history。
- request eligibility 的 deterministic history 部分。
- Turn Contract。
- Renderer 冻结 request text 并生成 receipt。
- atomic session commit。
- typed error / bounded recovery。
- material_refs 与 evidence_refs 分离。
- hidden DAG / E/S / simulator private state 不进入 Agent。

这些是 substrate，不因 policy 简化删除。

## 5. Tracker：保留 schema，简化任务说明

### 5.1 不新增状态 ontology

v2.5 不为当前失败增加：

```text
temporary
shadow
progress_score
confidence
loop_state
skip_edge
satisfaction_estimate
```

等字段。

用户改口、临时要求、恢复、拒绝、重新开启旧目标继续由：

```text
visible history
→ current semantic projection
```

统一处理。

### 5.2 Tracker 仍负责

- 当前 active / addressed / paused Goal。
- current 与 adjacent scope。
- 当前有效 Constraint。
- InformationNeed。
- semantic identity 延续。
- remaining_work。
- decisive evidence_refs。
- source-specific material_refs。
- explicit renewed authorization 的语义识别。

### 5.3 Tracker 不负责

- 本轮 action。
- 是否现在应该提问。
- adjacent 是否值得本轮推进。
- assistant 应该给什么具体答案。
- request 次数。
- satisfaction。
- retry / execution decision。

### 5.4 Tracker prompt 目标

生产 prompt 目标约 300–450 English words。

这是开发目标，不是代码硬限制。

建议完整 system prompt：

```text
## Role
Maintain the current semantic projection of the user's visible conversation.
Decide WHAT outcomes, constraints, information needs, and source materials
currently matter. Do not decide this turn's actions or write the answer.

## Rules
1. Current goals are user-visible outcomes needed now. Adjacent goals are
   distinct plausible next outcomes grounded in current work; a prerequisite
   of current work is not adjacent.
2. Keep only currently relevant state. Mark a goal addressed only from an
   actual visible assistant delivery, never from an old plan.
3. A Need is a user fact or actual source the assistant cannot truthfully
   choose for the user. Ordinary assistant-owned choices are not Needs.
   Unknown information is not automatically blocking.
4. Reuse the same semantic identity when the same goal, constraint, or Need
   continues across turns or changes scope. Do not merge different facts
   merely because they have similar names.
5. Constraints and inferred preferences must stay within visible evidence.
   Update when the user changes them; do not predict future reversal.
6. Keep decisive evidence references and all original material needed for
   source-specific work. Assistant examples or disputed drafts are not user facts.
7. A new authorization to ask for an old Need requires explicit new user
   evidence. Willingness to provide information is not the information itself.

## Output
Return only the requested SemanticProjection.
```

### 5.5 从当前 Tracker prompt 删除

删除或迁出生产 prompt：

- ASCII local-name 细节：schema/runtime 文档负责。
- “Never invent g/c/n IDs”等 runtime invariant 的重复说明。
- policy.adjacent_candidate_limit：schema/runtime 注入。
- changed_goal_ids：Runtime 计算。
- Goal.constraint_ids 反向索引说明：schema/runtime。
- 大量 completed-anchor 的实现细节：Context Builder / schema。
- synthetic tile example。
- 多次重复的 disputed draft 描述，只保留一条事实边界。
- “readiness/process signal”之类的具体失败历史措辞，除非 schema regression 证明删除后稳定退化。

## 6. Policy 层：Dual → True Joint

### 6.1 不再保留两个独立 Planner

v2.5 主路径删除：

```text
Intra Planner
Inter Planner
parallel planner synchronization
cross-planner request competition
```

旧 R20 Full 和现有 Joint profile 保留为冻结对照，不覆盖。

### 6.2 Joint 的职责

Joint 只回答：

> Given the current semantic state, what combination of assistant deliveries
> and at most one information request best advances this turn overall?

它同时看到：

- active current goals；
- registered adjacent goals；
- relevant constraints；
- relevant Needs + request eligibility；
- required materials；
- latest user；
- relevant evidence。

不读取 full history，不重建 semantic state。

### 6.3 不再输出 IntraPlan + InterPlan 两张表

现有：

```text
JointPlan:
  intra: IntraPlan
  inter: InterPlan
```

仍保留了 Dual 架构的概念边界。

v2.5 改为统一：

```text
TurnProposal:
  deliveries:
    - goal_id: GoalId
      target: short assistant-executable result
      required_need_ids: NeedId[]
      material_refs: EvidenceRef[]

  request:
    goal_id: GoalId
    need_id: NeedId
    question_text: string
    | null

  blocked_current:
    - goal_id: GoalId
      need_ids: NeedId[]
```

### 6.4 删除显式 action label

v2.5 主路径不要求 Joint 再分类：

```text
advance
revise
clarify
none
elicit
anticipate
```

这些 label 在 R20 中主要是中间 policy 描述，而真正被执行的是：

```text
delivery
request
blocked
```

v2.5 将行为语义直接映射为结构：

```text
current delivery       ≈ advance / revise
current request        ≈ clarify
adjacent delivery      ≈ anticipate
adjacent request       ≈ elicit
no adjacent proposal   ≈ none
blocked_current        ≈ null/blocked
```

Runtime 可从 goal.scope 与 proposal type 派生旧标签供 audit 使用；模型不再承担额外分类。

这样避免：

```text
action label 合法
但 target 实际不符合该 action
```

这一类冗余决策。

### 6.5 current / adjacent 仍由 Tracker 拥有

Joint 不重新判断 scope。

Assembler 根据 `goal_id → scope` 自动得到：

```text
current delivery = required
adjacent delivery = optional
current request > adjacent request
```

Joint 只决定是否提出对应内容。

### 6.6 Joint 的真正联合决策

Joint prompt 不写成：

```text
First solve Intra.
Then separately solve Inter.
```

而写成一个整体 policy：

```text
## Role
Choose the best overall progression for this turn from the supplied semantic
state. Do not reconstruct state and do not write the final reply.

## Decision
1. Ensure every active current goal has useful progress: a concrete delivery,
   one justified request, or an honest blocked record when no truthful useful
   work is possible.
2. Prefer doing useful work now. Ordinary preference uncertainty usually
   permits a provisional or conditional result; do not wait for perfect
   personalization.
3. Request an eligible Need only when knowing it now materially changes the
   result or next decision. If useful independent work exists, propose it too.
4. After current work is covered, include adjacent progression only when it is
   distinct, grounded, executable now, and adds clear value in the same turn.
   Do not repeat current work or treat a future user action as a delivery.
5. Delivery targets specify the result and essential parts, not the factual
   answer itself. Use only registered goals, Needs, constraints, and materials.
6. Keep the turn focused. Omit optional progression whose value is weak or
   whose inclusion competes with the current task.

## Output
Return only the requested TurnProposal.
```

目标约 250–400 English words。

不提供 synthetic example。

## 7. Assembler：从“协调两路”变成“验证并封闭统一提案”

Assembler 仍是 deterministic code。

输入：

```text
SemanticSnapshot
Interaction/request facts
TurnProposal
policy configuration
```

输出：

```text
TurnContract
```

### 7.1 保留的检查

- goal/Need/material refs 合法。
- scope/version 合法。
- required_need_ids 已 available。
- request need eligible。
- request_budget。
- current goal coverage。
- adjacent capacity。
- required vs optional criticality。
- blocked_current 的 Need 合法。
- variant capability。

### 7.2 删除的复杂度

不再需要：

- Intra request vs Inter request 的两个模型结果竞争。
- Intra/Inter planner result synchronization。
- 两路 proposal 的重复 unit 合并。
- 两个 Planner 分别修复后重新协调。
- 两路同时对同一 semantic target 作独立决策。

### 7.3 current coverage

每个 active current Goal 必须满足至少一个：

```text
有 current delivery
或
被唯一 selected request 合法推进
或
出现在 blocked_current
```

这是结构完整性，不等于 semantic quality。

### 7.4 request

Joint 只提出 `request | null`，默认每轮最多一个 user-facing request。

如果 request 属于 adjacent Goal，但存在未被妥善处理的 current Goal，
Assembler 不通过自动语义猜测修复；将其作为 `contract.coverage` 或 policy error
返回 Joint 在同一职责内修复。

## 8. Generator：进一步变薄

### 8.1 Generator 只看

```text
sealed TurnContract
latest_user
contract-relevant constraints
contract-relevant user facts
required material_refs
minimal local continuity
```

不看：

- raw TurnProposal；
- rejected proposal；
- full Snapshot；
- full history；
- retry history；
- hidden DAG；
- action labels。

### 8.2 Generator prompt 目标

生产 prompt 目标约 180–300 English words。

建议：

```text
## Role
Realize the sealed Turn Contract as complete, useful body units.
Do not re-plan the turn or change the selected work.

## Rules
1. Produce every contracted body unit completely. An introduction, promise,
   paraphrase of the target, or offer to continue is not a delivery.
2. For user-specific or source-specific claims, use the supplied facts and
   original material. General advice or newly requested content may use normal
   knowledge and reasoning. Do not invent user facts, unseen sources, or actions
   that were not performed.
3. Follow the supplied constraints and preserve uncertainty when the evidence
   does not establish a fact.
4. Do not add standalone questions or invitations to the user. Renderer appends
   the contracted request verbatim.
5. Report a local issue only when a genuinely required supplied input is missing
   or conflicting such that the unit cannot be truthfully produced.

## Output
Return only the requested units and issues.
```

### 8.3 从当前 Generator prompt 删除

- 大段 time-sensitive verification / certification / safety wording。
- 多次重复的 “previous assistant is not verification”。
- synthetic tile example。
- CurrentPlan/InterPlan 相关解释。
- unit ID uniqueness 等 schema 可检查规则的自然语言重复。
- `no_intra` mode。
- 不属于 realization 的 planning fallback。

信息真实性只保留一个统一边界：

> 不编造用户事实、源材料和未执行的外部行为。

## 9. Prompt ownership matrix

每条核心规则只出现于一个生产 prompt；其他组件依赖结构化输入。

| 规则                         | Tracker |      Joint      | Generator |      Runtime      |
| ---------------------------- | :-----: | :--------------: | :-------: | :----------------: |
| current / adjacent scope     |   ✓   |    read only    | read only |    validate ref    |
| semantic identity            |   ✓   |    read only    |    —    |    ID registry    |
| Need meaning/status          |   ✓   |    read only    | read only |   request facts   |
| request history/count        |   —   | read eligibility |    —    |         ✓         |
| whether to ask now           |   —   |        ✓        |    —    | budget/eligibility |
| whether to progress adjacent |   —   |        ✓        |    —    | optional capacity |
| target content               |   —   |        ✓        |  realize  | dependency checks |
| actual answer wording        |   —   |        —        |    ✓    |         —         |
| unit/request coverage        |   —   |     propose     |  produce  |         ✓         |
| retry                        |   —   |        —        |    —    |         ✓         |
| satisfaction                 |   —   |        —        |    —    |     benchmark     |

若某条自然语言规则需要同时出现在两个 prompt，必须说明两个消费者为何都需要做语义判断；否则视为 prompt duplication。

## 10. Context management

继续采用：

```text
Tracker:
  full visible history
  previous Snapshot
  relevant request facts
  previous compact execution bridge

Joint:
  latest user
  current goals
  adjacent goals
  relevant constraints
  relevant Needs + eligibility
  required materials/evidence

Generator:
  latest user
  Turn Contract
  contract dependency closure
```

### 10.1 不因 prompt simplification 恢复 full-history broadcast

Prompt 变短不意味着 Planner/Generator 重新获得完整 history。

若 Joint 缺少必要材料：

1. 先检查 Tracker material_refs；
2. 再检查 Context Builder dependency closure；
3. 若已有材料漏传，走 `context.missing`；
4. 若用户从未提供，保持正常 Need / blocked 状态。

不把 full history 当保险兜底。

### 10.2 上下文 token 继续审计

每轮记录：

```text
tracker_system_tokens
tracker_context_tokens

joint_system_tokens
joint_context_tokens

generator_system_tokens
generator_context_tokens
```

prompt rewrite 前后单独报告 system tokens，避免把 context reduction 和 prompt reduction 混为一谈。

## 11. Prompt rewrite 的验收方式

Prompt 简化不是简单比较字符数。

必须完成三类检查。

### 11.1 Ownership lint

人工或脚本检查：

- Tracker prompt 是否包含 action selection。
- Joint prompt 是否重新定义 scope/identity。
- Generator prompt 是否存在 planning instruction。
- runtime invariant 是否被多处重复描述。

### 11.2 Schema regression

使用现有合成测试验证：

- same Need identity。
- asked-but-unanswered 不被重新当 fresh。
- material source 不混淆。
- current/adjacent 不能错引用。
- request eligibility。
- user future action 不能成为 delivery。
- Generator 不漏 contracted unit。
- Renderer request receipt。
- transaction/retry。

这些测试验证系统 contract，不要求把测试案例全文重新写进 prompt。

### 11.3 Semantic prompt fixtures

保留少量完全离线 fixture，覆盖抽象边界：

```text
optional preference vs required Need
current prerequisite vs adjacent goal
user promise vs actual supplied material
rejected assumption vs valid revision
current delivery vs useful adjacent progression
```

fixture 用于开发检查，不自动进入生产 prompt。

## 12. Prompt 规模目标

初始目标：

| Role      |   当前主段 | v2.5 目标 |
| --------- | ---------: | --------: |
| Tracker   | ~934 words |  300–450 |
| Joint     | ~537 words |  250–400 |
| Generator | ~561 words |  180–300 |

总 production system prompt：

```text
current Joint path ≈ 2,032 words
v2.5 target        ≈ 730–1,150 words
```

即大致减少 40%–65%。

该数字是工程目标，不是 performance threshold。
若删除某条规则导致稳定回归，可以恢复最小必要语义，而不是为了满足词数目标牺牲正确性。

## 13. 文件与配置

建议新增独立 v2.5 文件，不覆盖 R20：

```text
assistant/configs/prompts/
  gp_v25_tracker.yaml
  gp_v25_joint.yaml
  gp_v25_generator.yaml

assistant/configs/models/
  qwen_3_6_27b_gp_v25.yaml
```

配置明确：

```yaml
goal_progression:
  architecture_version: v2_5_joint_minimal
  variant: joint_v25
```

不要复用同名 `qwen_3_6_27b_gp_joint.yaml` 后悄悄改变语义。

旧 R20：

```text
qwen_3_6_27b_gp
qwen_3_6_27b_gp_joint
```

保持冻结，可随时复现。

## 14. Runtime / recovery

继续使用 v2 typed recovery，但因只剩一个 Planner，错误 owner 简化为：

```text
tracker
joint
generator
runtime
```

### 14.1 Joint repair

若出现：

```text
contract.reference
contract.coverage
contract.dependency
contract.request
output.schema
```

只把：

- 原 Joint 输入的必要部分；
- 对应字段错误；
- 相关坏输出片段

返回同一个 Joint 修复。

不加入整个错误历史，不回到 Tracker，除非错误确实说明上游 semantic artifact 无法满足自己的 schema。

### 14.2 不用 retry 修 semantic disagreement

如果输出结构合法，但离线认为：

```text
adjacent value low
current target weak
semantic choice suboptimal
```

不触发在线 retry。

这些属于模型 policy quality，通过实验评价，而不是伪装成 contract error。

## 15. v2.5 不修改 simulator / benchmark

硬性冻结：

- `first_16.jsonl`。
- DAG。
- Controller。
- Satisfaction Updater。
- User Realizer。
- Hard node selection。
- Abstract realization。
- max turns。
- E/S evaluator。
- monotonic_satisfaction 当前正式设置。
- 其他已跑 baseline 的 benchmark 配置。

不做：

- 固定 t0 文本。
- solvability-based sample filtering。
- hidden-node runtime hint。
- simulator prompt 修复。
- 为 GP 改 satisfaction criterion。

Benchmark 随机性通过多次独立运行的均值和标准差处理。

## 16. 开发流程

### M1：冻结与复制

- 保存 R20 Full / Joint prompt、schema、profile 指纹。
- 新建 v2.5 architecture_version。
- 不修改旧结果和旧 profile。

### M2：统一 TurnProposal

- 新增 v2.5 `TurnProposal` schema。
- 删除 v2.5 主路径显式 IntraPlan/InterPlan。
- Runtime 从 goal scope 派生 current/adjacent criticality。
- Assembler 接受统一 proposal。
- 保留现有 TurnContract / Renderer。

### M3：Prompt 从零重写

不要编辑 R20 prompt 做删句。

直接建立三个新 prompt：

```text
gp_v25_tracker
gp_v25_joint
gp_v25_generator
```

按本文件第 5/6/8 节重新写。

完成后做 ownership lint 和 word-count report。

### M4：Context Builder

- Joint 使用原 Intra + Inter 所需信息的去重并集。
- 不传 raw history。
- 不复制同一 material/evidence。
- Generator 只取 Contract dependency closure。

### M5：Recovery

- 删除 v2.5 Dual-only invalidation / repair path。
- 验证 Joint 局部修复。
- 保留所有 attempt audit。

### M6：Regression

先运行全部 GP unit tests 和合成 contract tests。

随后在已有 16 条开发样本上做 smoke/development run。

用途仅为：

- 检查工程失败；
- 检查 prompt 是否出现明显职责漂移；
- 检查 context/token；
- 检查 catastrophic loop。

不得根据单个样本继续往 prompt 追加特例。

## 17. 第一阶段比较

v2.5 工程稳定后，比较：

```text
Prompted Base
Full R20
existing Joint R20
Joint v2.5
```

目的：

- Base：最终外部参照。
- Full R20：确认原四调用路线。
- existing Joint：区分“只是少一次调用”和真正 v2.5 改造。
- Joint v2.5：新候选。

开发阶段可先用 first_16 / Hard。

正式结论使用更大的既定评测集；first_16 已被反复开发使用，不作为唯一泛化证据。

## 18. 重复实验

Simulator / benchmark 不修改。

正式报告对每个设置执行多次独立运行，建议至少 3 次，资源允许时 5 次或更多。

报告：

```text
E mean ± std
S mean ± std
VisibleTokens mean ± std
turn-limit rate
component failure rate
internal input/output tokens
actual LLM calls
p50/p95 latency
```

如果能在同一 run index 上形成可解释的配对条件，同时报告：

```text
ΔE = E_method - E_base
ΔS = S_method - S_base
```

的均值和标准差。

不要假设相同 seed 会使 simulator 在不同 assistant trajectory 下产生相同后续用户文本。

## 19. v2.5 成功标准

优先级：

### Primary

```text
E 和 S 相对 Prompted Base 均获得稳定改善。
```

### Reliability

- turn-limit 不明显增加。
- component failure 不明显增加。
- 不出现比 R20 更严重的 recovery explosion。

### Efficiency

相对 Full R20：

- 正常调用 4 → 3。
- internal token 明显降低。
- latency 明显降低。

### Prompt quality

- 三个 production prompt 的职责无明显重叠。
- 不出现基于真实失败样本的规则堆叠。
- Prompt 增长必须有 owner 和抽象理由。

## 20. 何时进入 v3

以下情况才启动 `plan_v3.md`：

1. Joint v2.5 在多次独立实验中仍稳定落后 Prompted Base；
2. 失败主要不是工程错误，而是 semantic/policy bottleneck；
3. existing Joint 与 v2.5 prompt simplification 都不能解决；
4. no_tracker/direct-history 类路径持续给出更好的 semantic quality 信号；
5. 证据支持“Tracker Snapshot 作为强制下游信息瓶颈”本身是主要损失源。

若进入 v3，优先重构：

```text
semantic projection → policy information path
```

而不是继续增加 Tracker 字段、Reviewer 或更多 action label。

## 21. 明确不做

v2.5 不做：

- 修改 simulator / benchmark。
- 新增 Reviewer / Checker LLM。
- 增加 Tracker lifecycle ontology。
- RequestBundle，除非后续数据证明 one-Need contract 是结构瓶颈。
- full-history broadcast。
- memory learning / Trace2Skill。
- hidden DAG aware policy。
- 在线 satisfaction prediction。
- 为某个 first_16 样本增加关键词规则。
- 继续维护 Dual Planner 作为新主路径。
- 为了保留旧 action 名称而让模型重复分类已经由 proposal structure 表达的行为。

## 22. 最终架构

```text
                    Visible Events
                         |
                      Tracker
                 Semantic Projection
                         |
                  Context Builder
                         |
                    Joint Policy
                  TurnProposal
                         |
                     Assembler
                   TurnContract
                         |
                     Generator
                     body units
                         |
                      Renderer
                         |
             Visible Reply + Receipt
                         |
                  atomic Session commit
```

对应三个 LLM 语义问题：

```text
Tracker:
  What does the user currently need?

Joint:
  What is the best overall progression for this turn?

Generator:
  How do I completely realize the sealed work?
```

其他问题尽可能由 Runtime 回答：

```text
What IDs exist?
What was already requested?
Is this request eligible?
How many requests are allowed?
Which units are required?
Did the selected block actually reach the visible reply?
How many retries remain?
```

## 23. 最终验收原则

> v2.5 的目标不是让每个模型“更聪明地遵守更多规则”，而是让每个模型只需要解决更少、更清晰的语义问题。

最终验收要求：

1. 正常路径只有 Tracker → Joint → Generator 三次 LLM 调用。
2. v2 的 Events / Projection / Contract / Receipt / transaction substrate 保留。
3. Joint 真正做整轮联合 policy，而不是一次调用填写两张独立 Planner 表。
4. 模型不再输出冗余 action label；行为由 delivery/request/blocked 结构表达。
5. production prompt 从失败历史中解耦，不携带真实样本或重复 synthetic case。
6. Runtime 能检查的 invariant 不在多个 prompt 中重复解释。
7. Tracker / Joint / Generator prompt 都有唯一 owner 和清晰信息边界。
8. Planner/Generator 不重新读取完整历史。
9. simulator、Hard benchmark 与指标完全不变。
10. 只有多次实验的 E/S 与可靠性证据支持，才将 v2.5 升级为最终主架构。
