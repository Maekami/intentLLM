# CURRENT_STATE — V4 Static 工作入口

更新：2026-09-15。**V3 已归档；活动仓库经核验仍为 R20 Format，无需恢复。当前进入 V4 静态系统任务，Evo 暂停。** V4 启动材料已冻结，新 Actor–Validator 尚未实现，真实实验启动数为 0。

## 1. 当前任务与唯一写入范围

后续 working agent 的主要执行文档是 [assistant/plan_v4_static.md](assistant/plan_v4_static.md)。先读任务书，再核对 [START_STATE](runs_batch_qwen_full/gp_v4/static/bootstrap/START_STATE.json) 与 [基线核验](runs_batch_qwen_full/gp_v4/static/bootstrap/baseline_verification.json)；[review_v4.md](assistant/review_v4.md) 提供机制、归因与工程/文献背景。不要继续执行 V3、v2.6 或旧 C1/C2/C3 计划。

**唯一可写工作区：`/home/jiaying/zihang/intent/intentLLM/runs_batch_qwen_full/gp_v4/static/`。** 后续的源码、配置、版本、脚本、测试临时文件、缓存、下载、实验及报告都放在这里。主项目源码/数据、本 CURRENT_STATE、主任务书/review、其他 runs 和全部历史档案只读；不得回写、合并或在 `/tmp`/用户 HOME/共享环境写入任务产物。

此次归档和根文档更新属于已完成的整理任务，不是 working agent 写入工作区之外的权限。允许只读主项目与历史证据。不要从 Git HEAD、V3 候选或旧执行器启动；主工作树的未提交修改保留，不 reset/clean/checkout、提交或推送。

工作区已准备 `bootstrap/r20_format/`（248 个输入文件）、清单、空启动账本与 [STATUS](runs_batch_qwen_full/gp_v4/static/STATUS.md)。**开发 repo 尚未创建。** agent 先独立复制 bootstrap 到 `repo/`，所有导入指向自己的副本；每个参测 arm 再冻结到 `versions/<arm>/repo/`，运行输出写唯一 `runs/<arm>/seed_*/launch_*/`。bootstrap 保持冻结，不使用可写软/硬链接指回主项目。

## 2. 已核验的 R20 Format 起点

四子项目 assistant、interaction_pipeline、user_simulator、metrics 的 `src/**/*.py`、`configs/**/*.yaml/yml` 和 `pyproject.toml` 共 **187 文件**。本次与 V3 原 START_STATE 和 v2.6 的 R20 Format 源快照逐文件比较全部一致，聚合指纹为：

`4ea8194887c8622b35f994075234e54cf1d64565de4a9eb82d0f5caaa45bb6a8`

算法：仓库相对 POSIX 路径排序，SHA256 依次输入每份“路径 UTF-8＋NUL＋原始字节”。60 个测试/通用脚本也一致；本次从**当前活动仓库**复制运行、测试与 target_32 到 bootstrap，未使用 V3 改版覆盖主项目。该冻结副本的原 assistant 离线测试 **352 通过（9.93 秒）**，临时文件均在任务目录，未启动真实模型；见 [测试日志](runs_batch_qwen_full/gp_v4/static/artifacts/bootstrap_assistant_tests.log)。

- A0：Full R20 Format，`goal_progression / qwen_3_6_27b_gp / v2_contracts / full`，R20 逻辑与四份保真格式整理 prompt；不是 v2.6 Full，也不是 V3 A1/A2/A3。
- B：已有 `prompt_base / qwen_3_6_27b_vllm_non_thinking`，不重写 Base。
- 活动正常路径仍是 Tracker → Intra/Inter 并行 → 代码 Assembler → Generator → Renderer/提交。**新路径只在 V4 工作副本实现。**
- simulator 默认仍为 DAG_fixed；V4 每版派生配置只覆盖 dataset_path 到 target_32。不改共享默认配置，不直接 `--all` 跑出 292 题。

V4 允许在工作副本改 GP 架构及必要 assistant 配置/工厂注册、prompt/profile、相应测试；保留旧 A0/P 分支。interaction_pipeline、simulator、metrics、公共 LLM 客户端和 Base 行为冻结，评测规则与隐藏 DAG 不改。具体允许范围以任务书为准。

## 3. V4 已确定的机制

主路径：**Tracker → 两位并行 Actor → 代码组装完整候选 → 一次联合模型 Validator**，正常三个模型阶段。Tracker 维护目标/约束/Need/证据；Actors 选择行动并生成各自的正文或问题，不再把 target 交给独立 Generator。一个不可拆成品由一位 Actor 完整生成，Inter 不读取本轮尚未完成的 Intra 选择。

首版门控优先序 **② Tracker 正确性 → ④ 目标推进 → ③ Actor 正确性**；①确定性内容门控暂缓。Validator 读取完整可见交互历史，含上一轮完整用户与助手消息、本轮用户消息、稳定前态/当前提案、行动、完整候选；失败草稿另区标为未交付。不要仅提供助手产物或默默裁成最近两轮；上下文超过配置容量时走明确的独立恢复/故障流程。

模型输出精简为有序 checks：每门 pass/reason，失败 refs，仅④失败需 `cause=tracker_state|actor_behavior`。身份、版本、门编号、恢复起点、重试次数、交付标记与成本均由系统补齐；审计保留完整实际输入/响应和来源，不要求模型重复生成审计字段。

②失败或④归 Tracker 时从 Tracker 重做；③失败或④归 Actor 时保留 Tracker 并重做两位 Actors。默认每个用户回合共两次**校验重执行**，原因与原文证据进入修复上下文；其他超时/解析/引用/容量等错误恢复独立按 turn/owner/error_code 计账，不因校验回滚重置。`3+3*nT+2*nA≤9` 仅是校验主路径上限，实际技术恢复可使串行深度超过 9。

校验两次重执行耗尽仍失败时，**直接交付最后一次完整组装回复，继续交互**；保留校验失败记录，不改记为通过或直接报语义错误。只提交一次，真实正文与已问事实保留；被否决 Tracker 提案与稳定状态分开，不因回滚忘记实际问过的问题。没有完整候选的技术故障另按有限技术恢复处理。

Prompt 使用易读 YAML＋Markdown 章节和通用正反例，少输出冗余字段、不堆真实样本补丁。隐藏 DAG、队列、满足标签/评估理由、终局 E/S 不进角色输入；delivered、模型通过、用户目标满足是三件事。

## 4. 数据、目标、预算与双 key 并发

唯一真实实验数据：[target_32.jsonl](user_simulator/dataset/target_32.jsonl)，SHA256：

`9ef1009c6f2f2f19c3929dc4f32b0282dd40624f39114e21c47ddc8c5cdf45b7`

四次 R20 全量中先排除任何出现普通错误的题，保留超限，在剩余 275 题中按四次实际助手回合数的样本方差取前 32，同方差按原位置，JSONL 保留所选题的原始相对顺序。32 题的 128 条历史轨迹为 104 成功、24 超限、0 普通错误；22 个 ID 曾超限、10 个四次成功。数据与选择口径保持冻结，不重新抽样或因新版本失败删题。

| 观察口径 | E ↓ | S ↓ |
| --- | ---: | ---: |
| 历史四次批均值平均 | 7.4140625 | 10.4140625 |
| 每题取同一条最佳完整轨迹后的目标 | **4.21875** | **5.53125** |

证据见 [目标统计](assistant/proposal_artifacts/v4_static/target_statistics.json)、[选择清单](assistant/proposal_artifacts/v4_static/selection_manifest.json) 与 [独立核验](assistant/proposal_artifacts/v4_static/independent_validation.json)。高方差开发集存在均值回归、四次用户路径也不同；历史最佳不是线上保证，必须重跑同期 A0/B。

**30 次完整 batch 上限，最多 960 episode 槽。** B、A0、P、M、V0、V2 各三次共 18；最多三版 F prompt 迭代各三次共 9；工程预留 3。seed=42/142/242。P 仅把 A0 的 request_budget 从 1 改为 2；其余相邻候选/交付参数固定 2/1。M 无 Validator；V0 校验重执行 0、V2 为 2，同一提交规则。V0 的状态处理可能影响后续交互，不是纯观测。

用户允许 **OPENROUTER_API_KEY_1 和 OPENROUTER_API_KEY_2 同时启动两批**，每槽至多一批，单批 episode 并发 16、GP 请求上限 32，两批 GP 合计可达 64。统一任务内原子启动账本，先预留再启动；所有实际启动、失败、取消、重启计入 30。子进程仅在 env 中映射自己的 key 为 OPENROUTER_API_KEY，不将值写文件、argv 或日志；不改服务、不干预他人进程。

assistant 为同一 Qwen3.6-27B non-thinking，simulator 为 deepseek_v4_flash_0731；hard、20 回合、sample_retries=0、no-update-memory；monotonic_satisfaction=false；enforce_controller_prefix_closure=true；上下文总上限 262144、turn timeout 900 秒。生成参数冻结，新角色的继承映射在任务书中明确，不做温度搜索。真实运行从子进程清除 REASON_DAG_DATASET_PATH，避免覆盖配置。

原九次独立确认已用于 prompt 迭代，不补跑 442/542/642；没有新机制搜索或单题探路豁免，也不继承任何历史余额。本次整理未检查凭据有效性、未发起真实模型请求。

## 5. 评价与交付

E/S 越低越好。SUCCESS＋FAILURE_TURN_LIMIT 纳入原正式指标，普通错误 N/A 并另报；未达成记 21，已达 E 不因后续超限抹去。先每批均值，再三次 mean±样本 STD；同 seed 共同有效 ID 配对并注明分母。另报共同成功、固定 ID、超限转移和逐题波动；不能靠删除失败或都超限形成低方差。

校验耗尽交付不自动记成功，也不从指标排除。审阅真实候选、归因、误判、修复差异与后续反馈，另列内部调用/两类恢复/token/延迟。可见 token 是每条 episode 全部实际交付正文之和；缺失 usage 不填零。

最终在工作区写 FINAL_REPORT、STATUS、完整账本/manifest、逐样本结果和配对成员、原始日志、机制审阅、测试及可移植 patch。必须交付实际完整测试过的一个冻结版本；没有合格新方案保留 A0。开发结果不称独立确认或未见题泛化，不自动扩到 292、不恢复 Evo、不向主项目合并。working agent 完成后不回写本 CURRENT_STATE。

## 6. V3 与更早档案

[v3/README.md](v3/README.md) 是 V3 唯一历史索引：V3 任务书、启动说明、重复性/结果分析、补充脚本、原 gp_v3 整体工作区、专用 dev64/test16、DISCUSSION_HANDOFF 已迁入；更新前交接文档和当前 R20 源码/共享数据另留副本。

9 个入口共 9,315 个文件、30 条旧 pytest 软链接迁移前后逐字节/原目标核验一致，详见 [archive_manifest](v3/archive_manifest.json) 和 [verification](v3/verification.json)。原始历史报告与绝对路径未改写，旧路径没有兼容 symlink；读取时按 manifest 重定位，不能直接运行归档执行器。

既有 [V3 结果核验](v3/assistant/v3_results_analysis.md) 已确认 Static 15 批/240 episode、Evo 17 批/368 episode，Static A1/A2/A3 未通过综合验收，Evo 初版比较完成。该核验与归档完整性核验不同：本次没有重新执行旧全量指标、token 或学习流程复算，也未移植任何 V3 成果。

[v2.6](v2_6/README.md) 保留 R20 Format 292 题四次、DAG 修复与当时试验；[v2.5](v2_5/README.md)、[v2](v2/README.md)、[v1](v1/README.md) 均为更早历史。共享 first16、DAG_fixed/dev 全集留原位置；V3 专用 self-evo-test 已移入 v3，V4 不使用它做真实运行。
