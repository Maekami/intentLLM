# CURRENT_STATE — 静态 Agent 冻结为 R1

更新：2026-10-08。以本文件为当前入口；历史任务书与旧实验额度不再生效。用户已选定原始 R1，后续静态系统使用正式包 `assistant.static_r1`；自进化尚未启动，等待新任务。

## 正式代码与运行

- 架构：`(Tracker || Direct) → Intra → Inter → Editor`，四个串行阶段、最多五个角色调用；没有可用 adjacent 时跳过 Inter。
- 入口：`--baseline static_r1`；快捷配置 `assistant/configs/r1.yaml`。`goal_progression` 仍是历史 R20，不能当作 R1。
- 详细信息流、状态、动作空间、运行命令与计数规则见 [R1 架构与运行说明](assistant/R1_ARCHITECTURE.md)。
- 五个 prompt 位于 `assistant/src/assistant/static_r1/prompts.py`，通用策略位于同目录 general_policy.md；它们与原始 R1 保持一致，不加载 Trace2Skill 或演化记忆。
- 原始 R1 快照指纹：fa09d6ccac3de47568bd4b0c69de48c82439abc222a675785c2495c8fbb97744。
- 正式包实现指纹：03f55794dd5b114e6cc3e14399912043b9463054b6efc780b761739072a0256c。角色 prompt/源码指纹会进入运行快照，组件轨迹保存为 r1_turn 审计事件。

安装：`python -m pip install -e './assistant[audit]' -e ./user_simulator -e ./interaction_pipeline -e ./metrics`。

在 assistant 目录：

```bash
OPENROUTER_API_KEY="$OPENROUTER_API_KEY_1" assistant-demo --baseline static_r1 --model-profile gemini_3_6_flash_non_thinking
OPENROUTER_API_KEY="$OPENROUTER_API_KEY_2" assistant-demo --baseline static_r1 --model-profile gpt_5_6_luna_non_thinking
assistant-demo --config configs/r1.yaml
```

Qwen 继续使用本地 qwen_3_6_27b_vllm_non_thinking（8001），沿用原采样与输出预算。Gemini 为 minimal，Luna 为 none；本次未验证 high-thinking 配置。API 模型的采样/输出预算继承各自既有 baseline profile，未套用 Qwen 专属参数。

## 本轮兼容性验收

每模型 10 个合成场景、14 回合通过，另各通过一次正式 factory 合成调用。前 8 条真实 hard 测试结果如下：

| 模型 | Key | 首次 E / S | 超限重试后 E / S | 正文 Avg.token 首次 / 重试后 | 超限 首次 / 重试后 | 真实测试总费用 |
|---|---|---|---|---|---|---|
| gemini | KEY_1 | 5.0000 / 6.0000 | 5.0000 / 6.0000 | 3674.125 / 3674.125 | 0 / 0 | $1.254423 |
| luna | KEY_2 | 5.8750 / 7.1250 | 3.8750 / 5.1250 | 3234.500 / 2978.875 | 1 / 0 | $0.709515 |

无普通错误、无组件降级。Luna 的超限来自缺少对标节目名称时反复询问/猜测的任务推进卡点；运行兼容不等于所有任务成功。首次轨迹完整保留。详细证据在本地 [发布验收报告](runs_batch_qwen_full/r1_release_20261008/R1_RELEASE_REPORT.md)，原始输出不进入 Git。

Avg.token 仅计算实际交付正文，按 episode 求和再平均。本轮跨模型统一用 Qwen tokenizer。生产 API 调用可设 `R1_VISIBLE_TOKENIZER` 指向既有 tokenizer.json；未配置时明确计数未知，不能替换为内部模型输出 tokens。本地 Qwen 可直接使用 tokenize 接口。

29 项 R1 专项测试通过，wheel 独立安装导入通过。完整隔离回归有 25 项可在修改前复现的旧 GP 测试失败、1 项缺少旧 Evo 开发数据的失败；不宣称旧测试全绿。本次不加入无关的 Evo 数据。

## 发布与后续边界

- 默认评测数据为 292 条的 user_simulator/dataset/DAG_fixed.jsonl，SHA256=44accd8edfadd290b223018e50181f953ec72422616903b603d9848e62cc4770。该必要文件及 fast 模型配置纳入发布；不把旧 DAG.jsonl 误当成此次数据。
- 既有通用 batch runner 的重试会覆盖旧尝试。需要保留首次/重试后成绩时必须像本轮独立保存 attempts；普通 batch 演示使用 `--sample-retries 0`。本次只迁移静态 agent 与必要接口，未改写整个实验调度器。
- 普通错误最多 3 次额外重试，超限最多 3 次额外重试；API 错误不消耗这些次数。首次成绩只允许普通错误恢复。重试后的成绩专指再给予超限机会后的结果。
- 不自动启动全量实验、不继续旧开发预算、不启动演化。静态确认后，演化数据才可从冻结版本收集；新的数据划分/模型/额度由后续任务指定。
- 发布使用精确暂存清单；原先未提交的 .gitignore、旧文档删除和 result2.md 留在暂存区外。未自动 commit 或 push。
