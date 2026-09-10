# Simulator × Assistant 自动交互管线

该目录直接组合现有 `user_simulator` 和 `assistant`，提供四个共用同一交互核心的模式：

- 批量模式：异步并发运行多个样本，并限制同时活跃的样本数；单个样本报错后可从头重试。
- Evo-Memory 开发集演化模式：按冻结 memory snapshot 的 mini-batch 并发运行，在批次 barrier
  后按数据集顺序原子更新 memory。
- Trace2Skill 开发集制作模式：1000 条轨迹从同一个空 skill 并行出发，按固定 B32 层次合并
  为一个测试时冻结的静态 skill。
- 在线 Demo：通过浏览器面板实时查看 simulator/user 与 assistant 的完整回复。事件以
  “一条完整回复”为单位发送，不进行 token 级流式输出。

每个样本都会保存 simulator 兼容的 `events.jsonl`、`transcript.jsonl`、
`config_snapshot.yaml` 和 `final_state.json`，并额外生成便于人工检查的 `events.txt` 和
`transcript.txt`。

## 安装

要求 Python 3.11 或更高版本：

```bash
cd interaction_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ../user_simulator -e ../assistant
pip install -e '.[test]'
cp .env.example .env
```

在 `.env` 中填写 OpenRouter Key：

```dotenv
OPENROUTER_API_KEY=sk-or-v1-你的密钥
OPENROUTER_HTTP_REFERER=
OPENROUTER_APP_TITLE=Reason-DAG Interaction Pipeline
VLLM_API_KEY=EMPTY
```

默认组合为：

- simulator：`deepseek/deepseek-v4-flash-0731`。
- assistant：`openai/gpt-5.6-luna`。
- assistant baseline：`base`。
- difficulty：`medium`。

## 不安装项目，直接从源码运行

如果当前 Python 环境已经具备运行时依赖，可以不执行 `pip install -e`，只需通过
`PYTHONPATH` 把三个项目的 `src` 目录加入模块搜索路径。推荐从仓库根目录执行：

```bash
cd /path/to/intentLLM
export INTENTLLM_ROOT="$PWD"
export PYTHONPATH="${INTENTLLM_ROOT}/interaction_pipeline/src:${INTENTLLM_ROOT}/assistant/src:${INTENTLLM_ROOT}/user_simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENROUTER_API_KEY=sk-or-v1-你的密钥
```

其中 `/path/to/intentLLM` 替换为本仓库的实际路径。`PYTHONPATH` 必须同时包含上述三个
`src` 目录，并把它们放在已有搜索路径之前，从而确保运行的是当前工作区源码，而不是
环境中可能存在的旧安装版本。以上环境变量只对当前终端会话生效；新开终端后需要重新设置。

可以用以下命令确认三个包均从当前仓库加载：

```bash
python -c 'import interaction_pipeline, assistant, user_simulator; print(interaction_pipeline.__file__); print(assistant.__file__); print(user_simulator.__file__)'
```

直接启动在线 Demo：

```bash
python -m interaction_pipeline.cli.demo \
  --sample-id user20_task1_conversation1 \
  --baseline base \
  --port 8000
```

直接启动批量模式：

```bash
python -m interaction_pipeline.cli.batch \
  --sample-id user20_task1_conversation1 \
  --sample-id user21_task1_conversation1 \
  --concurrency 2 \
  --sample-retries 3
```

开发集 Evo-Memory 演化使用独立入口和独立配置 `configs/evolution.yaml`：

```bash
python -m interaction_pipeline.cli.evo \
  --all \
  --assistant-model-profile qwen_3_6_27b_exprag \
  --simulator-config /path/to/dev-simulator.yaml
```

assistant 也可以切换到 `assistant/configs/models/` 中预置的本地 vLLM profile。假设
Qwen 服务运行在 `127.0.0.1:8001`，thinking 模式可以这样运行：

```bash
python -m interaction_pipeline.cli.batch \
  --sample-id user20_task1_conversation1 \
  --assistant-model-profile qwen_3_6_27b_vllm \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --baseline base \
  --concurrency 1
```

关闭 thinking 时改用
`--assistant-model-profile qwen_3_6_27b_vllm_non_thinking`。本地 assistant 不消耗
OpenRouter assistant 请求，但当前 simulator 仍通过 OpenRouter 运行，所以完整 pipeline
仍需要 `OPENROUTER_API_KEY`。单卡本地部署建议先使用 `--concurrency 1`，再根据显存余量
和 vLLM 调度能力逐步提高并发数。

直接运行 pipeline 测试：

```bash
python -m pytest interaction_pipeline/tests -q
```

这里的“不安装”是指不安装 `interaction_pipeline`、`assistant` 和 `user_simulator` 三个
本地项目。Python 解释器仍需能够导入这些项目声明的第三方运行时依赖；如果环境中缺少依赖，
源码模式也会报 `ModuleNotFoundError`，此时需要切换到已准备好依赖的 Python 环境。

## 批量并发运行

指定多个样本：

```bash
python -m interaction_pipeline.cli.batch \
  --sample-id user20_task1_conversation1 \
  --sample-id user21_task1_conversation1 \
  --concurrency 2
```

运行期间默认显示以“单个样本完成”为粒度的进度条，包括已完成样本数、成功/失败数、
已用时间和预计剩余时间。并发样本的结果仍按照输入顺序写入 batch summary。如需将输出
交给其他日志系统，可以使用 `--no-progress` 关闭进度条。

显式运行全部样本并限制前 10 个：

```bash
python -m interaction_pipeline.cli.batch \
  --all \
  --limit 10 \
  --concurrency 4
```

`--all` 必须显式提供，避免误操作导致大量模型费用。常用覆盖参数：

```bash
python -m interaction_pipeline.cli.batch \
  --sample-id user20_task1_conversation1 \
  --baseline prompt_base \
  --assistant-model-profile qwen_3_6_27b_non_thinking \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --difficulty hard \
  --max-turns 12 \
  --sample-retries 3 \
  --concurrency 2
```

`--sample-retries N` 表示测试/评估 episode 失败后，最多再完整运行该样本 `N` 次；默认值为
`3`，因此持续失败的样本最多执行 `4` 次。达到 turn limit 的行为失败也会重跑。设为 `0`
可关闭完整 episode 重试。Evo 使用独立策略，达到 turn limit 时不会重跑，只重试基础设施
失败。每次完整重试都会重新构建 episode、assistant session，不会续用失败尝试的会话状态；
同一样本的 seed 在各次尝试中
保持不变。每个样本在批次内使用固定目录；重试前会删除该目录中上一轮的全部结果，再在同一
目录名下从头运行，因此磁盘上只保留最终一次尝试。
批量与 evolution scheduler 会在已接受 assistant 回合数到达预算时直接结束，不会先生成一个
注定被 simulator 拒绝的超预算回复。

每次批处理生成：

```text
runs/<batch-id>/
├── batch_config.yaml
├── batch_summary.json
├── batch_summary.txt
├── <sample-id>_0001/            # 固定样本目录，仅含最终一次尝试
│   ├── config_snapshot.yaml
│   ├── events.jsonl
│   ├── events.txt
│   ├── transcript.jsonl
│   ├── transcript.txt
│   └── final_state.json
└── ...
```

`batch_config.yaml` 会在样本开始执行前写入，记录本次批处理解析后的 pipeline、simulator
和 assistant 配置，包括实际模型 profile 的全部参数、simulator prompt 内容与哈希、当前
assistant baseline 及其生效 prompt、Git commit、并发数、样本重试上限、样本顺序和每个样本
的实际 seed。
该文件不会保存 API key。`batch_summary.*` 继续记录运行结果；因此即使某些样本失败，仍可
用全局配置快照核对该批次实际使用的 simulator 和 assistant。
summary 还把每条结果显式标为 `SUCCESS`、`FAILURE_TURN_LIMIT` 或
`INFRASTRUCTURE_FAILURE`，避免把用尽重试的基础设施问题混入行为失败。

文件开头的 `run_overview` 是便于快速核对的 setting 摘要：依次列出 difficulty、assistant
baseline，以及 simulator 各组件和 assistant 的 profile 名与实际 model ID。后面的字段保存
同一批次的完整参数和 provenance 信息。

批量中的单次样本失败不会取消其他样本。管线会在同一个并发槽位内按重试上限依次重建并
运行该样本，旧尝试的目录会在下一次尝试开始前删除。`batch_summary.*` 不保存历史尝试明细，
每个样本只记录最终状态、最终目录和实际 `retry_count`；批次顶层记录 `total_retries`、
`total_attempts`、完成数、失败数等聚合结果。`runs[].run_dir` 仍指向最终结果，因此现有的
下游汇总逻辑可以继续使用。

测试/评估入口还提供 `--update-memory/--no-update-memory`。默认由 `pipeline.yaml` 中的
`run.update_memory` 控制，并保持历史行为 `true`；如果测试集应只读取开发集演化后的固定
memory，请显式使用：

```bash
python -m interaction_pipeline.cli.batch \
  --all \
  --assistant-model-profile qwen_3_6_27b_exprag \
  --no-update-memory
```

### 只重跑既有评估批次中的失败样本

对于已经完成且使用 `--no-update-memory` 的评估批次，可以只重跑最终失败的样本：

```bash
python -m interaction_pipeline.cli.retry_failed \
  /path/to/existing/batch \
  --max-additional-attempts 3 \
  --concurrency 8
```

如果只希望选择最终分类为基础设施失败的样本，而保留 turn-limit 等行为失败结果，请增加：

```bash
python -m interaction_pipeline.cli.retry_failed \
  /path/to/existing/batch \
  --infrastructure-failures-only \
  --max-additional-attempts 3 \
  --concurrency 8
```

该入口从原 `batch_config.yaml` 恢复 dataset、difficulty、simulator/assistant profile 和每条
样本的原始 seed。原批次已经执行过一次，因此 `--max-additional-attempts 3` 表示最多再调用
三次，不会额外变成“新初次运行 + 三次重试”。旧批次及其成功样本保持不变；新结果默认写入
`SOURCE_BATCH/repairs/<batch-id>/`。新目录的标准 `batch_summary.json` 合并引用原成功结果和
新失败样本结果，仍然恰好包含原来的完整样本数，可以直接交给 `intent_metrics`，不会重复计数。
`retry_only_summary.json` 则单独保留本次实际调用的失败样本记录。

为了避免污染冻结 memory，该入口拒绝恢复曾以 `update_memory: true` 运行的批次；同时会检查
dataset、memory 的 SHA-256 和当前模型 profile 是否仍与原批次快照一致。旧版 batch 尚未记录
memory SHA-256 时，仅在 memory 文件修改时间早于原批次开始时间时允许恢复。新版 batch 会在
`assistant.memory_store` 中直接记录内容哈希。

## 浏览器实时 Demo

启动服务：

```bash
python -m interaction_pipeline.cli.demo \
  --sample-id user20_task1_conversation1 \
  --baseline prompt_base \
  --port 8000
```

打开 `http://127.0.0.1:8000`。页面会自动开始一次交互，并按以下粒度更新：

1. simulator 生成一条完整用户消息后显示；
2. assistant 生成一条完整回答后立即显示；
3. simulator 在后台完成 controller、satisfaction 和下一轮 user realization；
4. 重复直到 Episode 终止或失败。

其中 controller 只负责 intent 节点暴露。END 由 simulator 系统在更新 frontier 后根据 DAG
结构推导；END 暴露后不再调用 controller，但会继续 satisfaction 与必要的 user realization，
直到所有已暴露 intent 都 satisfied 才终止。

点击“重新运行”会为同一样本创建一个全新的会话和审计目录。即使浏览器断开，已经启动
的后台交互仍会继续完成并保存记录。

## 配置

默认管线配置位于 `configs/pipeline.yaml`。它指向相邻的 `user_simulator` 和 `assistant`
目录，并设置 difficulty、seed、并发数、样本重试次数、输出目录和 audit level。

CLI 支持额外配置：

- `--pipeline-config`：覆盖管线设置。
- `--simulator-config`：传给 simulator 的自定义配置。
- `--assistant-config`：传给 assistant 的自定义配置。
- `--simulator-model-profile`：一次覆盖 simulator 的四个 LLM 组件；启用 React action guard
  时，这一份 YAML 也是 guard 模型身份、provider、routing、reasoning 和 retry 的明确来源。
- `--assistant-model-profile`：覆盖 assistant 模型。
- `--baseline`：选择 `base`、`prompt_base`、`interactcomp_react` 或 `trace2skill`。`base`
  不添加 Prompted Base，但不会关闭模型 profile 自己绑定的 Trace2Skill；
  `--trace2skill-skill` 仅用于不带 skill binding 的旧式/临时 profile。
- `--react-action-guard` / `--no-react-action-guard`：开启或关闭 InteractComp ReAct 的
  第二层 LLM 动作边界校验；关闭后只保留 prompt + schema/parser 第一层约束。
- `--sample-retries`：覆盖单个样本失败后的最大完整重试次数，`0` 表示不重试。

如需让 simulator 的四个组件只使用 DeepSeek 官方 provider，同时不修改默认的
Baidu-first fallback profile，可在测试集或 evo 命令中加入：

```bash
--simulator-model-profile deepseek_v4_flash_0731_official
```

该覆盖只影响 simulator；assistant 的模型、memory 方法和输出目录保持不变。由于 DeepSeek
官方 endpoint 不提供原生 JSON Schema enforcement，这个 profile 使用 JSON Object 传输，
但仍注入完整 Schema 并执行同一套 strict 本地验证和结构化重试。

管线会把项目内的 dataset、prompt 和 model profile 路径解析为绝对路径，因此运行期间
无需在两个项目目录之间切换工作目录。

如果 simulator 使用在 `127.0.0.1:8005` 部署的 Qwen3.8-27B，可传入：

```bash
python -m interaction_pipeline.cli.batch \
  --simulator-model-profile qwen_3_8_27b_vllm \
  <其他参数>
```

该本地 simulator profile 不要求 `OPENROUTER_API_KEY`；assistant 是否需要该变量取决于
assistant 自己选用的 model profile。

可用的 assistant Qwen profile 对应关系如下：

| 调用方式 | Thinking | Profile |
| --- | --- | --- |
| OpenRouter | 开 | `qwen_3_6_27b` |
| OpenRouter | 关 | `qwen_3_6_27b_non_thinking` |
| 本地 vLLM | 开 | `qwen_3_6_27b_vllm` |
| 本地 vLLM | 关 | `qwen_3_6_27b_vllm_non_thinking` |

## InteractComp ReAct assistant

该条件使用 `--baseline interactcomp_react`，动作空间只有非终止的 `ask` 和 `answer`。它不挂载
memory/skill，不使用 search、probing budget 或 final-round forced answer。`answer` 后如果系统尚未
满足 END 条件，下一条用户消息仍会触发新一轮动作选择；第 20 个 accepted assistant turn 后仍未
自然终止则沿用现有 `FAILURE_TURN_LIMIT` 语义。

第二层 action guard 默认开启。它使用 `--simulator-model-profile` 指定的精确 YAML 创建独立
client，不复用 assistant profile。若省略该参数，仅当 simulator 的四个模型路径完全相同时才能
安全推断；不一致时会要求显式传参。hard 模式仍由 `realizer_abstract` 生成可见用户回复，guard
只共享该 YAML 的模型配置，不会改走 `realizer_clear`。

每个可见 assistant turn 有 10 个本地语义 action slot。每个 slot 内，schema/parser/action-name
错误按“首次 + 最多 3 次格式重试”修复，不占新 slot；一个 schema 合法候选被 guard 拒绝就会
消耗一个 slot，并以私有 action + `ask_invalid` / `answer_invalid` observation 继续 ReAct。第 10
个候选合法时照常接受，只有它也被拒绝才产生 typed episode failure；不会向 simulator 提交无效
payload 或固定 fallback 话术。显式传 `--no-react-action-guard` 时不加载或调用判定器，并退化成
仅有 prompt + schema/parser 的第一层版本。

为避免 guard 介入 agent 的对话策略，`ask` 校验只接收当前候选 question，不接收对话历史；
`answer` 校验保留可见对话，但只用它判断问题或请求是否要求当前用户回复。guard 不判断候选的
相关性、重复性、必要性、完整性或对话推进效果。每个 schema 合法候选只进行一次独立 validator
判定，默认不复核拒绝，也不缓存相同 payload 的 verdict；无法解析 validator 输出时与官方一致按
拒绝处理并消耗当前 slot。provider/transport 调用本身最终失败仍按管线统一错误语义处理。

三个对齐 profile 为 `qwen_3_6_27b_react`、`gpt_5_6_luna_react` 和
`gemini_3_6_flash_react`。batch 参数与其他正式 baseline 一致，下面以 Qwen/easy 为例；替换
profile、difficulty 和输出目录即可运行另外两种模型或难度：

```bash
python -m interaction_pipeline.cli.batch \
  --all \
  --assistant-model-profile qwen_3_6_27b_react \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --baseline interactcomp_react \
  --react-action-guard \
  --difficulty easy \
  --seed 42 \
  --concurrency 8 \
  --sample-retries 3 \
  --no-update-memory \
  --max-turns 20 \
  --output-dir runs_batch/qwen_3_6_27b_react/easy
```

试运行与其他 baseline 完全相同，可在上述命令的 `--all` 后增加 `--limit 8`（或任意正整数）；
guard 开关不会改变样本选择、20-turn budget 或 batch/sample retry 逻辑。

动作 JSON 和 hidden reasoning 不进入可见 transcript。审计中的 `interactcomp_action`、
`interactcomp_confidence`、`interactcomp_semantic_action_slots_used`、
`interactcomp_semantic_action_budget_exhausted`、`internal_call_count`、
`interactcomp_agent_call_count`、`interactcomp_guard_call_count`、
`interactcomp_format_rejection_count`、`interactcomp_guard_rejection_count`、
`interactcomp_guard_vote_count`、`interactcomp_guard_disagreement_count`、
`interactcomp_guard_cache_hit_count` 与 `interactcomp_guard_decisions` 可用于区分候选生成、语义校验
和 validator 波动。预算耗尽时同一份元数据写入 `assistant_generation_failed`。batch config
snapshot 还会冻结 guard 开关、prompt hash、共享 generation、精确 simulator model profile 及来源。
详细协议及官方实现差异见
`assistant/README_CN.md`。

## Evo-Memory assistant

assistant 还提供论文 *Evo-Memory* 中的 `ExpRAG` 和 `ReMem`。它们不是新的 simulator
difficulty，也不替代 `base` / `prompt_base`；模型 profile 的 `memory` 块负责挂载记忆
framework，`--baseline` 仍负责选择底层 assistant system prompt。预置示例：

```bash
python -m interaction_pipeline.cli.batch \
  --all --limit 30 \
  --assistant-model-profile qwen_3_6_27b_exprag \
  --baseline base \
  --concurrency 1

python -m interaction_pipeline.cli.batch \
  --all --limit 30 \
  --assistant-model-profile qwen_3_6_27b_remem \
  --baseline prompt_base \
  --concurrency 1
```

管线把一个完整样本视为一个 Evo-Memory task，但在每次可见 user→assistant 交互前重新检索。
query 由截至当前轮已显式出现的用户消息组成，最新请求优先；同一 episode 始终在其冻结
memory bank 上检索。样本正常终止后把完整对话、最终回复和终止反馈写回，默认不保存失败或
超过 turn limit 的样本。严格的 test-time learning 任务流应使用 `--concurrency 1`；并发模式
的文件写入仍然安全，但样本间检索和更新顺序会交错。完整配置字段、embedding 检索可选依赖、
本地 vLLM 组合方式和记忆 JSON 格式见 `assistant/README_CN.md`。

### 独立开发集 evolution

开发集 memory evolution 不复用上述测试 batch scheduler，而使用独立入口：

```bash
python -m interaction_pipeline.cli.evo \
  --all \
  --assistant-model-profile qwen_3_6_27b_exprag
```

默认 evo 数据由下面的离线脚本一次性生成。它读取筛选后的 1000 条 `direct/DAG.jsonl`，
用 `seed + sample_id` 的 SHA-256 排名把样本稳定地划分为 `easy=334`、`medium=333`、
`hard=333`；写文件时仍严格保持源 JSONL 行序，不会按难度重排：

```bash
python user_simulator/dataset/thoughttrace_3plus_no_dag/annotation_runs_full/\
gpt_5_6_sol_reason_dag_v7/assign_evolution_difficulties.py
```

脚本默认使用 `--seed 42`，也可显式传入其他整数。目标文件已经存在时脚本会拒绝覆盖；
确认需要重新划分时使用 `--force`。它在 `user_simulator/dataset/` 下生成：

- `DAG_evolution_dev_original.jsonl`：源数据的逐字节副本；
- `DAG_evolution_dev.jsonl`：保持原顺序，并在每条样本顶层加入固定 `difficulty`；
- `DAG_evolution_dev_metadata.json`：记录划分 seed、算法、数量和各文件 SHA-256。

ExpRAG 和 ReMem 都从同一个处理后文件读取，因此两种设置使用完全相同的
`sample_id + difficulty` 对。难度分配只在运行前由该脚本确定，evo 运行期间没有重新随机
划分逻辑。

默认参数来自带逐字段备注的 `configs/evolution.yaml`：

```yaml
dataset:
  path: ../user_simulator/dataset/DAG_evolution_dev.jsonl
  original_path: ../user_simulator/dataset/DAG_evolution_dev_original.jsonl
  metadata_path: ../user_simulator/dataset/DAG_evolution_dev_metadata.json
  difficulty_field: difficulty
  expected_sample_count: 1000
  expected_difficulty_counts:
    easy: 334
    medium: 333
    hard: 333
run:
  mini_batch_size: 8
  concurrency: 8
  sample_retries: 0
  seed: 42
  max_turns: null
  output_dir: runs_evolution
  group_output_by_profile: true
```

evo 启动模型调用前会完整校验 1000 条处理后数据：样本 ID 唯一、三种难度合法且数量符合
配置、处理版与原始副本逐行同序、除 `difficulty` 外样本内容完全一致，并核对元数据中的
文件哈希。任一条件不满足都会直接终止，避免 ExpRAG 与 ReMem 静默使用不同的数据版本。

每个 mini-batch 开始时只读取一次 memory snapshot。批内样本无论先后完成、内部经历多少轮
user→assistant 交互，都只能在该 snapshot 上重新计算 per-turn query 和 top-k，绝不会读取
本批刚产生的 memory。全部结束后，成功 episode 的独立 `MemoryEntry` 按开发集顺序通过一次
文件锁和一次原子替换提交。下一批随后读取更新后的 memory。若运行期间有其他进程修改同一
memory 文件，barrier commit 会拒绝覆盖并终止 evolution，以避免静默混合两个实验流。

CLI 的 `--mini-batch-size`、`--concurrency`、`--sample-retries`、`--seed`、
`--max-turns` 和 `--output-dir` 可以覆盖 evolution YAML。`concurrency` 不得大于
`mini_batch_size`。默认 `sample_retries: 0` 表示每条开发样本只运行一个完整 episode；模型
profile 内部的传输重试和 ReMem Think/Prune 迭代不属于 full-sample retry。

这里的 `run.seed` 只控制 simulator 的样本内随机行为，不参与难度划分；每条样本使用
`run.seed + 原始数据集零基下标`。`--all`、`--limit` 和 `--sample-id` 都不会重新分配难度；
指定多个 `--sample-id` 时也会恢复为原始 JSONL 顺序后再运行。

每次运行生成独立的 `evo_<id>/` 目录，其中 `evolution_config.yaml` 保存完整解析配置、样本
顺序、每条样本的实际 difficulty/seed、数据文件哈希、模型/prompt、memory 路径及初始
条目数，`evolution_summary.*` 保存相同的数据 provenance，以及每个 mini-batch 的 memory
前后条目数、候选数和实际写入数。

### 六组模型 × memory 方法并行运行

默认 `group_output_by_profile: true` 会把 `run.output_dir` 当作输出根目录，并使用解析后的
memory-enabled assistant `profile_name` 自动追加“模型+方法”子目录。六组输出结构为：

```text
interaction_pipeline/runs_evolution/
├── qwen_3_6_27b_exprag/evo_<id>/
├── qwen_3_6_27b_remem/evo_<id>/
├── gpt_5_6_luna_exprag/evo_<id>/
├── gpt_5_6_luna_remem/evo_<id>/
├── gemini_3_6_flash_exprag/evo_<id>/
└── gemini_3_6_flash_remem/evo_<id>/
```

`--output-dir /path/to/root` 只覆盖这里的根目录，profile 子目录仍会自动追加。确实需要旧的
扁平布局时，可在自定义 evo YAML 中设置 `group_output_by_profile: false`。六个 profile 的
persistent memory 也分别写入 `assistant/runs/memory/<profile_name>.json`，不会跨条件共享。
六个 Evo profile 的思考模式与生成参数均对齐各自底座的 non-thinking profile；其中 Gemini
必须保留 thinking，因此采用 `effort: minimal` 作为 non-thinking 近似。ExpRAG 与 ReMem
只改变 memory 方法，不改变同一底座模型的这些参数。

从仓库根目录、完成前文 `PYTHONPATH` 和 `OPENROUTER_API_KEY` 设置后，可以用下面的 Bash
循环同时启动六个独立进程：

```bash
profiles=(
  qwen_3_6_27b_exprag
  qwen_3_6_27b_remem
  gpt_5_6_luna_exprag
  gpt_5_6_luna_remem
  gemini_3_6_flash_exprag
  gemini_3_6_flash_remem
)

mkdir -p interaction_pipeline/runs_evolution
pids=()
for profile in "${profiles[@]}"; do
  python -m interaction_pipeline.cli.evo \
    --all \
    --assistant-model-profile "$profile" \
    --no-progress \
    > "interaction_pipeline/runs_evolution/${profile}.launcher.log" 2>&1 &
  pids+=("$!")
done

printf 'Started evo processes: %s\n' "${pids[*]}"
wait "${pids[@]}"
```

默认每个进程内部 `concurrency: 8`，六组并行时理论上最多会同时活跃 48 个 episode，并且
ReMem 的一个可见回复可能包含多次内部模型调用。如果 OpenRouter 配额、本地 vLLM 吞吐或
本机资源不足，建议先给每个命令增加 `--concurrency 2`；`mini_batch_size` 仍可保持 8，
因此 memory barrier 语义不变。

六个预置 Evo-Memory profile 使用官方检索设置
`BAAI/bge-base-en-v1.5 + cosine similarity + top_k=4`。BGE 固定运行在 CPU；每个 evo 进程
独立加载一次模型，并跨样本缓存文本 embedding。同一 mini-batch 的八条样本共享冻结
memory snapshot 对应的归一化 embedding index，不会各自重新编码全部历史 memory。首次
并行运行前应先安装 assistant 的 `memory-semantic` extra 并完成一次 BGE 权重下载：

```bash
pip install -e './assistant[memory-semantic]'
python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")'
```

## Trace2Skill-Visible-Combined-Parallel-B32

本实现以 [Trace2Skill 官方仓库固定提交 `3d0b52a`](https://github.com/Qwen-Applications/Trace2Skill/tree/3d0b52a140f002a512930252b613c49048f7d5ac)
的 combined parallel JSON pipeline 为对齐基准。它使用与 ExpRAG/ReMem 相同的冻结 1000 条
开发集，不修改数据文件，也不挂载 memory；rollout、分析、Skill 制作和测试均使用该目标
assistant 自己的同一模型 profile 及 generation/reasoning/retry 设置。

与官方方法相比，只保留四项任务必需适配：

- analyst 只能看到有序的可见 user/assistant dialogue；二元 outcome 只用于选择 success 或 failure
  analyst。Reason-DAG、difficulty、controller/satisfaction、simulator prompt、隐藏 reasoning、样本
  ID、最终 state 和评估解释均不可见；
- `SUCCESS` 表示在一至二十个已接受 assistant 回合内自然完成，`FAILURE_TURN_LIMIT` 表示二十
  回合仍未完成。其他错误属于基础设施失败，只在采集阶段按既定策略重试；
- 为与无 Prompted Base 的 ExpRAG/ReMem 对齐，开发 rollout 和 Skill 制作的冻结初始
  `SKILL.md` 都为空；
- intentLLM 运行时把一个非空 Markdown `SKILL.md` 原样作为唯一 system message，因此不使用
  Agent Skills YAML frontmatter、references、scripts 或 assets；首层 merge batch 固定为需求指定的
  B32，而不是官方默认 B5。

Build 的方法流程与官方实现对齐如下：

1. success/failure analyst 对每条可见轨迹各执行一次分析，输出官方格式的 Lean Solution Path /
   Success Memory Items 或 Failure Cause / Failure Memory Items；官方 parser 将报告转为 analysis
   record。解析不到 item 或单次请求失败时只排除该条，不把整批标为 pending；
2. 每个有效 analysis record 独立执行一次 MAP，针对同一个冻结空 skill 生成官方 JSON
   `reasoning + edits + changelog_entries` patch；MAP batch size 为一。官方的 create/link 配对检查
   在本任务中收窄为单文件布局检查：外部文件 edit 及其会造成悬空 reference 的链接 edit 在
   REDUCE 前成对排除，原始 MAP 响应仍完整保留供审计；
3. patch 输出不完整时最多续写两轮；完整但不可解析时最多进行两轮携带 parser feedback 的格式
   修复，并保留官方的窄范围 malformed-JSON recovery；
4. 与官方 combined runner 一致，进入 REDUCE 前先放失败 records、再放成功 records，各类内部
   按 sample id 的字典序排列，并排除 MAP 请求或解析失败的项。随后按固定三十二个有效 patch
   分组；当一千条都成功生成 patch 时，
   首层形成三十一组三十二个和一组八个，共三十二个输出 patch，再携带同一个冻结初始 skill
   做下一层 merge。任一 merge 请求/解析失败时保留该组原 patch 继续；最多五层后仍有多个
   patch 时执行官方 forced merge，失败则采用第一个剩余 patch。若 fallback 导致需要更多常规
   层，后续分组沿用官方对小尾组进行 round-robin 再均衡的 `chunk_list`；
5. 对最终 patch 的每个 edit 独立执行 translation，使文本定位与冻结初始文件精确匹配；translation
   失败时沿用原 edit。随后仅做单文件布局所需的路径/op 过滤；
6. APPLY 不再调用 LLM，而是按官方实现用 Python 确定性地依序应用 edit；之后运行任务适配的
   validator（`SKILL.md` 非空且不超过五百行）。验证失败时最多执行三轮官方式
   validation-error → patch → programmatic apply 修复。

因此旧版的 `PATCH/ABSTAIN` semantic patch、额外全局 leakage validator、三次盲重试和
LLM 整文件 APPLY 均已移除。信息边界由输入构造和不可变 corpus 保证，不再用会误杀正常文本的
后置内容过滤器替代官方 parser 语义。

三个正式条件各自使用一一绑定的 profile：

- `qwen_3_6_27b_trace2skill`
- `gpt_5_6_luna_trace2skill`
- `gemini_3_6_flash_trace2skill`

它们分别绑定独立的 `assistant/runs/skills/<profile_name>/SKILL.md`。Collection 会关闭 profile
skill attachment，即使该路径已有旧 Skill，rollout 仍是 No Skill、No Prompted Base。

### 阶段一：采集并封存 trajectory

正式 Collection 只运行交互和封存，不创建分析/蒸馏 client：

```bash
python -m interaction_pipeline.cli.trace2skill collect \
  --assistant-model-profile qwen_3_6_27b_trace2skill \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --trajectory-concurrency 16 \
  --output-dir runs_batch_qwen_full/trace2skill
```

输出位于
`<output-dir>/<profile_name>/trajectory_collections/trace2skill_trajectories_<id>/`。每条样本保留
全部 episode 尝试和原子 `trajectory_slots/<index>.json`；一千条全部可评分后才生成
`corpus.json`。seal 记录逐条 trajectory hash、整体 corpus hash、数据集 hash 和 assistant
行为配置 hash。`summary.json` 明确记录 `distillation_call_count: 0`。

采集阶段也支持 `--limit N`（`--smoke-limit` 是兼容别名）审计前 N 条：

```bash
python -m interaction_pipeline.cli.trace2skill collect \
  --assistant-model-profile qwen_3_6_27b_trace2skill \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --trajectory-concurrency 16 \
  --limit 64
```

该 trial 仍生成完整中间审计文件，但 seal 标记 `skill_build_allowed: false`，不能进入 Build 或
测试集评估，也不能原地扩展为正式 Collection。

进程中断时增加 `--resume <collection-dir>`；已完成 slot 不会重跑。真正耗尽三次基础设施尝试的
slot 保持 pending，只有显式授权后才会继续：

```bash
python -m interaction_pipeline.cli.trace2skill collect \
  --assistant-model-profile qwen_3_6_27b_trace2skill \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --trajectory-concurrency 16 \
  --output-dir runs_batch_qwen_full/trace2skill \
  --resume /absolute/path/to/trace2skill_trajectories_<id> \
  --retry-exhausted
```

`--retry-exhausted` 每次为已耗尽 slot 授权另一批三次 infrastructure-only 尝试，保留旧 attempt、
错误和目录；全部完成并生成 `corpus.json` 后 Build 才会放行。

### 阶段二：从完整 corpus 试运行 Build

以下命令只读完整 corpus 的前六十四条，不会再次运行 Assistant 或 Simulator：

```bash
python -m interaction_pipeline.cli.trace2skill build \
  --assistant-model-profile qwen_3_6_27b_trace2skill \
  --trajectory-run /absolute/path/to/trace2skill_trajectories_<id> \
  --analysis-concurrency 16 \
  --merge-concurrency 16 \
  --limit 64
```

Build 会先重新验证完整一千条 corpus 及当前模型行为，再选择源顺序前 N 条。trial 隔离在
`<profile_name>/skill_builds/trial_n<N>/trace2skill_build_trial_n<N>_<id>/`，生成
`SMOKE_ONLY.md` 与 `SMOKE_SKILL.md`，但不发布、不生成可用于正式评估的绑定。

每个 run 保留：

- `trajectory_analysis/*.json`：analyst 输入 hash、原报告、parser record、调用与排除原因；
- `map_patches/*.json`：每条 record 的 MAP 对话、原响应、修复轮次和 patch；
- `merge_levels/level_*/group_*.json`：每组输入 hash、输出 patch 或官方 fallback；
- `final_patch.json`、`translation/edit_*.json`、`translated_final_patch.json`；
- `verification/round_*.json`、`apply.json`、`candidate_skill.md` 和最终 trial/canonical skill；
- `manifest.json`：官方 commit、prompt/实现 hash、对齐项与必要适配；`summary.json`：实际调用数、
  排除数、artifact inventory、最终 hash 和 `interaction_call_count: 0`。

Build resume 只接受新版 official-aligned manifest、同一 corpus/prefix、模型行为、prompt 与方法参数；
完成的阶段 artifact 会直接复用：

```bash
python -m interaction_pipeline.cli.trace2skill build \
  --assistant-model-profile qwen_3_6_27b_trace2skill \
  --trajectory-run /absolute/path/to/trace2skill_trajectories_<id> \
  --limit 64 \
  --resume /absolute/path/to/trace2skill_build_trial_n64_<id>
```

旧自定义 build artifact 与新版 schema 不兼容，但已封存 Collection 无需重采，可直接启动新的
Build。原先无子命令的一体化 legacy evolution 入口已移除，以保证 collect/build 解耦且只有一套
可调用的实现。

### 正式构建、发布与测试

省略 `--limit` 即使用完整一千条：

```bash
python -m interaction_pipeline.cli.trace2skill build \
  --assistant-model-profile qwen_3_6_27b_trace2skill \
  --trajectory-run /absolute/path/to/trace2skill_trajectories_<id> \
  --analysis-concurrency 16 \
  --merge-concurrency 16
```

完整成功后，run-local `SKILL.md` 会原子发布到该 profile 的固定 Skill 路径，并生成
`SKILL.provenance.json` 与 run-local `published_skill.json`。不同模型使用各自 corpus、profile 和
Skill 路径，互不读写。

测试时选择同一个 profile。显式 `--baseline base` 只表示不叠加 Prompted Base；profile-bound
Trace2Skill 仍是唯一 system message：

```bash
python -m interaction_pipeline.cli.batch \
  --all \
  --assistant-model-profile qwen_3_6_27b_trace2skill \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --baseline base \
  --difficulty easy \
  --seed 42 \
  --concurrency 8 \
  --sample-retries 3 \
  --max-turns 20 \
  --no-update-memory \
  --output-dir runs_batch/qwen_3_6_27b_trace2skill/easy
```

测试 batch 的配置快照记录测试数据 hash、静态 skill 内容及 hash；Skill 制作阶段不读取测试集，
测试阶段也不读取开发轨迹或更新 skill。

## 审计语义

`events.jsonl` 保留 simulator 原有事件，并增加：

- `pipeline_run_started` / `pipeline_run_completed` / `pipeline_run_failed`；
- `assistant_generation_requested` / `assistant_generation_completed`。
- 使用 Evo-Memory 时还会记录 `assistant_memory_updated`；持久化失败时记录
  `assistant_memory_update_failed` 并将该样本标为失败；因失败样本或空会话而不满足保存条件
  时记录 `assistant_memory_update_skipped`。独立 evo 入口在 barrier 前记录
  `assistant_memory_update_deferred`；测试使用 `--no-update-memory` 时记录
  `assistant_memory_update_disabled`。

`transcript.jsonl` 只包含实际可见的 user/assistant 对话。Qwen thinking profile 的隐藏
reasoning 不写入 transcript，也不会复制到管线事件中。`events.txt` 是结构化事件的缩进
版，`transcript.txt` 则按轮次和角色排版。

## 离线测试

```bash
PYTHONPATH=src:../assistant/src:../user_simulator/src pytest -q
```

测试使用 simulator mock 和假的 assistant，不访问 OpenRouter，也不会产生费用。

## Goal-Progression 接入

现有 batch/demo 支持 `--baseline goal_progression`，配合 assistant 的六份
`qwen_3_6_27b_gp*.yaml`。完整配置和 smoke 命令见
[assistant README](../assistant/README_CN.md#goal-progression-静态系统)。
batch snapshot 的 `assistant.goal_progression`、episode snapshot 的
`assistant_goal_progression` 包含有效 variant、角色参数和全部启用提示的内容/hash。
成功及失败生成事件保留 `llm_call.goal_progression` 的私有调用审计；不向 assistant
传递 simulator 隐藏状态，也不把角色计划写进可见 transcript。

GP 各阶段还会即时写入独立事件（full/summary 均保留关键输出）：

- `assistant_{tracker/intra/inter/joint/generator}_requested` / `_raw_result`：
  角色/尝试号、原始 `raw_output`、用量和耗时；传输失败或取消对应 `_failed` / `_cancelled`。
- 结构化角色的 `_validated` / `_validation_failed`：`structured_result` 或校验错误。
- `assistant_gp_fallback_applied` / `assistant_gp_assembly_completed`：回退原因、有效计划及组装取舍。
- `assistant_gp_state_committed` / `assistant_gp_turn_failed`：本地事务结果、状态、最终回复和可见 tokens。

这些私有输出只写 AuditLogger，不发送到 UI 消息流或 simulator 可见对话；原聚合字段仍保留，
指标不重复计算。旧批次不会自动回填事件，可从 `llm_call.goal_progression` 读取已有输出。

新实验显式指定 `--sample-retries 0 --no-update-memory --concurrency 1`，
Hard、max-turns=20、同样样本顺序和 seed。`--no-update-memory` 不禁用会话内 Tracker。
已有 16 样本真实试运行，但组件回退率很高，尚无超过 Prompted Base 的结论；详见
[试运行分析](../assistant/gp_test_analysis.md)。新增审计代码通过 274 项离线回归，未重跑实验。
