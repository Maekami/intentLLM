# Reason-DAG 评估指标

该目录以仓库根目录 `README.md` 为定义指导，并适配当前 `assistant`、
`user_simulator` 和 `interaction_pipeline` 的真实落盘格式，实现四项 episode 级指标：

1. `Avg. All-Node Exposure Turns ↓`
2. `Avg. All-Node Satisfaction Turns ↓`
3. `Avg. Tokens ↓`
4. `AITR ↑`

实现是纯评估层，不会改变 assistant 回复、Controller 决策、Satisfaction Updater 状态或
user simulator 的后续行为。

## 与当前实现的适配

- 从每个 run 的 `config_snapshot.yaml` 读取 `simulator.dataset_path`，再通过
  simulator 的 canonical `DatasetLoader` 定位样本；也可用 `--dataset` 显式覆盖。终端节点
  依据 `node_type == "terminal"` 排除。
- exposure 读取 `nodes_exposed`、`satisfaction_requested` 和
  `backbone_node_auto_exposed`；satisfaction 读取每轮 `satisfaction_applied.after`。两项均只
  统计 simulator 已接受的 assistant turn，固定预算为 20，未完成记 21。
- Token 优先累计 `assistant_generation_completed.payload.llm_call.output_tokens`。当前
  assistant client 的 `output_tokens` 已包含 reasoning/thinking tokens；缺失时才回退到
  `answer_tokens + thinking_tokens`，并在 episode record 中记录 warning。
- 当前 pipeline 达到 20 轮后，可能先生成一次标号为 21 的回复，再由 simulator 抛出
  `EpisodeTurnLimitError`。这次生成没有 `assistant_message_received`，也不在自然 transcript
  中，因此不属于评估 turn，Token 指标会明确排除它。预算耗尽时 simulator 还可能保存一条
  未被 assistant 消费的末尾 user message；AITR 会在最后一条已接受的 assistant 回复处截断，
  避免把“没有可用回复轮次”误判为 assistant 的互动策略问题。
- AITR 只接收 `transcript.jsonl` 中可见的 user/assistant 消息，不接收难度、DAG、状态、
  模型身份或 token 信息。评审复用 simulator 已有的 OpenRouter strict JSON Schema client，
  因而沿用其传输重试和结构化结果校验逻辑。Judge 的模型、reasoning、generation、routing
  和 retry 参数统一位于 `configs/aitr.yaml`；完整 rubric 与 conversation 模板位于
  `configs/prompts/aitr.yaml`，Python 源码不再内嵌提示词正文。

## 安装

```bash
cd /path/to/intentLLM
pip install -e ./user_simulator -e './metrics[test]'
```

也可以直接从源码运行：

```bash
export INTENTLLM_ROOT="$PWD"
export PYTHONPATH="${INTENTLLM_ROOT}/metrics/src:${INTENTLLM_ROOT}/user_simulator/src${PYTHONPATH:+:${PYTHONPATH}}"
```

## 使用

完整计算四项指标（AITR 会调用 OpenRouter）：

```bash
export OPENROUTER_API_KEY=sk-or-v1-你的密钥
python -m intent_metrics \
  interaction_pipeline/runs_batch/<batch-id>
```

只计算三个确定性指标，不进行外部 API 调用：

```bash
python -m intent_metrics \
  interaction_pipeline/runs_batch/<batch-id> \
  --skip-aitr
```

输入既可以是一个包含 `events.jsonl` 的 run 目录，也可以是包含
`batch_summary.json` 的 batch 目录。常用覆盖项：

```bash
python -m intent_metrics <run-or-batch-dir> \
  --dataset user_simulator/dataset/DAG.jsonl \
  --output-dir /path/to/results \
  --cache /path/to/aitr_cache.json \
  --aitr-config /path/to/custom_aitr.yaml \
  --aitr-model openai/gpt-5.6-luna \
  --aitr-reasoning-effort high
```

## AITR Judge 配置

默认配置是 `metrics/configs/aitr.yaml`，结构与 simulator/assistant 的模型配置一致：

```yaml
profile_name: aitr_judge
provider: openrouter
model_id: openai/gpt-5.6-luna
base_url: https://openrouter.ai/api/v1
prompt: configs/prompts/aitr.yaml

routing:
  require_parameters: true
  allow_fallbacks: true

reasoning:
  enabled: true
  effort: high
  exclude_from_response: true

generation:
  aitr:
    temperature: null
    max_completion_tokens: 4096

structured_output:
  type: json_schema
  strict: true
  require_parameters: true

retry:
  max_attempts: 3
  initial_backoff_seconds: 1.0
  maximum_backoff_seconds: 8.0
```

默认 prompt 文件使用与 simulator/assistant prompt 相同的 YAML 组织方式：

```yaml
name: aitr
version: v1
system: |-
  ...完整 AITR evaluator rubric...
user_template: |-
  <conversation>
  {conversation}
  </conversation>
```

`user_template` 必须且只能包含 `{conversation}` 占位符。加载器会校验字段、占位符和非空
内容，并根据完整 prompt 内容生成 SHA-256。`configs/aitr.yaml` 中的相对 prompt 路径以
`metrics/` 为根目录解析；自定义 prompt 也可以使用绝对路径。

`--aitr-config` 接受一个部分或完整 YAML，并与默认配置做深度合并。覆盖顺序为：

```text
默认 configs/aitr.yaml < 自定义 --aitr-config < 环境变量 < CLI 单项参数
```

API Key 和 OpenRouter 请求头不属于模型配置，仍仅从环境读取，避免密钥进入 YAML。AITR
同时保留以下可选环境覆盖：

```dotenv
OPENROUTER_API_KEY=
AITR_MODEL=openai/gpt-5.6-luna
AITR_REASONING_EFFORT=high
AITR_MAX_COMPLETION_TOKENS=4096
AITR_MAX_ATTEMPTS=3
AITR_INITIAL_BACKOFF_SECONDS=1.0
AITR_MAXIMUM_BACKOFF_SECONDS=8.0
```

AITR cache key 同时包含完整 Judge YAML 的配置哈希、prompt version 和 prompt 内容哈希；
模型、reasoning、generation、routing、retry 或提示词正文发生变化时都不会错误复用旧结果。

## 输出

默认写入 `<input>/metrics_results/`：

```text
metrics_results/
├── episode_metrics.jsonl
├── aggregate_metrics.json
├── aggregate_metrics.txt
└── aitr_cache.json          # 运行 AITR 时生成
```

- `episode_metrics.jsonl` 保留每个 episode 的四项值、AITR 理由、cache 命中状态、warning
  和显式 evaluation error。
- `aggregate_metrics.json` 按 assistant model 和 difficulty 分组，并对 episode 做等权宏平均。
- `aggregate_metrics.txt` 只在主表展示四项指标；failure count 等信息放在 diagnostics 中。
- aggregate 的浮点指标仅在最终输出阶段四舍五入，并按两位小数序列化。

如果任意 AITR 调用在所有重试后仍失败，对应 episode 会保留 `aitr_error`，该难度组的
aggregate `aitr` 为 `null`，不会把失败 episode 从分母中静默删除。trace 本身缺字段、
数据集不可解析或因非轮数预算原因失败时，同样会进入 `evaluation_errors`，CLI 返回非零状态。

## 测试

```bash
PYTHONPATH=metrics/src:user_simulator/src pytest -q metrics/tests
```

测试不访问 OpenRouter。
