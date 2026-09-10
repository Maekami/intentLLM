# Reason-DAG User Simulator：Demo 启动教程

本文只介绍如何安装、配置并启动交互式 Demo。Demo 中，程序扮演用户，
你在终端中扮演 Assistant；用户的潜在意图会依据 Reason-DAG 逐步暴露。

## 1. 环境准备

要求 Python 3.11 或更高版本。请先进入项目根目录：

```bash
cd user_simulator
python --version
```

创建虚拟环境并安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

后续命令都应在本目录执行，否则相对路径形式的配置和数据集可能无法解析。

## 2. 先运行离线 Demo

离线模式使用确定性的 Mock 组件，不调用 OpenRouter，也不需要 API Key：

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --mock \
  --audit-level full
```

也可以使用安装后生成的命令：

```bash
reason-dag-demo --random-sample --difficulty medium --seed 42 --mock
```

启动后会先打印一条 `User:` 消息。Assistant 输入支持真正的多行内容：每次回车
只增加一行，单独输入一行 `/send` 才会提交整段回复。程序随后生成下一轮用户消息，
直到任务完成、达到轮数上限，或手动退出。

## 3. 配置 OpenRouter

在线模式会产生真实模型调用和费用。复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`，至少填写 API Key：

```dotenv
OPENROUTER_API_KEY=sk-or-v1-你的密钥
OPENROUTER_HTTP_REFERER=https://example.org
OPENROUTER_APP_TITLE=Reason-DAG User Simulator
```

只有 `OPENROUTER_API_KEY` 是必填项，后两项可留空。`.env` 已被 Git 忽略，
不要把真实密钥写入 YAML、README、日志或提交记录。

## 4. 启动真实模型 Demo

### 使用默认的 DeepSeek V4 Flash 0731

`configs/simulator.yaml` 默认让四个 LLM 组件都使用固定版本
`deepseek_v4_flash_0731`，实际 OpenRouter 模型 ID 为
`deepseek/deepseek-v4-flash-0731`：

该 profile 优先使用 Baidu，并按 `Baidu → SiliconFlow → NextBit → DeepInfra` 回退。
`only` 将 endpoint 限制在这四个已审核 provider 内，`quantizations: [fp8]` 保持量化一致，
`require_parameters: true` 会继续过滤不支持当前 reasoning 与 strict JSON Schema 参数的
endpoint，`allow_fallbacks: true` 允许 OpenRouter 在前序 provider 故障或限流后尝试下一项。

该 profile 还为 OpenRouter/provider 池耗尽后的 429 配置了独立重试预算。RPM 限流从 15 秒开始
指数退避，TPM 限流从 60 秒开始，无法识别类型的 429 从 30 秒开始，最大基础等待均为
60 秒；如果响应包含 `Retry-After`，则以它作为最低等待时间。每次等待额外加入最多 25%
随机抖动，避免 mini-batch 内的并发请求同时再次撞限。429 不消耗普通传输错误或结构化
输出的四次尝试预算；详细参数均可在对应模型 YAML 的 `retry.rate_limit` 下调整。

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --audit-level full
```

如果需要保持同一模型版本和生成参数，但将所有请求严格锁定到 DeepSeek 官方 provider，
使用独立 profile `deepseek_v4_flash_0731_official`：

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --model-profile deepseek_v4_flash_0731_official \
  --audit-level full
```

该 profile 的 `order` 和 `only` 均为 `[deepseek]`，并关闭 provider fallback。它有意不设置
`quantizations`：OpenRouter 当前将官方 endpoint 的量化标记为 `unknown`，添加 FP8 过滤会
导致该 endpoint 被排除。官方 endpoint 只声明支持 JSON Object，不声明原生 JSON Schema
enforcement，因此该 profile 使用 `transport: json_object`。客户端会把完整 Schema 加入系统
指令，响应仍必须通过同一个 strict Pydantic 校验，失败时进入原有结构化重试。原有
Baidu-first fallback profile 继续使用 provider 原生 `json_schema`，不受影响。

### 通过 OpenRouter 使用 GPT-5.6 Luna

用 `--model-profile` 一次覆盖 controller、satisfaction、clear realizer 和
abstract realizer：

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --model-profile gpt_5_6_luna \
  --audit-level full
```

对应配置文件为 `configs/models/gpt_5_6_luna.yaml`，实际发送给 OpenRouter
的模型 ID 是 `openai/gpt-5.6-luna`。

如果希望长期将 Luna 设为默认模型，可在自定义 YAML 中覆盖四项模型配置：

```yaml
models:
  controller: gpt_5_6_luna
  satisfaction: gpt_5_6_luna
  realizer_clear: gpt_5_6_luna
  realizer_abstract: gpt_5_6_luna
```

然后启动：

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --config path/to/your_config.yaml
```

## 5. 选择样本和难度

每次必须且只能选择一种样本方式：

- `--random-sample`：按 `--seed` 可复现地随机选择样本。
- `--sample-id <ID>`：运行指定样本。

例如：

```bash
python -m user_simulator.cli.demo \
  --sample-id <数据集中的sample_id> \
  --difficulty hard \
  --seed 7 \
  --mock
```

可选难度如下：

| 难度 | 用户行为 |
| --- | --- |
| `easy` | 一次清晰表达全部已暴露但未解决的需求 |
| `medium` | 按随机种子选择有序前缀，并在 Clear/Abstract 表达间切换 |
| `hard` | 每次只表达最早的未解决需求，并使用更含蓄的表达 |

Medium 难度的两个随机维度分别配置：

- `medium_clear_probability`：选择 Clear 表达模式的概率，范围为 `0.0–1.0`。
- `medium_min_nodes`：有序前缀最少选择的节点数，必须大于等于 1。
- `medium_max_nodes`：有序前缀最多选择的节点数；`null` 表示不额外限制上限。

节点数量会在包含上下界的范围内均匀随机，并自动限制到当前实际可用的未解决节点数。
如果希望固定选择前 `n` 个节点，把最小值和最大值都设为 `n`：

```yaml
policy:
  medium_clear_probability: 0.5
  medium_min_nodes: 2
  medium_max_nodes: 2
```

此时只要队列中至少有两个节点，Medium 策略都会固定选择前两个；如果只剩一个，
则选择唯一可用的节点。Easy 仍选择全部节点，Hard 仍只选择第一个节点。

`monotonic_satisfaction: true` 表示满意度只能保持或上升：
`unsatisfied → partially_satisfied → satisfied`。如果后续模型把已经 satisfied 的节点
重新判断为 partial/unsatisfied，系统会保留 satisfied，并在审计中记录一次归一化违规。
设为 `false` 后则完全采用最新判断，允许满意度下降；这可能使已经移出未解决队列的节点
再次进入队列。

每个满意度判断还包含审计用的 `reason` 和可空的 `remaining_gap`。只有
`partially_satisfied` 必须提供非空 gap；其他状态必须返回 `null`。完成节点选择后，
系统只把被选中节点的 gap 传给 user realizer，不传递 reason 或未选中节点的 gap。

满意度采用 Stop-Now Test：假设对话在最新 Assistant 回复后立刻结束，只有 Assistant
已经为该节点提供了可用答案、建议、指导或实际要求的交互行为，才可以获得 partial 或
satisfied。单纯追问、收集前置信息或承诺稍后帮助仍是 unsatisfied；但对于“放慢节奏、
减少问题、改变表达方式”等交互风格节点，Assistant 已经发生的行为变化本身可以构成帮助。

`remaining_gap` 只能描述当前节点明确要求但尚未交付的核心内容。用户已经陈述的事实可以
细化当前节点已有的要求，但不能扩展出新的交付目标；Assistant 自己提出的前置信息、可选的
个性化输入、后续未暴露节点的信息，即使后来被用户复述，也不能成为当前节点的硬性 gap。
如果当前节点要求的可用答案已经给出，只是缺少让答案更个性化或更优的信息，应判为
`satisfied`，而不是让节点长期停留在 `partially_satisfied`。

## END 暴露与终止

Controller 只接收按顺序排列的 intent candidate，并且只返回这些 intent 的暴露判断；
它不接收 END 边信息，也不输出 `end_exposed`。系统应用 prefix closure 并更新 frontier 后，
会在同一回合根据 DAG 结构推导 END：当前前缀完整且原 frontier 存在 END 边时暴露 END；
或者该完整步骤推进到一个只剩 END 边的最终 frontier 时立即暴露 END。后者用于避免一次
跨多个 intent 的暴露已经到达最终节点，但 END 仍滞后一回合。

END 一旦暴露便不可逆，后续回合永久跳过 Controller，只继续进行 satisfaction 更新和必要的
user realization。只有 `end_exposed=true` 且全部已暴露 intent 都为 `satisfied` 时，系统才
真正终止 episode。因此 END 的结构暴露和任务满意度判定仍然是两套独立机制。

Controller 只允许直接作用于下一条用户回复的 exposure：直接提供相关帮助、直接追问该
语义槽，或者 Assistant 的假设、节奏、方案和方向直接触发对应的纠正或重定向。不得通过
“先暴露一个节点、该节点未来又可能引出另一个节点”的假设性中间用户轮次暴露后续节点。

Clear 和 Abstract realizer 使用相同的事实权限边界：`TASK SUMMARY` 和
`TASK EXPECTATION` 只控制全局方向与一致性；当前消息中的具体内容只能来自 selected
nodes 或用户已经在可见历史中说过的事实。Assistant 的假设和示例不会自动变成用户事实，
未指定的领域、活动、工具、流程、偏好和约束必须继续保持未指定。Abstract 可以减少具体性，
但不能增加具体性。未选中 unresolved nodes 只以 ID 传给 realizer，不传递语义详情。

常用启动参数：

- `--max-turns 20`：设置最大 Assistant 轮数，必须大于等于 1。
- `--audit-level full|summary`：选择完整或脱敏摘要审计。
- `--show-latent-summary`：显式显示仅供审计的隐藏任务摘要。
- `--config <path>`：加载并合并自定义模拟器配置。
- `--model-profile <name|path>`：让所有 LLM 组件使用指定模型配置。

查看完整帮助：

```bash
python -m user_simulator.cli.demo --help
```

## 6. 交互命令

在 `Assistant:` 提示符后逐行输入回复，例如：

```text
... 第一段第一行
... 第一段第二行
...
... 第二段
... /send
```

提交给模拟器的内容会保留为：

```text
第一段第一行
第一段第二行

第二段
```

`/send` 必须独占一行且不会进入回复正文。以下管理命令如果作为第一行输入，
会立即执行，不需要再输入 `/send`：

| 命令 | 作用 |
| --- | --- |
| `/audit` | 查看最近一次结构化审计 |
| `/raw-audit` | 查看最近一次原始审计数据 |
| `/prompts` | 查看 Prompt 路径和内容哈希 |
| `/schemas` | 查看结构化输出 Schema 哈希 |
| `/state` | 查看当前 Episode 状态 |
| `/dag` | 查看隐藏 DAG，仅用于审计 |
| `/nodes` | 查看已暴露节点，仅用于审计 |
| `/history` | 查看当前可见对话历史 |
| `/save` | 立即刷新日志 |
| `/quit` | 保存状态并退出 |
| `/send` | 提交当前多行 Assistant 回复 |

## 7. 输出和审计文件

启用审计后，每次运行会在 `runs/<run-id>/` 下写入：

- `config_snapshot.yaml`：本次运行的最终配置和模型配置快照。
- `events.jsonl`：逐事件审计记录。
- `transcript.jsonl`：可见对话记录。
- `final_state.json`：保存或结束时的最终状态。

每次 simulator 模型调用的 `llm_call` 会记录 `input_tokens` 和 `output_tokens`。
当 OpenRouter 返回 `completion_tokens_details.reasoning_tokens` 时，还会分别记录
`thinking_tokens`，并计算 `answer_tokens = output_tokens - thinking_tokens`。
OpenRouter 的 `output_tokens` 已包含 thinking；如果 provider 不返回拆分信息，
`thinking_tokens` 和 `answer_tokens` 会保存为 `null`，不会根据配置猜测。

`full` 适合本地调试；`summary` 会隐藏 Prompt、原始消息和潜在节点文本等敏感细节。

## 8. 启动前验证

校验数据集：

```bash
python -m user_simulator.cli.validate_dataset \
  --dataset dataset/DAG.jsonl \
  --strict
```

运行不访问模型的 Prompt 合约检查：

```bash
python -m user_simulator.cli.prompt_smoke_test \
  --mode contract \
  --component all \
  --cases tests/fixtures/prompt_cases.jsonl \
  --output-dir runs/prompt_smoke
```

针对事实边界、reaction exposure、multi-step exposure 和 clarification satisfaction 的
专项回归输入位于 `tests/fixtures/review_regressions.jsonl`，可通过同一命令的 `--cases`
参数运行。

运行完整测试：

```bash
pytest -q
```

## 9. 模型参数调整说明

模型 YAML 已用英文注释标出 `ADJUSTABLE`、`FIXED` 和推荐范围：

- `configs/models/deepseek_v4_flash_0731.yaml`
- `configs/models/deepseek_v4_flash_0731_official.yaml`
- `configs/models/deepseek_v4_pro.yaml`
- `configs/models/gpt_5_6_luna.yaml`
- `configs/models/qwen_3_8_27b_vllm.yaml`

每个 `generation` 项都可以独立设置 `reasoning`，因此 controller、satisfaction、
clear realizer 和 abstract realizer 可以使用不同的 reasoning 强度。使用
`generation.<component>.reasoning.enabled: false` 可以完全关闭该组件的 reasoning；
启用时通过同一块中的 `effort` 调整强度。顶层 `reasoning` 仅作为兼容旧 profile 的默认
回退值，组件级配置优先。各 profile 的实际组件设置以对应 YAML 为准；新增的本地 Qwen3.8
profile 将 controller 和 satisfaction 设置为 `low`，两个 realizer 关闭 reasoning。也可以
独立调整各组件的输出 token 上限、回退开关和重试参数。`retry.rate_limit` 是可选配置；
启用后会独立处理 RPM/TPM 429，并在审计 metadata 中记录
`rate_limit_retry_count`、`rate_limit_wait_seconds` 和 `last_rate_limit_type`。
严格结构化输出相关的 `type`、`strict`、`require_parameters` 不应修改。`transport` 默认是
provider 原生 `json_schema`；只有不提供原生 Schema enforcement 的 DeepSeek 官方 profile
使用 `json_object` 传输，并保留完整 Schema 指令、严格本地验证及结构化重试。
DeepSeek 支持调整 `temperature`；Luna 当前在 OpenRouter 上不公开该参数，
因此其配置必须保持 `temperature: null`，客户端会完全省略该请求字段。

### 本地 vLLM：Qwen3.8-27B

内置 profile `qwen_3_8_27b_vllm` 连接 `http://127.0.0.1:8005/v1`，模型名为
`Qwen/Qwen3.8-27B`。可按下面的方式启动服务；显存、并行度和上下文长度参数应根据本机
硬件调整：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8005 \
  --served-model-name Qwen/Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --language-model-only
```

Simulator 可直接选择该 profile，不需要 `OPENROUTER_API_KEY`：

```bash
python -m user_simulator.cli.demo \
  --sample-id <sample_id> \
  --difficulty hard \
  --model-profile qwen_3_8_27b_vllm
```

该 profile 使用 Qwen3.8 官方 generation defaults：`temperature=1.0`、
`top_p=0.95`、`top_k=20`。Controller 和 satisfaction 通过官方 chat-template 参数启用
thinking 并设置 `reasoning_effort=low`；两个 realizer 通过 `enable_thinking=false` 关闭
thinking。所有组件仍使用现有严格 JSON Schema、语义校验、重试、token 统计和审计机制。

参数能力参考：

- [OpenRouter：DeepSeek V4 Flash 0731](https://openrouter.ai/deepseek/deepseek-v4-flash-0731)
- [OpenRouter：DeepSeek V4 Pro](https://openrouter.ai/deepseek/deepseek-v4-pro)
- [OpenRouter：GPT-5.6 Luna](https://openrouter.ai/openai/gpt-5.6-luna)
- [OpenRouter：Provider Routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [OpenRouter：Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [DeepSeek：Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [OpenAI：GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)
- [Qwen/Qwen3.8-27B 模型页](https://huggingface.co/Qwen/Qwen3.8-27B)

## 10. 常见问题

`OPENROUTER_API_KEY is required`：确认 `.env` 位于项目根目录，变量名拼写正确，
并重新启动命令。

`ModuleNotFoundError: user_simulator`：确认已激活 `.venv`，并在项目根目录执行
`pip install -e '.[test]'`。

HTTP 402：OpenRouter 账户或密钥额度不足。HTTP 429：触发限流，请稍后重试；
配置中的指数退避会处理可重试错误。

结构化输出或参数不兼容：不要关闭 `require_parameters`，先确认模型配置中的参数
仍被当前 OpenRouter 模型端点支持。模型能力会更新，线上部署前应重新检查模型页。

输出被截断：适度提高对应组件的 `max_completion_tokens`，同时注意模型上限、
上下文长度、延迟和费用。
