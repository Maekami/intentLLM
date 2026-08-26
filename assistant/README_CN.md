# Assistant Baselines

该目录按照 `user_simulator` 的 `src/ + configs/ + tests/ + CLI` 形式构建，提供可扩展的
OpenRouter 或本地 vLLM Assistant baseline。当前支持：

- 模型：`openai/gpt-5.6-luna`（默认）、`deepseek/deepseek-v4-flash-0731`
  和 `qwen/qwen3.6-27b`。
- Qwen 提供两个严格分离的 profile：`qwen_3_6_27b`（thinking）和
  `qwen_3_6_27b_non_thinking`（non-thinking）。
- 同一 Qwen 模型也提供本地 vLLM profile：`qwen_3_6_27b_vllm`（thinking）和
  `qwen_3_6_27b_vllm_non_thinking`（non-thinking）。
- `base`：不添加 system prompt 或其他隐藏指令，只发送可见的多轮对话。
- `prompt_base`：在同一份可见对话前添加 `configs/prompts/prompt_base.yaml` 中的 system
  prompt。
- Evo-Memory `ExpRAG`：从历史成功 episode 中检索相似经验，构造成 in-context experience，
  并在当前 episode 结束后更新持久记忆。
- Evo-Memory `ReMem`：在每次可见回答前运行模型驱动的 `Think` / `Think-Prune` /
  `Final Answer` 循环，并在 episode 结束后更新同一类经验记忆。

`base` / `prompt_base` 与 `ExpRAG` / `ReMem` 是两个可组合的维度：前者由
`configs/assistant.yaml` 的 `components.baseline` 选择；后者由所选模型 profile 中的
`memory` 字段控制。省略 `memory` 或设置为 `null` 时，行为与增加 Evo-Memory 前完全一致。

## 安装与配置

要求 Python 3.11 或更高版本。进入本目录后安装：

```bash
cd assistant
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
```

在 `.env` 中填写：

```dotenv
OPENROUTER_API_KEY=sk-or-v1-你的密钥
OPENROUTER_HTTP_REFERER=
OPENROUTER_APP_TITLE=Intent Assistant Baselines
VLLM_API_KEY=EMPTY
```

## 使用 CLI

运行默认组合（`base` + GPT-5.6 Luna）：

```bash
python -m assistant.cli.demo
```

运行 prompt baseline：

```bash
python -m assistant.cli.demo --baseline prompt_base
```

切换到 Qwen：

```bash
python -m assistant.cli.demo \
  --baseline prompt_base \
  --model-profile qwen_3_6_27b
```

测试 Qwen non-thinking：

```bash
python -m assistant.cli.demo \
  --baseline prompt_base \
  --model-profile qwen_3_6_27b_non_thinking
```

使用本地 vLLM thinking 模型：

```bash
python -m assistant.cli.demo \
  --baseline prompt_base \
  --model-profile qwen_3_6_27b_vllm
```

使用同一服务的 non-thinking 模式：

```bash
python -m assistant.cli.demo \
  --baseline prompt_base \
  --model-profile qwen_3_6_27b_vllm_non_thinking
```

本地 profile 不要求 `OPENROUTER_API_KEY`。如果 vLLM 启动时配置了 `--api-key`，把对应
值写入 `VLLM_API_KEY`；未配置认证时保留默认的 `EMPTY` 即可。

切换到固定版本的 DeepSeek V4 Flash 0731：

```bash
python -m assistant.cli.demo \
  --baseline base \
  --model-profile deepseek_v4_flash_0731
```

单轮调用后退出：

```bash
python -m assistant.cli.demo --message "Help me plan a trip."
```

交互模式支持真正的多行消息：单独输入 `/send` 提交；`/history` 查看历史，`/reset`
清空历史，`/quit` 退出。

## 程序化调用

```python
import asyncio

from assistant import EnvironmentSettings, build_assistant_components, load_config


async def main() -> None:
    config = load_config(
        cli_overrides={"components": {"baseline": "prompt_base"}}
    )
    components = build_assistant_components(
        config=config,
        environment=EnvironmentSettings(),
    )
    reply = await components.session.respond("I need help with a report.")
    print(reply)


asyncio.run(main())
```

`AssistantSession` 会保留成功调用的 user/assistant 历史。API 调用失败时，本轮 user 消息
不会写入历史，因此可以安全重试。

## 配置说明

- `configs/assistant.yaml` 选择组件、baseline、prompt 路径和默认模型。
- `configs/models/gpt_5_6_luna.yaml`、`configs/models/gpt_5_6_luna_non_thinking.yaml`、
  `configs/models/deepseek_v4_flash_0731.yaml`、
  `configs/models/qwen_3_6_27b.yaml` 和
  `configs/models/qwen_3_6_27b_non_thinking.yaml` 是 OpenRouter profile；对应的
  `qwen_3_6_27b_vllm.yaml` 和 `qwen_3_6_27b_vllm_non_thinking.yaml` 是本地 profile。
  `qwen_3_6_27b_exprag.yaml` 和 `qwen_3_6_27b_remem.yaml` 是开启两种 Evo-Memory
  framework 的 OpenRouter 示例；同一个 `memory` 块也可加入任意其他模型 profile。
  它们都遵循 simulator 模型 profile 的组织方式，注释列出了 `ADJUSTABLE`、`FIXED`、
  合法范围和推荐值。
- `configs/prompts/prompt_base.yaml` 保存 prompt baseline 的 system prompt。

也可以传入自定义配置或模型 YAML：

```bash
python -m assistant.cli.demo --config path/to/assistant.yaml
python -m assistant.cli.demo --model-profile path/to/model.yaml
```

环境变量 `ASSISTANT_BASELINE` 和 `ASSISTANT_MODEL_PROFILE` 可分别覆盖默认 baseline 和
模型；显式 CLI 参数优先级更高。

模型输出为自然语言，因此这里不会像 simulator 的内部判定组件一样请求严格 JSON
Schema。其余 OpenRouter 路由、reasoning、generation 和 retry 配置保持同类结构。

## Qwen3.6 与本地部署对齐

配置遵循 Qwen 官方 Hugging Face 模型卡，而不是用 OpenRouter 独有的 effort 档位模拟
本地行为：

| 模式 | profile | thinking 开关 | 官方采样参数 |
| --- | --- | --- | --- |
| Thinking | `qwen_3_6_27b` | `enable_thinking: true` | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` |
| Non-thinking | `qwen_3_6_27b_non_thinking` | `enable_thinking: false` | `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` |

两个 profile 都使用官方建议的 `max_completion_tokens: 32768`。Qwen3.6 不支持通过
`/think` 或 `/nothink` 软切换，所以不要把这两个字符串加入 prompt。

项目已经提供本地 profile。以下启动命令与 `qwen_3_6_27b_vllm*` 中的默认连接设置完全
匹配：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.6-27B \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen3.6-27b \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --language-model-only
```

因此通常只需通过 `--model-profile` 在 OpenRouter 和本地服务之间切换。若端口或
`--served-model-name` 不同，只修改本地 profile 的以下两项：

```yaml
model_id: qwen3.6-27b
base_url: http://127.0.0.1:8001/v1
```

当 `provider: vllm` 时，客户端会向本地服务发送：

```json
{"chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}
```

non-thinking profile 则发送两个 `false`。thinking profile 还会保留 API 返回的
`reasoning`/`reasoning_content`，并在后续请求中使用当前 vLLM 的 `reasoning` 字段，
使多轮对话可以使用官方的
`preserve_thinking` 行为，但 `AssistantSession.respond()` 只返回最终可见回答。

你的 `--max-model-len 65536` 可以运行这两个 profile，但低于 Qwen 模型卡为完整 thinking
能力建议的至少 128K 上下文。若要尽量比较 OpenRouter 与本地部署的模型能力，应在显存允许
时对齐上下文长度、模型权重/精度、vLLM 版本和采样配置；对于较短对话，profile 已对齐所有
请求侧可控的 thinking 与采样参数。

本地 vLLM 会正常保存 `input_tokens` 和总 `output_tokens`。只有服务端在
`completion_tokens_details.reasoning_tokens` 中返回可靠拆分时，才会进一步保存
`thinking_tokens` 和 `answer_tokens`；若当前 vLLM 版本未提供该字段，两项保持 `null`，
客户端不会通过对 reasoning 文本重新分词来制造可能不一致的统计。

本地服务应使用最新版 vLLM 或 SGLang，并启用 Qwen reasoning parser。Qwen 官方 vLLM
启动方式的关键参数如下：

```bash
vllm serve Qwen/Qwen3.6-27B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

显卡数量不足时可调整 tensor parallel 和上下文长度，但官方建议为了保留 thinking 能力，
上下文不要低于 128K。量化方式、上下文长度、推理框架版本和 MTP 是否启用仍可能造成性能
差异；当前配置已对齐可以由请求侧控制的 chat template 与采样参数。

官方依据：

- [Qwen/Qwen3.6-27B Hugging Face 模型卡](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.6 tokenizer chat template](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/tokenizer_config.json)

## 测试

```bash
pytest -q
```

测试使用假的 OpenAI-compatible SDK 响应，不会发起真实网络请求，也不产生费用。

## Evo-Memory：ExpRAG 与 ReMem

实现位于 `src/assistant/memory/`，依据论文提出的 `(F, U, R, C)` 和
Search-Synthesize-Evolve 流程，并参考
[官方 evo_mem 仓库](https://github.com/zhaosnw/evo_mem)（实现核对基准：
[`10486d8`](https://github.com/zhaosnw/evo_mem/commit/10486d80a2a59903d543dd1e0438505b08ec2275)）：

1. **Search**：每个新 episode 的第一条用户消息到来时，从共享 JSON memory 中检索 top-k
   历史任务经验；同一 episode 后续回合沿用该 working set。
2. **Synthesize**：ExpRAG 把检索结果按 `Task / Trajectory / Output / Feedback / Result`
   组织为上下文；ReMem 额外让模型在完整回复级别选择 `Think`、`Think-Prune` 或
   `Final Answer`。
3. **Evolve**：将一整个 simulator × assistant episode 视为论文中的一个 multi-turn
   task。pipeline 获得外部终止结果后，调用 `finalize_task()` 一次；默认只保存正常终止的
   episode，失败或超过 turn limit 的 episode 不进入记忆。

ReMem 的 `Think-Prune` 只删除当前 episode 已检索到的 working-memory 项，不删除持久 JSON
中的历史经验，这与官方参考实现一致。正常完成前的 `Think` 操作只进入内部 working trace，
不会单独写入可见 transcript；内部调用产生的 `input_tokens`、总 `output_tokens`、
`thinking_tokens`、`answer_tokens`、延迟和重试次数会合并后记在当前 assistant 回合，
`internal_call_count` 和 `remem_operation_counts` 也会进入审计元数据。

### 直接使用预置 profile

仓库提供两个 OpenRouter Qwen thinking 示例：

```bash
# ExpRAG
python -m assistant.cli.demo --model-profile qwen_3_6_27b_exprag

# ReMem
python -m assistant.cli.demo --model-profile qwen_3_6_27b_remem
```

交互式 assistant CLI 没有外部 correctness/success evaluator，因此它可以读取已有 memory，
但不会把一次普通 `/quit` 或 `/reset` 擅自当作成功经验写回。完整的自动更新应通过
interaction pipeline 运行，或在程序调用中显式结束任务：

```python
reply = await components.session.respond("Help me plan a trip.")
update = components.session.finalize_task(
    task_id="my-stream-task-001",
    success=True,
    feedback="Externally evaluated as successful.",
)
```

pipeline 示例（从仓库根目录并按前文设置好 `PYTHONPATH`）：

```bash
python -m interaction_pipeline.cli.batch \
  --all --limit 30 \
  --assistant-model-profile qwen_3_6_27b_exprag \
  --baseline base \
  --concurrency 1
```

Evo-Memory 评估是有顺序的任务流。JSON store 使用文件锁和原子替换，因而并发写入不会破坏
文件；但 `--concurrency > 1` 会让不同 episode 的检索/更新时间交错，无法保证严格的
`M_t -> M_{t+1}` 顺序。需要复现论文式 test-time learning 曲线时请使用
`--concurrency 1`，并为每个 setting 配置全新的 memory 路径。

### 在任意模型 profile 上开启

`qwen_3_6_27b_exprag.yaml` 和 `qwen_3_6_27b_remem.yaml` 展示了所有可调字段。也可以把下面
的 `memory` 块加入任意现有 OpenRouter 或本地 vLLM profile；底座模型、thinking 开关和
generation 参数仍由该 profile 原有字段决定：

```yaml
memory:
  # exprag | remem；删除整个字段或设置 null 即关闭记忆框架
  framework: exprag
  # 相对路径以当前模型 YAML 所在目录为基准解析
  path: ../../runs/memory/my_qwen_vllm_exprag.json
  max_entries: 1000
  prune_oldest_when_full: true
  store_successful_only: true
  retrieval:
    backend: bm25                 # bm25 | sentence_transformers
    top_k: 4
    min_score: 0.0
    embedding_model: BAAI/bge-base-en-v1.5
    device: null                  # null | cpu | cuda 等
    bm25_k1: 1.5
    bm25_b: 0.75
  context:
    include_trajectory: true
    include_feedback: true
    max_characters: 8000
  remem:
    max_iterations: 10
    enable_pruning: true
```

默认 `backend: bm25` 是纯 Python 实现，不新增必装依赖，因此仍可直接从源码运行。官方参考
代码默认使用 `BAAI/bge-base-en-v1.5` embedding；若要对齐该检索路径，将 backend 改为
`sentence_transformers`，并确保当前环境已有该可选包，或执行：

```bash
pip install -e '.[memory-semantic]'
```

这只改变记忆检索器，不改变 Qwen thinking、采样参数或 OpenRouter/vLLM 调用方式。记忆文件
采用可人工审计的格式化 JSON，默认写入 `assistant/runs/memory/`；同一个 `task_id` 再次运行
时更新原条目，容量满时按配置删除最旧条目或停止新增。

论文与实现依据：

- [Evo-Memory 论文（arXiv）](https://arxiv.org/abs/2511.20857)
- [Evo-Memory 官方参考代码](https://github.com/zhaosnw/evo_mem)
