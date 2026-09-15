# Assistant Baselines

该目录按照 `user_simulator` 的 `src/ + configs/ + tests/ + CLI` 形式构建，提供可扩展的
OpenRouter 或本地 vLLM Assistant baseline。当前支持：

- 模型：`openai/gpt-5.6-luna`、`google/gemini-3.6-flash`、
  `deepseek/deepseek-v4-flash-0731` 和 `qwen/qwen3.6-27b`。
- Qwen 提供两个严格分离的 profile：`qwen_3_6_27b`（thinking）和
  `qwen_3_6_27b_non_thinking`（non-thinking）。
- 同一 Qwen 模型也提供本地 vLLM profile：`qwen_3_6_27b_vllm`（thinking）和
  `qwen_3_6_27b_vllm_non_thinking`（non-thinking）。
- `base`：不添加 system prompt 或其他隐藏指令，只发送可见的多轮对话。
- `prompt_base`：在同一份可见对话前添加 `configs/prompts/prompt_base.yaml` 中的 system
  prompt。
- `interactcomp_react`：对齐 InteractComp 的 ReAct `ask_only` 动作集，只保留 `ask` 和
  `answer`；动作由 JSON 选择，默认再经过可关闭的语义 action guard，对 simulator 只暴露
  通过校验的自然语言 payload。
- `trace2skill`：测试时把离线生成且冻结的 `SKILL.md` 作为唯一 system message；强制
  `memory: null`，不在测试时检索或更新。
- Evo-Memory `ExpRAG`：从历史成功 episode 中检索相似经验，构造成 in-context experience，
  并在当前 episode 结束后更新持久记忆。
- Evo-Memory `ReMem`：在每次可见回答前运行模型驱动的 `Think` / `Think-Prune` /
  `Final Answer` 循环，并在 episode 结束后更新同一类经验记忆。

`base` / `prompt_base` 表示是否叠加 Prompted-Base system prompt：显式 `base` 即“不添加
Prompted Base”。ExpRAG/ReMem 由所选模型 profile 的 `memory` 字段控制；Trace2Skill 也可以
由 profile 的 `skill` 字段绑定。因而 `--baseline base --model-profile
qwen_3_6_27b_trace2skill` 仍会加载该 profile 自己的静态 skill，但不会再叠加 Prompted Base。
profile-bound Trace2Skill 与 memory framework 互斥。

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
  `qwen_3_6_27b_exprag.yaml` 和 `qwen_3_6_27b_remem.yaml` 是调用本地 vLLM 的两种
  Evo-Memory 示例；同一个 `memory` 块也可加入任意其他模型 profile。
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

## InteractComp ReAct baseline

`interactcomp_react` 以 [InteractComp 论文](https://arxiv.org/abs/2510.24668)和
[官方实现](https://github.com/FoundationAgents/InteractComp/tree/9cdf7f804f527ad32a405efaa6c86aae03692556)为优先参照：沿用每轮只选择一个
动作、`{"action": ..., "params": ...}` schema、自然语言单一范围澄清问题，以及“首次生成 +
最多 3 次无效动作重试”。适配到本项目时有以下有意差异：

- 删除 `search`，动作空间固定为 `ask` / `answer`；
- 不设置 probing budget，也不强制最少 ask 次数；
- 将官方 `ask_NL` 重命名为 `ask`，仍接收短自然语言回复，但回复来自本项目的实时 simulator，
  而不是官方基于静态隐藏 context 的 `Responder`；
- `answer` 不是终止动作，后续用户消息到来时仍继续选择 `ask` 或 `answer`；
- 不使用官方 final-round forced-answer prompt，也不在 baseline 内保存 episode 的 20-turn
  可见回合预算；下面的 10-slot 预算只管理单个可见 turn 内的无效 action 恢复。

因此终止语义完全沿用本项目：simulator 系统推导 END，在所有已暴露 intent node 满足后自然
终止；否则由外层 `max_turns` 截止。

默认启用的第二层约束沿用官方 `AskNLValidator` 的独立调用角色和 `{ok, reason}` 输出接口，但为本
项目同时检查 `ask` 和非终止的 `answer`。guard 只判断通信方向的动作边界：`ask` 是否符合官方
“一个属性/关系/时间/地点或一个 yes/no 命题，且不索取完整列表/概述、不组合多个子问题或维度”
规则并且不夹带解答，`answer` 是否提供实质内容且不向当前用户索取回复；它不判断事实质量、任务
相关性、新颖性、重复性、必要性、充分性、完整性、节点满足率、是否应该继续追问或追问次数。为
贴近官方权限边界，校验 `ask` 时只提交当前候选 `question`，不提交对话历史；校验 `answer` 时才
提交可见对话和候选动作，而且对话只能用于判断问题或请求是否要求当前用户回复。prompt 中不嵌入
针对某个数据样本的规则或措施。

官方 runner 从 `test_llm` 构造被测 assistant，并从 `user_llm` 另外构造 `responder_llm`；同一个
`responder_llm` 实例同时供 `AskNLValidator` 和 `Responder` 使用。因此两者是不同的实验角色、配置
入口和客户端实例，但配置恰好相同时并不保证 model ID 不同。本项目按这个边界把 guard client 与
assistant client 分离：其模型身份、provider、routing、reasoning、transport retry 和结构化输出
能力来自命令中 `--simulator-model-profile` 指定的**同一份 YAML**。没有显式传参时，仅当 simulator
的 controller、satisfaction、clear realizer、abstract realizer 四个路径完全相同才允许推断；否则
在创建 batch 和付费调用前报配置错误。hard 模式的可见用户回复仍走
`generation.realizer_abstract`；这不改变 guard 的模型来源，guard 不从 `realizer_clear` 或
`realizer_abstract` 二选一。

guard 的共享采样参数位于 `configs/assistant.yaml`，默认 `temperature: 1.0`、`top_p: 1.0`，与官方
runner 对 user/responder LLM 的温度覆盖对齐；三个 assistant React profile 不再各自保存 guard
generation。项目 adapter 使用 simulator client 的严格结构化输出通道来实现 `{ok, reason}`，这是
对现有管线 transport/retry 的必要适配，不改变 validator 的语义权限。

官方 10 个 action round 在这里映射为**每个 simulator 可见 turn 内的 10 个语义 action slot**：

1. 每个 slot 先生成一个 schema 合法候选；格式/schema/action-name 错误在同一 slot 内按官方
   `max_invalid_retries=3` 执行“首次 + 最多 3 次重试”，不消耗新的语义 slot。
2. 每个 schema 合法候选只调用一次 guard，与官方 `AskNLValidator` 的单次判定一致；判为不合法
   就消耗一个语义 slot，并把规范化候选与 `ask_invalid` / `answer_invalid` observation 写入 agent
   的私有 ReAct 轨迹；判定理由只进入审计，不反馈给 agent。默认不进行拒绝复核，也不缓存相同
   payload 的 verdict。
3. 第 10 个 slot 的合法候选仍会正常提交。只有第 10 个候选也被拒绝时才抛出有类型的预算耗尽
   错误；不会伪造自然语言 assistant/user 消息，也不会用固定话术让外部 turn 加一。随后是否重跑
   整个样本仍由统一的 episode-level `sample_retries` 控制。

与官方无法解析 validator 输出即返回 `False` 一致，单次 guard 输出不符合 `{ok, reason}` 时按
拒绝处理并消耗当前语义 slot，而不是直接报告 episode 错误；provider/transport 调用本身最终失败
仍按管线统一错误语义处理。设置 `interactcomp_react.action_guard.enabled: false` 或传入
`--no-react-action-guard` 会完全跳过这些额外调用，退化为 prompt + schema/parser 第一层约束。

模型的正式历史保存规范化的已接受动作 JSON；被 guard 拒绝的 action/observation 只保留在 agent
私有历史，transcript 和 user simulator 始终只看到通过校验的 `question` / `answer` 文本。成功和
失败 turn 的审计都会记录语义 slot、格式重试、guard verdict 与无效输出、模型调用数、成本和预算
是否耗尽。guard prompt 位于 `configs/prompts/interactcomp_action_guard.yaml`。

三个正式 profile 均不挂载 memory 或 static skill，并与对应模型的现有实验参数对齐：

- `qwen_3_6_27b_react`
- `gpt_5_6_luna_react`
- `gemini_3_6_flash_react`

guard client 需要由 interaction pipeline 注入，因此正式运行和 `--limit` 烟测均使用与其他
baseline 相同的 batch 入口，例如：

```bash
python -m interaction_pipeline.cli.batch \
  --all --limit 8 \
  --baseline interactcomp_react \
  --assistant-model-profile qwen_3_6_27b_react \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --react-action-guard \
  --max-turns 20
```

若要复现实装第二层之前的第一层版本，把 `--react-action-guard` 替换为
`--no-react-action-guard`；此时 standalone assistant demo 也不再需要 simulator profile。

## Trace2Skill 静态 baseline

`trace2skill` baseline 读取离线 evolution 生成的 Markdown `SKILL.md`，并把它作为唯一一条
system message 放在可见对话之前。它不是第三种 memory framework：选择该 baseline 时模型
profile 必须保持 `memory: null`，运行期间没有检索、开发集访问或写回。

正式实验优先使用三个一一绑定的 profile：

- `qwen_3_6_27b_trace2skill`
- `gpt_5_6_luna_trace2skill`
- `gemini_3_6_flash_trace2skill`

每份 profile 的模型、thinking、generation 和 retry 参数与同模型的 ExpRAG/ReMem profile
完全一致，并分别绑定 `runs/skills/<profile_name>/SKILL.md`。选择 profile 后不需要再传 skill
路径；`--baseline base` 可显式保留，用来强调没有 Prompted Base：

```bash
python -m assistant.cli.demo \
  --baseline base \
  --model-profile qwen_3_6_27b_trace2skill
```

下面的显式路径形式仅作为临时、自定义 skill 的兼容入口：

```bash
python -m assistant.cli.demo \
  --baseline trace2skill \
  --trace2skill-skill /absolute/path/to/SKILL.md \
  --model-profile qwen_3_6_27b
```

离线制作分成 `trace2skill collect` 和 `trace2skill build`：正式 Collection 一次性封存该模型的
1000 条 No-Skill trajectory；Collection 也支持 `--limit N` 采集前缀以审计交互与封存流程，但
这种 corpus 明确禁止进入 Build。Skill 制作的 `--limit N` 则只从完整 corpus 读取前 N 条，便于
反复审计而不重新运行 Simulator；Build 已对齐官方的 analysis-record、parallel JSON MAP/REDUCE、
逐 edit translation、确定性 APPLY 和 validation-fix 流程。只有使用完整 1000 条的 Build 才会
发布到上述 profile 路径。
严格信息边界、resume 和测试集评估命令见 `interaction_pipeline/README_CN.md` 的 Trace2Skill
章节。

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

1. **Search**：每次可见的 user→assistant 交互都重新检索 top-k 历史任务经验。检索 query
   只使用截至当前轮已显式出现的用户消息，把最新请求放在最前并按
   `query_max_characters` 截断；不读取 simulator 的隐藏目标。整个 episode 的可检索 memory
   bank 保持冻结，evolution 模式下就是 mini-batch 开始时的冻结 snapshot。
2. **Synthesize**：ExpRAG 每轮用新的 top-k，按
   `Task / Trajectory / Output / Feedback / Result` 组织为上下文；ReMem 每轮先建立新的
   working set，再让模型在该轮完整回复级别选择 `Think`、`Think-Prune` 或 `Final Answer`。
3. **Evolve**：将一整个 simulator × assistant episode 视为论文中的一个 multi-turn
   task。pipeline 获得外部终止结果后，调用 `finalize_task()` 一次；默认只保存正常终止的
   episode，失败或超过 turn limit 的 episode 不进入记忆。

ReMem 的 `Think-Prune` 只删除当前 user→assistant 交互的 working-memory 项，不删除冻结 bank
或持久 JSON；下一轮会从同一冻结 bank 重新检索。正常完成前的 `Think` 操作只进入本轮内部
working trace，不会单独写入可见 transcript；内部调用产生的 `input_tokens`、总
`output_tokens`、`thinking_tokens`、`answer_tokens`、延迟和重试次数会合并后记在当前
assistant 回合，`internal_call_count` 和 `remem_operation_counts` 也会进入审计元数据。
`memory_retrieved_*` 统一表示本轮初始 top-k；ReMem 另记 `memory_retained_*` 和
`memory_pruned_*`，避免把“检索后全部 prune”误判为“没有检索”。

官方 Evo-Memory 的 multi-turn 环境会在开始时向 agent 提供完整 goal，因此只检索一次；本仓库
面对的是用户逐轮暴露意图的对话，采用上述 per-turn conversational retrieval 适配。它只使用
assistant 当时可见的信息，同时保留官方的 top-k、上下文结构和 ReMem 内部循环。

### 直接使用预置 profile

仓库为三个底座模型分别提供 ExpRAG 和 ReMem profile：

| 底座模型 | ExpRAG profile | ReMem profile |
| --- | --- | --- |
| Qwen3.6-27B | `qwen_3_6_27b_exprag` | `qwen_3_6_27b_remem` |
| GPT-5.6 Luna | `gpt_5_6_luna_exprag` | `gpt_5_6_luna_remem` |
| Gemini 3.6 Flash | `gemini_3_6_flash_exprag` | `gemini_3_6_flash_remem` |

六个 profile 的 `reasoning` 和 `generation.assistant` 均逐字段对齐各自的 non-thinking
profile：Qwen 关闭 thinking 并采用 instruct/non-thinking 采样参数，GPT-5.6 Luna 使用
`effort: none`，Gemini 3.6 Flash 因模型要求必须 thinking，使用其 non-thinking 近似
`effort: minimal`。它们分别使用独立的 `assistant/runs/memory/<profile_name>.json`，因此可以
作为六个并行 evolution 条件运行。其中 Qwen 两组调用 `http://127.0.0.1:8001/v1` 的本地
vLLM，Luna/Gemini 两组通过 OpenRouter 调用。单独启动 Qwen 示例：

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

开发集批量 evolution 应使用 interaction pipeline 的独立 evo 入口。其默认
`mini_batch_size: 8`、`concurrency: 8`、`sample_retries: 0`，批内读取冻结 snapshot，批末
统一提交：

```bash
python -m interaction_pipeline.cli.evo \
  --all \
  --assistant-model-profile qwen_3_6_27b_exprag
```

Evo-Memory 评估是有顺序的任务流。JSON store 使用文件锁和原子替换，因而并发写入不会破坏
文件；但 `--concurrency > 1` 会让不同 episode 的检索/更新时间交错，无法保证严格的
`M_t -> M_{t+1}` 顺序。需要复现论文式 test-time learning 曲线时请使用
`--concurrency 1`，并为每个 setting 配置全新的 memory 路径。

### 在任意模型 profile 上开启

上述六个 memory profile 展示了所有可调字段。也可以把下面的 `memory` 块加入任意现有
OpenRouter 或本地 vLLM profile；底座模型、thinking 开关和 generation 参数仍由该 profile
原有字段决定：

```yaml
memory:
  # exprag | remem；删除整个字段或设置 null 即关闭记忆框架
  framework: exprag
  # 相对路径以当前模型 YAML 所在目录为基准解析
  path: ../../runs/memory/my_qwen_vllm_exprag.json
  max_entries: 2000
  prune_oldest_when_full: true
  store_successful_only: true
  retrieval:
    backend: bm25                 # bm25 | sentence_transformers
    top_k: 4
    min_score: 0.0
    query_max_characters: 2048    # 最新用户请求优先的 per-turn query 上限
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

通用配置 schema 和上面的自定义示例仍以 `backend: bm25` 作为无额外依赖的 fallback。六个
预置 ExpRAG/ReMem profile 已按官方参考检索路径设置为
`sentence_transformers + BAAI/bge-base-en-v1.5 + top_k=4 + min_score=0.0`，并显式使用 CPU，
避免六个独立 evo 进程与本地 Qwen vLLM 争抢 GPU。使用这些预置 profile 前请确保当前环境
已有可选依赖，或执行：

```bash
pip install -e '.[memory-semantic]'
```

首次使用时 `sentence-transformers` 会下载 BGE 权重，建议在并行启动六个 tmux 任务前先完成
一次模型加载。模型对象和文本 embedding 在单个 evo 进程内跨样本复用；相同 mini-batch
冻结 snapshot 的归一化 embedding 矩阵也会复用。不同 tmux 进程不共享内存，但共享本机的
Hugging Face 文件缓存。

这只改变记忆检索器，不改变 profile 已选定的 non-thinking 思考模式、生成参数或
OpenRouter/vLLM 调用方式。记忆文件采用可人工审计的格式化 JSON，默认写入
`assistant/runs/memory/`；同一个 `task_id` 再次运行时更新原条目，容量满时按配置删除最旧
条目或停止新增。

论文与实现依据：

- [Evo-Memory 论文（arXiv）](https://arxiv.org/abs/2511.20857)
- [Evo-Memory 官方参考代码](https://github.com/zhaosnw/evo_mem)

## Goal-Progression 静态系统（当前 R20 Format）

当前执行先读 [CURRENT_STATE](../CURRENT_STATE.md) 和 [V4 Static 任务书](plan_v4_static.md)。活动运行代码仍是经核验的 R20 Format；V4 的 Actor–Validator 尚待在独占工作区 `runs_batch_qwen_full/gp_v4/static/` 实现，后续 agent 的全部写入限于该目录。真实实验最多 30 批，允许两个 API key 同时运行两批；Evo 暂停。[V3 文档、代码副本与结果](../v3/README.md) 已整体归档，V2.6 与全量实验见 [v2_6](../v2_6/README.md)。下方通用接口示例不是要求在主目录启动 V4。

`goal_progression`通过现有factory与interaction pipeline使用。用户在v2.6阶段结束后明确指定 **R20 Format** 为主版本：R20逻辑与profile保持不变，仅接入四个已冻结的格式整理prompt，架构仍为`v2_contracts`，profile仍为`qwen_3_6_27b_gp`。默认数据改为`DAG_fixed.jsonl`，已完成全292样本四次实验，两个key各两次，共1168次尝试，sample_retries=0。四次均值E/S为4.5348/6.0649；每样本按S最小、再E最小选择完整轨迹后为3.5308/4.7808（287 SUCCESS、5超限）。接入后352项assistant测试通过（14.59s），独立统计复核通过；见[全量最终结果](../v2_6/runs_batch_qwen_full/r20_format_full292/RESULTS.md)。[v2.6结论](../v2_6/runs_batch_qwen_full/gp_v2_6/RESULTS.md)保留为历史证据。

Full主路径为Tracker → Intra / Inter并行 → Generator：正常有候选和正文时四次模型调用、三个串行阶段。Tracker维护可见目标、Need、约束与来源；Intra覆盖当前结果，Inter至多选择一个后续目标；Assembler检查引用/依赖、当前请求优先、去重和预算。Renderer拼接获批正文并追加冻结问题；可见历史、语义状态、请求台账和回执原子提交。失败草稿与prepared回执不代表用户收到回复。

当前Full实际配置：

| 字段 | 值 / 含义 |
| --- | --- |
| architecture_version / variant | v2_contracts / full |
| policy.adjacent_candidate_limit | 2，活跃相邻候选上限 |
| policy.max_adjacent_deliveries | 1，相邻正文单元上限 |
| policy.request_budget | 1，全局独立请求预算，当前优先 |
| structured_decoding.generator | prompt，普通JSON生成后强制本地schema、引用、单元覆盖校验 |
| realization.output_format | json_units，正文单元JSON；Renderer请求使用frozen_block |
| recovery.default_max_retries | 每回合每owner/code首次失败后的额外3次，合法范围0–3，可按角色与错误覆盖 |
| retry.max_attempts | 1，底层单调用总尝试次数；与组件修复及episode重跑分开 |
| recovery.call_timeout_seconds / turn_timeout_seconds | 180 / 900秒 |
| max_in_flight_requests | 32，GP请求并发；不同于batch的episode并发 |
| context.hard_context_tokens | 262144；角色软预算/扩容预算分别配置，不静默删必要原文 |

其他启用结构化角色默认发送服务端schema，同时做本地校验。Generator虽采用prompt解码，也不能放行非法JSON或未知引用。正常停滞与缺少用户材料由正常策略处理；内部错误按所有权恢复，不能靠删除受影响的required工作冒充成功。当前R20没有v2.6新增的Goal.progress、max_adjacent_goals或额外执行反馈窗口。

必须同时选择baseline和对应GP profile；GP与其他baseline、memory、skill不混用。本次Base仍使用既有 `prompt_base` + `qwen_3_6_27b_vllm_non_thinking`。历史no_tracker/no_intra/no_inter/joint/no_anticipate profile随R20原件恢复，其旧实验含义和结果见 [CURRENT_STATE](../CURRENT_STATE.md)；它们不是本轮v2.6单因素消融。

从仓库根目录运行assistant离线验收（不请求真实模型）：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=assistant/src:interaction_pipeline/src:user_simulator/src:metrics/src \
python -m pytest -q assistant/tests
```

这些合成和假模型测试验证协议、来源、恢复与事务，不证明真实模型一定正确完成工作，也不代表全仓库测试通过。

获准的新实验可沿用既有batch入口。下面仅说明R20的启动方式，不追加本阶段额度；必须使用新的输出目录：

```bash
PYTHONPATH=assistant/src:interaction_pipeline/src:user_simulator/src:metrics/src \
python -m interaction_pipeline.cli.batch \
  --all --limit 16 \
  --baseline goal_progression --assistant-model-profile qwen_3_6_27b_gp \
  --simulator-model-profile deepseek_v4_flash_0731 \
  --difficulty hard --max-turns 20 --seed 42 \
  --concurrency 16 --sample-retries 0 --no-update-memory \
  --output-dir runs_batch_qwen_full/new_r20_experiment
```

原生事件保留各角色requested/raw_result/validated/failed、快照、契约、修复与提交回执；R20原生schema_version=2。实际可见回复需与pipeline的assistant_generation_completed对齐，不能仅用写前事件判定交付。provider尝试、内部修复、角色调用及延迟分开统计；缺失usage标未知，内部completion tokens不能代替最终可见正文tokens。

v2.6候选采用不同schema/checkpoint协议，两个新消融profile已从活动目录移除，完整源码可在 [冻结版本](../v2_6/runs_batch_qwen_full/gp_v2_6/versions/v26_1/) 和 [候选目录](../v2_6/runs_batch_qwen_full/gp_v2_6/workspaces/v26_candidate/) 还原。不能将新profile单独放进R20源码执行。参见 [实现](../v2_6/runs_batch_qwen_full/gp_v2_6/IMPLEMENTATION.md)、[参数](../v2_6/runs_batch_qwen_full/gp_v2_6/PARAMETERS.md)、[离线复算](../v2_6/runs_batch_qwen_full/gp_v2_6/OFFLINE_REPRODUCTION.md)。

v2.6历史阶段使用18/30次常规batch及1次前缀豁免。本次全量阶段已独立完成4次、1168次样本尝试，不沿用旧额度；DAG_fixed已修复5个样本，实际轨迹观测前缀异常0，运行转移代码未改动。四次择优是事后挑选表现，不能当作一次运行期望或相对Base的稳定优势。四个batch真实退出0；原执行器因外部修改未使用的first_16.jsonl而在完成后的严格检查中退出1，实际fixed数据和运行输入一致，详见最终结果中的证据记录。
