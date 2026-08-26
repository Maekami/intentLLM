# Simulator × Assistant 自动交互管线

该目录直接组合现有 `user_simulator` 和 `assistant`，提供两个共用同一交互核心的模式：

- 批量模式：异步并发运行多个样本，并限制同时活跃的样本数；单个样本报错后可从头重试。
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

`--sample-retries N` 表示首次运行失败后，最多再完整运行该样本 `N` 次；默认值为 `3`，
因此单个样本最多执行 `4` 次。设为 `0` 可关闭重试。每次尝试都会重新构建 episode、assistant
session，不会续用失败尝试的会话状态；同一样本的 seed 在各次尝试中保持不变。只有失败才
触发重试，任意一次成功后立即停止。每个样本在批次内使用固定目录；重试前会删除该目录中
上一轮的全部结果，再在同一目录名下从头运行，因此磁盘上只保留最终一次尝试。

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

文件开头的 `run_overview` 是便于快速核对的 setting 摘要：依次列出 difficulty、assistant
baseline，以及 simulator 各组件和 assistant 的 profile 名与实际 model ID。后面的字段保存
同一批次的完整参数和 provenance 信息。

批量中的单次样本失败不会取消其他样本。管线会在同一个并发槽位内按重试上限依次重建并
运行该样本，旧尝试的目录会在下一次尝试开始前删除。`batch_summary.*` 不保存历史尝试明细，
每个样本只记录最终状态、最终目录和实际 `retry_count`；批次顶层记录 `total_retries`、
`total_attempts`、完成数、失败数等聚合结果。`runs[].run_dir` 仍指向最终结果，因此现有的
下游汇总逻辑可以继续使用。

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
- `--simulator-model-profile`：一次覆盖 simulator 的四个 LLM 组件。
- `--assistant-model-profile`：覆盖 assistant 模型。
- `--baseline`：选择 `base` 或 `prompt_base`。
- `--sample-retries`：覆盖单个样本失败后的最大完整重试次数，`0` 表示不重试。

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

管线把一个完整样本视为一个 Evo-Memory task：新样本第一次生成 assistant 回复前执行检索；
样本正常终止后把完整对话、最终回复和终止反馈写回。默认不保存失败或超过 turn limit 的
样本。严格的 test-time learning 任务流应使用 `--concurrency 1`；并发模式的文件写入仍然
安全，但样本间检索和更新顺序会交错。完整配置字段、embedding 检索可选依赖、本地 vLLM
组合方式和记忆 JSON 格式见 `assistant/README_CN.md`。

## 审计语义

`events.jsonl` 保留 simulator 原有事件，并增加：

- `pipeline_run_started` / `pipeline_run_completed` / `pipeline_run_failed`；
- `assistant_generation_requested` / `assistant_generation_completed`。
- 使用 Evo-Memory 时还会记录 `assistant_memory_updated`；持久化失败时记录
  `assistant_memory_update_failed` 并将该样本标为失败；因失败样本或空会话而不满足保存条件
  时记录 `assistant_memory_update_skipped`。

`transcript.jsonl` 只包含实际可见的 user/assistant 对话。Qwen thinking profile 的隐藏
reasoning 不写入 transcript，也不会复制到管线事件中。`events.txt` 是结构化事件的缩进
版，`transcript.txt` 则按轮次和角色排版。

## 离线测试

```bash
PYTHONPATH=src:../assistant/src:../user_simulator/src pytest -q
```

测试使用 simulator mock 和假的 assistant，不访问 OpenRouter，也不会产生费用。
