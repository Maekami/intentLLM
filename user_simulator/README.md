# Reason-DAG User Simulator

This project simulates the user side of a multi-turn conversation from a hidden
reason DAG. It exposes intents gradually, measures only assistant-provided help,
selects unresolved needs according to a difficulty policy, and realizes those
needs as natural user messages. Every model call uses strict structured output
and can be audited without exposing latent context in the normal conversation.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Run commands from the `user_simulator/` directory so relative config and dataset
paths resolve consistently.

## Environment variables

OpenRouter-backed components use:

```bash
export OPENROUTER_API_KEY='...'
export OPENROUTER_HTTP_REFERER='https://example.org'  # optional
export OPENROUTER_APP_TITLE='Reason-DAG User Simulator'  # optional
```

Local vLLM profiles use `VLLM_API_KEY` when their server enables authentication;
otherwise the SDK uses the non-secret placeholder `EMPTY`. The deterministic
mock demo, local unauthenticated vLLM, and contract smoke test do not need an
OpenRouter key.
The following narrowly scoped runtime overrides are also supported:

- `REASON_DAG_DATASET_PATH`
- `REASON_DAG_AUDIT_LEVEL`
- `REASON_DAG_MAX_TURNS`

## Dataset and validation

The canonical dataset is `dataset/DAG.jsonl`. Each JSONL record contains a
sample ID, intent and terminal nodes, edges, and retained task-level context.
The loader validates node identifiers, the natural backbone, END structure,
edge references, and graph constraints.

Do not modify the dataset when changing simulator behavior. Validate it with:

```bash
python -m user_simulator.cli.validate_dataset \
  --dataset dataset/DAG.jsonl \
  --strict
```

## Episode state and turn order

An episode tracks the current frontier, exposed node IDs, a satisfaction label
for every exposed intent, the visible conversation, the latched END-exposure state, the turn
index, termination state, difficulty, and random seed. Latent task context is
not shown in the conversation by default.

The initial turn exposes `N1`, chooses a realization mode, and generates the
first user message. Each later assistant turn follows this exact order:

1. append the assistant response to visible history;
2. evaluate ordered outgoing intent candidates with the controller, unless there
   are no intent candidates or END was exposed on an earlier turn;
3. normalize the result to prefix closure and update exposure/frontier state;
4. let the system derive END exposure from the pre- and post-transition graph
   frontier;
5. evaluate cumulative satisfaction for every exposed intent;
6. apply monotonic satisfaction normalization;
7. check system-owned termination;
8. naturally expose the next backbone node if all exposed nodes are satisfied
   before END is exposed;
9. build the unresolved queue in node order;
10. select nodes according to difficulty;
11. select Clear or Abstract realization mode;
12. generate one user message from the selected nodes.

The episode fails clearly on an empty unresolved queue in a non-terminal state,
an invalid final-node deadlock, or a turn-limit breach.

## END semantics

END is not a user intent: it is never added to `exposed_nodes`, selected,
assigned satisfaction, or generated as a user message. END exposure is a
system-owned structural transition: the controller receives only ordered intent
candidates and returns only intent decisions. After prefix normalization updates
the frontier, the system checks graph closure in the same turn. It exposes END
when the complete current prefix has an outgoing END edge, or when that complete
step advances to a terminal-only frontier. This second check prevents a
multi-node exposure step from reaching the final intent while leaving END one
turn behind. The state is irreversible, and the controller is not called on
later turns. The engine terminates only when END is exposed and every exposed
intent is satisfied.

Controller exposure requires a direct next-reply link. A candidate can be
exposed by useful assistant content, a targeted question whose direct answer
reveals it, or assistant behavior that directly triggers the corresponding
correction, constraint, preference, objection, or redirection. Exposure cannot
use a hypothetical intermediate user turn. Prefix closure only prevents holes;
it never turns a false decision into a true one.

## Satisfaction labels

Satisfaction measures assistance supplied by the assistant across the complete
visible conversation. User disclosure is not satisfaction evidence.

- `unsatisfied`: no meaningful help for the core intent, including a mere
  restatement or clarification-only response.
- `partially_satisfied`: useful help covers a meaningful portion, but an
  important component, constraint, decision, or actionable detail remains.
- `satisfied`: the core need and important stated constraints are sufficiently
  addressed, so the user could reasonably stop asking about that node.

The updater applies a stop-now test: if the conversation ended after the latest
assistant message, some usable assistance for the node must already exist before
it can receive partial credit. Clarification can expose an intent but does not by
itself satisfy it. For interaction-style intents, an actual change in pace,
question load, tone, or communication behavior is itself usable assistance.

When `monotonic_satisfaction` is enabled, the engine prevents later model
evaluations from moving a node backward (`satisfied` to partial/unsatisfied, or
partial to unsatisfied). It retains the previous label and records a
normalization violation. Disabling it applies the latest label even when it
regresses.

Each satisfaction decision includes an audit `reason` and a nullable
`remaining_gap`. The gap must be non-empty only for `partially_satisfied`
nodes. After node selection, only gaps belonging to selected nodes are passed
to the user realizer; reasons and unselected-node gaps are not passed.

`remaining_gap` may describe only a core deliverable explicitly required by the
current node. User-stated facts may specialize an existing requirement but may
not add a new objective. An assistant-introduced prerequisite, optional
personalization input, or information from a later unexposed node cannot become
a hard gap, even if a later user message repeats it. If usable assistance has
covered the node as written and only optional tailoring remains, the node is
`satisfied` rather than `partially_satisfied`.

## Difficulty policies

Selection and realization remain separate policy components.

| Difficulty | Node selection | Realization |
| --- | --- | --- |
| Easy | all unresolved exposed nodes | Clear |
| Medium | seeded random ordered prefix bounded by `medium_min_nodes` and `medium_max_nodes` | seeded Clear/Abstract choice using `medium_clear_probability` |
| Hard | earliest unresolved node only | Abstract |

Easy messages cover all selected needs clearly in short natural wording.
Medium randomness changes neither node order nor the policy-selected set after
selection. Hard messages are sparse and indirect, but retain enough evidence for
the selected need to be recoverable from visible history.

For Medium selection, the prefix length is sampled uniformly from the inclusive
configured range after clamping it to the number of available unresolved nodes.
`medium_max_nodes: null` means all available nodes are eligible and preserves the
original `1..queue length` behavior. Setting both bounds to `n` always selects the
first `n` unresolved nodes, or every available node when fewer than `n` remain.

## Component configuration

The default `configs/simulator.yaml` selects each replaceable component:

```yaml
dataset_path: dataset/DAG.jsonl

components:
  controller: llm_controller
  satisfaction_updater: llm_satisfaction_updater
  selection_policy: difficulty_selection
  realization_policy: difficulty_realization
  user_realizer: llm_user_realizer

prompts:
  controller: configs/prompts/controller.yaml
  satisfaction: configs/prompts/satisfaction.yaml
  realizer_clear: configs/prompts/user_clear.yaml
  realizer_abstract: configs/prompts/user_abstract.yaml

models:
  controller: deepseek_v4_flash_0731
  satisfaction: deepseek_v4_flash_0731
  realizer_clear: deepseek_v4_flash_0731
  realizer_abstract: deepseek_v4_flash_0731
```

Policy and audit settings are in the same file. A custom YAML is deep-merged over
the defaults. CLI overrides take precedence.

Component registries store a `ComponentSpec` with the constructor and an
explicit `requires_model` capability. API-key decisions come from the provider
in each resolved model profile and never depend on a name prefix. The factory
loads only prompts and model profiles needed by the selected concrete
components. Fully mocked configurations create no model client and load no
model-bound prompts or profiles. `SimulatorComponents` reports whether any
resolved profile actually uses OpenRouter.

## Prompts

The four canonical prompts are:

- `configs/prompts/controller.yaml`
- `configs/prompts/satisfaction.yaml`
- `configs/prompts/user_clear.yaml`
- `configs/prompts/user_abstract.yaml`

Each YAML contains only `name`, `schema_name`, `system`, and `user_template`.
Edit a canonical prompt in place, keep all required placeholders, and run the
contract smoke test. Prompt selection comes only from resolved configuration;
LLM component constructors require explicit prompt injection and have no hidden
fallback path.

The rendered prompt hash covers the complete system and rendered user messages.
Audit provenance records `prompt_name`, `prompt_path`, `prompt_hash`,
`schema_name`, `schema_hash`, and the current Git commit when available. Git
history and content hashes, rather than manual revision integers or parallel
files, provide version provenance and reproducibility.

## Human-like user realization

Both realizer modes speak only as the human user. They avoid simulator, prompt,
schema, node, policy, and hidden-state language; minimize user effort; match the
user's apparent vocabulary and knowledge; react to the latest assistant reply;
keep preferences grounded; and remain focused on the original task.

Clear mode directly communicates every selected intent with the shortest
understandable wording, normally one to three sentences. Abstract mode usually
uses one or two sentences and expresses selected needs through a symptom,
consequence, uncertainty, high-level objective, or context-dependent follow-up.
Abstract remains recoverable, not random or adversarial.

Most turns contain no deliberate mistake. A small minority may contain at most
one mild surface imperfection when it fits the established voice. Prompts never
force typos, broken grammar, emotional language, slang, factual errors, invented
identity, or fabricated constraints.

`TASK SUMMARY` and `TASK EXPECTATION` remain internal context for every Clear
and Abstract call, but only as global direction and consistency constraints.
Specific task content in a generated message may come only from selected nodes
or facts already stated by the user in visible history. Assistant assumptions
and examples do not become user facts. Abstract realization may remove
specificity but cannot add it, and unspecified slots remain unspecified.

Selected nodes alone determine which unresolved intents are expressed in the
current turn. To reduce prompt leakage, the realizer receives only IDs—not
semantic details—for unselected unresolved nodes.

The realizer's coverage and unsupported-task-content fields are model-reported checks.
They support diagnostics and retries but are not independent proof of
faithfulness or human-likeness. A planned blind human discrimination study and
blind faithfulness evaluation remain the authoritative behavioral evaluation.

The focused regression cases in `tests/fixtures/review_regressions.jsonl` cover
assistant-triggered corrections, direct versus hypothetical exposure,
clarification-only satisfaction, interaction-style satisfaction, and
unspecified-slot preservation.

## Strict structured output

The canonical result models live in `src/user_simulator/domain/results.py` and
inherit from one strict Pydantic base (`extra="forbid"`, `strict=True`). Schema
names uniquely identify `ControllerResult`, `SatisfactionUpdateResult`, and
`UserGenerationResult` in `llm/schema_registry.py`.

Every OpenRouter or local vLLM request enforces the same `response_format`:

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": schema_name,
        "strict": True,
        "schema": recursive_closed_json_schema,
    },
}
stream = False
```

OpenRouter requests additionally set
`extra_body["provider"]["require_parameters"] = True`. Local vLLM requests
omit OpenRouter routing, reasoning, and plugin fields; Qwen thinking controls
are sent through `chat_template_kwargs` instead.

Schemas recursively set `additionalProperties: false` and have canonical
content hashes. Responses are parsed with Pydantic `model_validate_json`.
There is no local regex recovery, `json_object` downgrade, or free-text
fallback. Model profiles may additionally set
`structured_output.response_healing: true`; the default DeepSeek V4 Flash 0731
profile enables OpenRouter's non-streaming malformed-JSON repair layer. This
layer handles syntax and wrappers only and does not replace schema or semantic
validation.

No choices, refusal, empty content, truncation, invalid JSON, additional
properties, invalid enums, missing fields, and retry exhaustion all fail or
retry through the structured path. A structured retry includes the prior
invalid output and exact local validation errors so the model can replace it
under the same schema. Component-level semantic validation can
request a corrected complete structured result without weakening the transport
contract.

## Demo

Run the deterministic offline demo:

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --mock \
  --audit-level full
```

Run the configured live components after setting `OPENROUTER_API_KEY`:

```bash
python -m user_simulator.cli.demo --random-sample --difficulty easy --seed 42
```

Useful overrides include `--controller-component`,
`--satisfaction-component`, `--realizer-component`, `--selection-policy`,
`--realization-policy`, the four prompt options, `--model-profile`,
`--max-turns`, and `--config`.

During a demo, `/audit`, `/raw-audit`, `/prompts`, `/schemas`, `/state`, `/dag`,
`/nodes`, `/history`, `/save`, and `/quit` are available. Latent summary display
requires the explicit `--show-latent-summary` option.

Assistant replies use multiline input. Press Enter to add a line, preserve blank
lines normally, and type `/send` on its own line to submit the complete reply.
Slash commands entered as the first line execute immediately without `/send`.

## Audit

Each enabled run writes under `runs/<run-id>/`:

- `config_snapshot.yaml`
- `events.jsonl`
- `transcript.jsonl`
- `final_state.json` after save, termination, or failure

Per-call audit metadata includes component, model ID/profile, prompt and schema
names/hashes, prompt path, Git commit, provider request data, finish reason,
latency, input/output/thinking/answer token counts, transport and semantic retry
counts, structured and semantic validation status, and normalization violations.

OpenRouter reports thinking tokens as part of `output_tokens`. When the provider
supplies `completion_tokens_details.reasoning_tokens`, the audit also records
`thinking_tokens` and computes `answer_tokens = output_tokens - thinking_tokens`.
If the provider omits that breakdown, both split fields remain `null` rather than
guessing from the configured reasoning mode.

`--audit-level full` retains rendered messages and structured payloads for local
debugging. `--audit-level summary` hides rendered prompts/messages, summarizes
raw structured responses, hides latent node text, and redacts API keys and
authorization fields. It never records a dirty patch or full Git diff.

## Prompt smoke tests

Contract mode is deterministic and suitable for CI:

```bash
python -m user_simulator.cli.prompt_smoke_test \
  --mode contract \
  --component all \
  --cases tests/fixtures/prompt_cases.jsonl \
  --output-dir runs/prompt_smoke
```

It loads configured canonical prompts, renders actual component inputs, verifies
placeholder and metadata contracts, and validates optional fixture responses
plus semantic invariants. It makes no API request and reports no retry rate.
Metrics include prompt-render, schema-fixture, semantic-fixture rates and the
prefix-normalization violation count.

Live mode requires `OPENROUTER_API_KEY`:

```bash
python -m user_simulator.cli.prompt_smoke_test \
  --mode live \
  --component all \
  --cases tests/fixtures/prompt_cases.jsonl \
  --output-dir runs/prompt_smoke_live
```

It invokes the real configured components through strict JSON Schema requests,
saves actual structured responses, performs component semantic validation, and
reports real transport and semantic retries. User metrics are explicitly named
as model-reported coverage and unsupported-task-content checks. Character,
meta-language, over-elaboration, and repeated-opening checks are lightweight
diagnostics only.

## Tests

Run the full suite:

```bash
pytest -q
```

The suite covers dataset contracts, graph navigation, prefix closure, END
normalization, monotonic satisfaction, natural backbone exposure, seeded policy
reproducibility, all difficulty modes, termination/deadlock behavior, schema
strictness, provider failures and retries, audit privacy, lazy factory resource
loading, prompt contracts, realizer context, smoke modes, and the offline demo.
Live tests are skipped when `OPENROUTER_API_KEY` is absent.

## Extending with ablations

Implement the existing controller, satisfaction, selection, realization, or
realizer protocol; register the constructor under a descriptive canonical name;
and declare `requires_model` in its `ComponentSpec`. Do not encode
capabilities in naming conventions. Reuse canonical result models where the
wire contract is unchanged. Add any genuinely different structured output to
the canonical schema registry by unique schema name, inject required prompts
through configuration, and keep resource loading lazy.

For reproducible ablations, record resolved config, model profile, seed, prompt
and schema hashes, and Git commit. Keep latent context out of the visible
conversation and retain the same blind human and faithfulness evaluation plan.
