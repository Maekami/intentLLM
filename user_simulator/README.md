
# Reason-DAG User Simulator

## 1. Objective

Implement a modular user simulator grounded in the reason-based DAG dataset located at:

```text
user_simulator/dataset/DAG.jsonl
```

The file currently contains 292 processed single-complex-task samples.

Each episode simulates a user whose latent intents are represented by an ordered DAG. The system progressively exposes intent nodes, updates the satisfaction state of all exposed nodes, selects which unresolved intents the user expresses next, and generates the next user message.

The implementation must prioritize:

* modularity for component-level ablations;
* deterministic and auditable state transitions;
* strict structured LLM outputs;
* easy replacement of models, prompts, controllers, policies, and satisfaction updaters;
* an interactive demo for manually acting as the assistant;
* no modification of the source dataset.

---

## 2. Core Simulator Semantics

Each sample contains ordered intent nodes:

```text
N1, N2, ..., Nk
```

and one explicit terminal node:

```text
END
```

The original trajectory forms a backbone:

```text
N1 -> N2 -> ... -> Nk -> END
```

Additional DAG edges represent possible exposure paths that may advance the frontier by more than one intent node.

### 2.1 Runtime state

Each episode must maintain at least:

```python
EpisodeState(
    sample_id: str,
    difficulty: Difficulty,
    current_frontier: str,
    exposed_nodes: list[str],
    satisfaction: dict[str, SatisfactionLevel],
    conversation_history: list[ChatMessage],
    end_reachable: bool,
    terminated: bool,
    turn_index: int,
    random_seed: int,
)
```

Use these enums:

```python
class SatisfactionLevel(str, Enum):
    UNSATISFIED = "unsatisfied"
    PARTIALLY_SATISFIED = "partially_satisfied"
    SATISFIED = "satisfied"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RealizationMode(str, Enum):
    CLEAR = "clear"
    ABSTRACT = "abstract"
```

`END` must never:

* appear in `exposed_nodes`;
* receive a satisfaction state;
* enter the unresolved-node queue;
* be sent to the user-message generator.

It is only a system-level termination permission.

---

## 3. Episode Initialization

At episode start:

```text
current_frontier = N1
exposed_nodes = [N1]
satisfaction[N1] = unsatisfied
end_reachable = false
terminated = false
```

Generate the initial user message from `N1`.

Use the same realization policy as later turns:

| Difficulty | Initial realization        |
| ---------- | -------------------------- |
| Easy       | Clear                      |
| Medium     | Randomly Clear or Abstract |
| Hard       | Abstract                   |

The generator may use the node's:

* `node_intent`;
* `reason_text`;
* `surface_user_message`;
* task summary and expectation;
* retained metadata when relevant.

It must not simply copy the original surface message unless that is the most natural realization.

---

## 4. Turn Execution Order

After the human or agent supplies an assistant response, execute the following steps in exactly this order.

### Step 1: Append the assistant response

Append the response to `conversation_history`.

Do not classify it as an answer, clarification question, or mixed response. Every assistant input follows the same controller and satisfaction-update pipeline.

### Step 2: Obtain sorted outgoing targets

Read all outgoing targets from `current_frontier`.

Separate them into:

```text
intent_targets
END
```

Sort intent targets by numeric node index. Treat `END` as logically occurring after all intent targets.

Example:

```text
current_frontier = N2
outgoing targets = [N3, N4, N5, END]
```

### Step 3: Run the controller

The controller receives:

* full conversation history;
* latest assistant response;
* current frontier;
* currently exposed nodes;
* current satisfaction states;
* sorted outgoing intent targets;
* node contents for the candidates;
* whether an outgoing END edge exists.

The controller decides which candidate intent nodes can be exposed and whether END is structurally reachable.

#### Prefix-closure rule

Candidate intent nodes must be evaluated from lowest to highest index.

Once one candidate cannot be exposed, all later candidates are automatically treated as not exposable and do not need further semantic evaluation.

Example:

```text
N3: exposable
N4: exposable
N5: not exposable
```

Final result:

```text
newly exposed = [N3, N4]
N5 and all later targets = not exposed
END = not reachable
```

The output must always form a prefix of the sorted candidate list.

The implementation must enforce prefix closure in code even if the model output violates it.

#### END rule

`end_reachable` may be true only when:

1. an outgoing edge from the current frontier to END exists;
2. every preceding outgoing intent candidate is exposable under prefix closure.

The controller does not terminate the episode. It only returns whether END is currently structurally reachable.

### Step 4: Apply exposure updates

For every newly exposed intent node:

```text
add it to exposed_nodes
initialize satisfaction to unsatisfied
```

If at least one new intent node is exposed:

```text
current_frontier = highest-index newly exposed node
```

Otherwise, keep the current frontier unchanged.

Store the controller's END result in:

```text
state.end_reachable
```

Do not add END to `exposed_nodes`.

### Step 5: Update satisfaction

Run the satisfaction updater after all controller-selected nodes have been exposed.

The updater receives:

* full conversation history;
* latest assistant response;
* every currently exposed intent node;
* previous satisfaction states.

It must return exactly one updated state for every exposed node:

```text
unsatisfied
partially_satisfied
satisfied
```

The updater evaluates cumulative satisfaction from the complete interaction, not only the latest response.

For the initial version, satisfaction should be monotonic:

```text
unsatisfied -> partially_satisfied -> satisfied
```

A node must not move backward unless a future ablation explicitly enables satisfaction regression.

This monotonic rule must be implemented outside the LLM response parser so it cannot be violated by model output.

A clarification question will usually leave satisfaction unchanged, but this must emerge from the same satisfaction updater rather than from a separate assistant-action classifier.

### Step 6: System-level termination

After satisfaction has been updated, calculate:

```python
all_exposed_satisfied = all(
    satisfaction[node_id] == SatisfactionLevel.SATISFIED
    for node_id in exposed_nodes
)
```

Terminate the episode only when:

```text
end_reachable == true
AND
all_exposed_satisfied == true
```

When this condition holds:

* set `terminated = true`;
* do not invoke the user-message generator;
* write the final audit state;
* return a terminal result to the caller.

### Step 7: Natural backbone exposure

A deadlock can occur when:

```text
end_reachable == false
AND
all exposed intent nodes are satisfied
```

In this case, there is no unresolved node from which the next user message can be generated.

Resolve this by automatically exposing the next backbone node:

```text
current_frontier = Ni
next backbone node = N(i+1)
```

Then:

```text
expose N(i+1)
satisfaction[N(i+1)] = unsatisfied
current_frontier = N(i+1)
```

This automatic exposure:

* does not call the controller again;
* exposes exactly one node;
* follows only the mandatory backbone;
* occurs after satisfaction updating and before queue construction.

If `current_frontier` is already `Nk`, all exposed nodes are satisfied, and END is not reachable, raise a clear runtime error because the episode is in an invalid terminal state.

### Step 8: Build the unresolved queue

Construct:

```python
unresolved_queue = [
    node_id
    for node_id in exposed_nodes
    if satisfaction[node_id] != SatisfactionLevel.SATISFIED
]
```

Sort it by numeric node index.

Both of these states remain in the queue:

```text
unsatisfied
partially_satisfied
```

Example:

```text
N1 = satisfied
N2 = partially_satisfied
N3 = unsatisfied
N4 = satisfied
N5 = partially_satisfied
```

Queue:

```text
[N2, N3, N5]
```

The queue must be non-empty before node selection.

### Step 9: Select nodes according to difficulty

#### Easy

Select every unresolved exposed node:

```python
selected_nodes = unresolved_queue
```

#### Medium

Uniformly sample a prefix length:

```python
prefix_length = rng.randint(1, len(unresolved_queue))
selected_nodes = unresolved_queue[:prefix_length]
```

This is a random ordered prefix, not an arbitrary subset.

For:

```text
[N2, N4, N5]
```

valid selections are:

```text
[N2]
[N2, N4]
[N2, N4, N5]
```

Invalid selections include:

```text
[N4]
[N2, N5]
[N4, N5]
```

#### Hard

Select only the earliest unresolved node:

```python
selected_nodes = unresolved_queue[:1]
```

### Step 10: Select realization mode

| Difficulty | Realization mode           |
| ---------- | -------------------------- |
| Easy       | Clear                      |
| Medium     | Randomly Clear or Abstract |
| Hard       | Abstract                   |

For Medium, use a configurable probability with the initial default:

```text
P(clear) = 0.5
P(abstract) = 0.5
```

All random operations must use the episode's seeded random-number generator.

### Step 11: Generate the next user message

The realizer receives:

* selected node IDs and their full node contents;
* each selected node's satisfaction state;
* complete conversation history;
* latest assistant response;
* realization mode;
* task summary and expectation;
* explicit instructions not to express unselected nodes.

#### Clear realization

The message must:

* state the selected intents directly;
* retain important goals, constraints, and preferences;
* naturally combine multiple selected nodes;
* focus on still-unresolved aspects;
* avoid requiring unnecessary inference from the assistant.

#### Abstract realization

The message must:

* express selected intents at a more abstract, indirect, or incomplete level;
* avoid directly copying `node_intent`;
* potentially express a symptom, concern, high-level objective, or indirect clue;
* remain semantically consistent with every selected node;
* preserve enough information for the message to remain meaningful;
* avoid becoming unrelated or impossible to interpret.

#### Shared constraints

Both modes must:

* cover every selected node;
* avoid expressing unselected exposed nodes;
* avoid introducing intents absent from the DAG;
* remain consistent with the conversation history;
* avoid repeating already satisfied portions of a partially satisfied node;
* produce one natural user message rather than a list of annotations.

Append the generated user message to the conversation history and return it.

---

## 5. Modular Architecture

Implement components behind explicit interfaces so each can be replaced independently in later ablations.

Recommended project structure:

```text
user_simulator/
├── README.md
├── pyproject.toml
├── .env.example
├── dataset/
│   └── DAG.jsonl
├── configs/
│   ├── simulator.yaml
│   ├── models/
│   │   └── deepseek_v4_pro.yaml
│   └── prompts/
│       ├── controller.yaml
│       ├── satisfaction.yaml
│       ├── user_clear.yaml
│       └── user_abstract.yaml
├── src/
│   └── user_simulator/
│       ├── __init__.py
│       ├── config.py
│       ├── domain/
│       │   ├── enums.py
│       │   ├── dag.py
│       │   ├── state.py
│       │   ├── messages.py
│       │   └── results.py
│       ├── data/
│       │   ├── loader.py
│       │   └── validator.py
│       ├── graph/
│       │   ├── navigator.py
│       │   └── validation.py
│       ├── llm/
│       │   ├── base.py
│       │   ├── openrouter_client.py
│       │   ├── structured_call.py
│       │   ├── schemas.py
│       │   └── retry.py
│       ├── controller/
│       │   ├── base.py
│       │   └── llm_controller.py
│       ├── satisfaction/
│       │   ├── base.py
│       │   └── llm_updater.py
│       ├── policy/
│       │   ├── selection.py
│       │   └── realization.py
│       ├── realizer/
│       │   ├── base.py
│       │   └── llm_realizer.py
│       ├── engine/
│       │   ├── episode.py
│       │   ├── transitions.py
│       │   └── termination.py
│       ├── audit/
│       │   ├── events.py
│       │   ├── logger.py
│       │   └── console_renderer.py
│       └── cli/
│           ├── demo.py
│           └── validate_dataset.py
├── tests/
│   ├── unit/
│   │   ├── test_dataset_validation.py
│   │   ├── test_graph_navigation.py
│   │   ├── test_prefix_closure.py
│   │   ├── test_satisfaction_updates.py
│   │   ├── test_selection_policy.py
│   │   ├── test_termination.py
│   │   └── test_natural_exposure.py
│   ├── integration/
│   │   ├── test_episode_with_mock_llm.py
│   │   └── test_structured_output_retry.py
│   └── live/
│       └── test_openrouter_live.py
└── runs/
    └── .gitkeep
```

The names may be adjusted slightly, but component boundaries must remain separate.

### Required interfaces

Define replaceable interfaces or protocols for:

```python
class Controller(Protocol):
    async def decide(...) -> ControllerResult: ...


class SatisfactionUpdater(Protocol):
    async def update(...) -> SatisfactionUpdateResult: ...


class NodeSelectionPolicy(Protocol):
    def select(...) -> SelectionResult: ...


class RealizationPolicy(Protocol):
    def choose_mode(...) -> RealizationMode: ...


class UserRealizer(Protocol):
    async def generate(...) -> UserGenerationResult: ...


class StructuredLLMClient(Protocol):
    async def generate_structured(...) -> BaseModel: ...
```

The episode engine must depend on these interfaces rather than concrete implementations.

---

## 6. Dataset Loading and Validation

Load:

```text
dataset/DAG.jsonl
```

Do not modify or rewrite it.

Validate every sample before use:

* valid JSON;
* unique `sample_id`;
* exactly one `END` node;
* intent IDs follow `N1...Nk`;
* no duplicate node IDs;
* no duplicate edges;
* every edge endpoint exists;
* all edges point forward;
* no self-loops;
* no cycles;
* mandatory backbone edges exist;
* END has no outgoing edge;
* every intent node can reach END;
* N1 can reach END.

Also validate the expected prefix-closure property of outgoing skip targets.

For a source `Ni`, if an edge to `Nj` exists, earlier forward targets between them should also exist. Report violations clearly. Treat this as a configurable warning or strict error, not a silent correction.

The loader must support:

```python
load_all_samples()
load_sample_by_id(sample_id)
sample_randomly(seed)
```

Do not hard-code the count of 292 in runtime logic.

---

## 7. OpenRouter Model Integration

### 7.1 Default model

Use the default model for:

* controller;
* satisfaction updater;
* clear user realizer;
* abstract user realizer.

```text
deepseek/deepseek-v4-pro
```

Each component must reference a model profile so later ablations can assign different models to different components.

### 7.2 Environment variables

Create `.env.example`:

```bash
OPENROUTER_API_KEY=
OPENROUTER_HTTP_REFERER=
OPENROUTER_APP_TITLE=Reason-DAG User Simulator
```

Never commit real API keys.

### 7.3 Model profile

Create:

```text
configs/models/deepseek_v4_pro.yaml
```

Suggested contents:

```yaml
profile_name: deepseek_v4_pro
provider: openrouter
model_id: deepseek/deepseek-v4-pro
base_url: https://openrouter.ai/api/v1

routing:
  require_parameters: true
  allow_fallbacks: true

reasoning:
  effort: high
  exclude_from_response: true

generation:
  controller:
    temperature: 0.0
    max_completion_tokens: 1600
  satisfaction:
    temperature: 0.0
    max_completion_tokens: 2000
  realizer_clear:
    temperature: 0.3
    max_completion_tokens: 800
  realizer_abstract:
    temperature: 0.7
    max_completion_tokens: 800

structured_output:
  type: json_schema
  strict: true

retry:
  max_attempts: 3
  initial_backoff_seconds: 1.0
  maximum_backoff_seconds: 8.0
```

All model-bound settings must live in configuration rather than being scattered across implementation files.

The code must allow command-line or YAML overrides.

### 7.4 Client implementation

Use the asynchronous OpenAI Python SDK with OpenRouter's API base URL.

Conceptual setup:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)
```

For every model call:

* use `stream=False`;
* provide a strict JSON Schema response format;
* set OpenRouter provider routing to require all requested parameters;
* validate the returned JSON with a Pydantic model;
* retry on transport errors, invalid structured output, empty output, or schema validation failure;
* never parse free-form prose with regex as the normal path;
* never silently accept missing or hallucinated fields.

Conceptual request structure:

```python
response = await client.chat.completions.create(
    model=model_profile.model_id,
    messages=messages,
    temperature=temperature,
    max_completion_tokens=max_completion_tokens,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": pydantic_model.model_json_schema(),
        },
    },
    extra_body={
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": True,
        },
        "reasoning": {
            "effort": "high",
            "exclude": True,
        },
    },
)
```

Adapt field names only when required by the installed SDK version.

All strict schemas must:

* use `type: object`;
* list required fields explicitly;
* set `additionalProperties: false` wherever supported;
* use enums for finite labels;
* include short field descriptions.

At startup, optionally query OpenRouter model metadata and log whether `response_format` or structured outputs are listed as supported. The actual request must still use `require_parameters: true`, because endpoint-level support may differ.

### 7.5 Structured-output schemas

#### Controller output

```python
class CandidateExposureDecision(BaseModel):
    node_id: str
    exposable: bool
    reason: str


class ControllerResult(BaseModel):
    decisions: list[CandidateExposureDecision]
    end_reachable: bool
    summary: str
```

Requirements:

* `decisions` must contain each sorted outgoing intent candidate exactly once;
* no unknown node IDs;
* reasons must be short audit explanations, not hidden chain-of-thought;
* code must normalize results using prefix closure;
* later `true` decisions after the first `false` must be forced to `false` and recorded as a model-output violation;
* `end_reachable` must be forced to `false` if END is not an outgoing target or prefix closure is incomplete.

#### Satisfaction output

```python
class NodeSatisfactionDecision(BaseModel):
    node_id: str
    status: SatisfactionLevel
    reason: str


class SatisfactionUpdateResult(BaseModel):
    updates: list[NodeSatisfactionDecision]
    summary: str
```

Requirements:

* every exposed intent node appears exactly once;
* no END entry;
* no unknown node IDs;
* code enforces monotonic satisfaction;
* any missing or duplicate node causes schema-level or semantic validation failure and retry.

#### User-generation output

```python
class UserGenerationResult(BaseModel):
    user_message: str
    selected_node_ids: list[str]
    realization_mode: RealizationMode
    coverage_check: dict[str, bool]
    contains_unsupported_intent: bool
    summary: str
```

`coverage_check` maps every selected node ID to whether the generated message expresses it.

Requirements:

* `selected_node_ids` must exactly match the policy-selected nodes;
* `realization_mode` must match the system-selected mode;
* every selected node must appear in `coverage_check`;
* `contains_unsupported_intent` should be false;
* `user_message` must be non-empty;
* invalid self-checks should trigger one regeneration attempt with explicit correction feedback.

The realizer's self-check is for runtime validation and auditing, not a substitute for later human evaluation.

---

## 8. Prompt Organization

Store prompts outside Python source files:

```text
configs/prompts/controller.yaml
configs/prompts/satisfaction.yaml
configs/prompts/user_clear.yaml
configs/prompts/user_abstract.yaml
```

Each prompt file should contain:

```yaml
name: controller_v1
version: 1
system: |
  ...
user_template: |
  ...
```

Every model call must log:

* prompt name;
* prompt version;
* model profile;
* rendered prompt hash.

Prompts must instruct the model to return only the schema-conforming result, even though JSON Schema enforcement is also enabled.

### Controller prompt requirements

The controller prompt must explain:

* an edge means that a target may be exposed from the current frontier;
* exposure can be justified by either a useful answer or a clarification question;
* assistant action type must not be classified;
* candidates are considered in index order;
* prefix closure is mandatory;
* once a candidate fails, later candidates and END are unavailable;
* END means structural reachability, not immediate termination.

### Satisfaction prompt requirements

The satisfaction prompt must explain:

* evaluate all exposed nodes;
* use the complete conversation history;
* judge cumulative satisfaction;
* use only the three discrete labels;
* a clarification question may leave states unchanged;
* partially useful answers should normally map to `partially_satisfied`;
* do not evaluate END.

### User-realizer prompt requirements

The clear and abstract prompts must share all grounding and faithfulness constraints but differ in realization instructions.

Both must receive:

* selected nodes;
* unselected unresolved nodes;
* satisfaction states;
* full conversation history;
* latest assistant response;
* task summary and expectation.

The prompt must explicitly forbid expressing unselected intents.

---

## 9. Audit System

Every episode must produce an auditable event stream.

Use an append-only JSONL log:

```text
runs/<run_id>/events.jsonl
```

Also save:

```text
runs/<run_id>/transcript.jsonl
runs/<run_id>/final_state.json
runs/<run_id>/config_snapshot.yaml
```

### Required audit events

At minimum, log:

```text
episode_started
initial_user_generated
assistant_message_received
controller_requested
controller_raw_result
controller_prefix_normalized
nodes_exposed
satisfaction_requested
satisfaction_raw_result
satisfaction_applied
termination_checked
backbone_node_auto_exposed
unresolved_queue_built
nodes_selected
realization_mode_selected
user_generation_requested
user_generation_raw_result
user_message_generated
episode_terminated
episode_failed
```

Each event should include:

```python
AuditEvent(
    event_type: str,
    timestamp: str,
    sample_id: str,
    turn_index: int,
    payload: dict,
)
```

For every LLM request, record:

* component name;
* model ID;
* prompt name and version;
* request ID when available;
* latency;
* input and output token usage when available;
* retry count;
* structured validation status;
* concise raw structured response;
* provider metadata when available.

Never log the API key.

Provide two audit levels:

```text
summary
full
```

`summary` hides full rendered prompts.
`full` stores rendered prompts and structured model responses for debugging.

---

## 10. Interactive Demo

Implement a terminal demo that lets a human act as the assistant while the simulator acts as the user.

Example command:

```bash
python -m user_simulator.cli.demo \
  --sample-id user1006_task1_conversation1 \
  --difficulty medium \
  --seed 42 \
  --audit-level full
```

Also support:

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty easy \
  --seed 42
```

### Demo loop

1. Load and validate the selected sample.
2. Display basic sample metadata without revealing all latent intents by default.
3. Generate and display the initial user message.
4. Prompt the human for an assistant response.
5. Run the complete controller, satisfaction, system, selection, and realization pipeline.
6. Display the next user message or termination.
7. Repeat until termination, explicit quit, or maximum turns.

### Audit display

Use a readable terminal UI, preferably with `rich`.

After every assistant response, display an audit panel containing:

```text
Turn
Current frontier before/after
Outgoing candidates
Controller decisions
Prefix-closure normalization
Newly exposed nodes
END reachable
Satisfaction before/after
All exposed satisfied
Automatic backbone exposure, if any
Unresolved queue
Selected nodes
Realization mode
Generated user message
Model latency and token usage
```

Use commands during the demo:

```text
/audit       show the latest full audit
/state       show current simulator state
/dag         show node IDs and edges
/nodes       show exposed node details
/history     show conversation transcript
/save        flush current logs
/quit        end the demo
```

By default, `/dag` and `/nodes` may expose latent simulator information because this demo is for debugging. Clearly label them as audit-only views that an evaluated assistant would not receive.

The demo must not send audit information to the human assistant as part of the simulated conversation.

---

## 11. Configuration

Create `configs/simulator.yaml`:

```yaml
dataset_path: dataset/DAG.jsonl

components:
  controller: llm_controller
  satisfaction_updater: llm_satisfaction_updater
  user_realizer: llm_user_realizer
  selection_policy: difficulty_selection_v1
  realization_policy: difficulty_realization_v1

models:
  controller: deepseek_v4_pro
  satisfaction: deepseek_v4_pro
  realizer_clear: deepseek_v4_pro
  realizer_abstract: deepseek_v4_pro

policy:
  medium_clear_probability: 0.5
  monotonic_satisfaction: true
  auto_expose_backbone_on_empty_queue: true
  enforce_controller_prefix_closure: true
  max_turns: 20

audit:
  enabled: true
  level: full
  output_dir: runs
```

Configuration precedence:

```text
default YAML < custom YAML < environment variables < CLI arguments
```

Save the fully resolved configuration with every run.

---

## 12. Error Handling

Use explicit exceptions for:

```text
DatasetValidationError
GraphInvariantError
StructuredOutputError
ControllerOutputError
SatisfactionOutputError
UserGenerationError
InvalidTerminalStateError
EpisodeTurnLimitError
OpenRouterRequestError
```

The engine must never silently continue after:

* malformed DAG data;
* missing structured fields;
* unknown node IDs;
* impossible prefix decisions;
* absent satisfaction decisions;
* an empty unresolved queue that cannot be resolved;
* END inconsistency;
* repeated LLM schema failures.

Retries are appropriate only for transient API failures or invalid model output. Logical state errors should fail immediately with an audit event.

---

## 13. Testing Requirements

### Unit tests

Test without network calls:

* dataset parsing and validation;
* END exclusion from exposed nodes;
* ordered outgoing target retrieval;
* controller prefix closure;
* controller END normalization;
* satisfaction enum parsing;
* monotonic satisfaction updates;
* unresolved queue construction;
* Easy selection;
* deterministic Medium random-prefix selection;
* Hard earliest-node selection;
* Medium realization sampling;
* termination after END plus complete satisfaction;
* no termination when END is reachable but nodes remain unresolved;
* automatic backbone exposure when the queue would otherwise be empty;
* invalid final-node deadlock detection;
* structured-output semantic validation.

### Integration tests

Use a deterministic mock LLM client to test complete episodes.

At least one fixture should cover:

```text
N1 -> N2 -> N3 -> END
N1 -> N3
N2 -> END
```

Test:

1. a direct helpful answer exposing multiple nodes;
2. a clarification question exposing nodes without improving satisfaction;
3. partially satisfied nodes remaining in the queue;
4. Easy, Medium, and Hard producing different selections;
5. END becoming reachable before satisfaction is complete;
6. natural backbone exposure;
7. final termination.

### Live tests

Place OpenRouter-dependent tests under:

```text
tests/live/
```

They must:

* be skipped unless `OPENROUTER_API_KEY` is set;
* make a minimal number of calls;
* verify strict structured output for all three LLM components;
* never run as part of ordinary unit tests.

---

## 14. Command-Line Utilities

Implement:

```bash
python -m user_simulator.cli.validate_dataset
```

Options:

```text
--dataset
--strict
--show-warnings
```

Implement:

```bash
python -m user_simulator.cli.demo
```

Options:

```text
--sample-id
--random-sample
--difficulty
--seed
--config
--model-profile
--audit-level
--max-turns
```

Add a concise `--help` description for every argument.

---

## 15. Dependencies

Use a modern Python version and include at least:

```text
openai
pydantic
pydantic-settings
pyyaml
python-dotenv
tenacity
typer
rich
pytest
pytest-asyncio
```

Use `pyproject.toml` for package configuration and dependencies.

Use type hints throughout. The code should pass a standard formatter and linter.

---

## 16. Implementation Order

Implement in this order:

1. domain models and enums;
2. dataset loader and DAG validator;
3. graph navigator;
4. configuration loading;
5. structured OpenRouter client;
6. mock LLM client;
7. controller;
8. satisfaction updater;
9. selection and realization policies;
10. user realizer;
11. episode engine;
12. audit logging;
13. interactive demo;
14. unit and integration tests;
15. optional live OpenRouter tests.

Do not begin with the interactive demo before the state engine and mock integration tests are complete.

---

## 17. Acceptance Criteria

The implementation is complete when:

* `dataset/DAG.jsonl` can be loaded and validated without modification;
* components are independently replaceable;
* all LLM calls use strict structured outputs;
* all default LLM components use `deepseek/deepseek-v4-pro`;
* model-specific parameters are isolated in model configuration;
* controller output is prefix-closed after code-level normalization;
* satisfaction uses exactly three discrete states;
* END is handled only by the system;
* natural backbone exposure prevents an empty unresolved queue;
* Easy, Medium, and Hard follow the specified policies;
* all random behavior is reproducible from a seed;
* the interactive demo supports real-time conversation;
* every internal decision is auditable;
* unit tests require no network;
* live tests are opt-in;
* no API secrets are logged or committed;
* the source DAG dataset remains unchanged.

---

## 18. Non-Goals for Version 1

Do not implement yet:

* dynamic user personality;
* demographic-conditioned policies;
* continuous satisfaction scores;
* arbitrary Medium subsets;
* node-importance ranking;
* emergent intent creation;
* dynamic difficulty changes;
* satisfaction regression by default;
* learned selection policies;
* agent evaluation metrics;
* training pipelines;
* web search or external tools inside the simulator;
* a web UI.

Keep extension points for these features, but do not add unnecessary complexity to Version 1.

---

## 19. Running the Implementation

Install the package in editable mode:

```bash
python -m pip install -e '.[test]'
```

Validate the dataset and run the offline test suite:

```bash
python -m user_simulator.cli.validate_dataset --show-warnings
pytest -q
```

To use the interactive demo, copy `.env.example` to `.env`, set
`OPENROUTER_API_KEY`, and run:

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42
```

Live OpenRouter tests are skipped unless `OPENROUTER_API_KEY` is set:

```bash
pytest -q -m live
```
