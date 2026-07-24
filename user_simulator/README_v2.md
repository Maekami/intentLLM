
# README_v2 — Reason-DAG User Simulator Revision Specification

## 0. Purpose

This document specifies the required V2 revision of the existing implementation in:

```text
user_simulator/
```

The current implementation already contains a useful modular skeleton and most of the agreed episode-state logic. Do **not** rewrite the project from scratch. Preserve working behavior and make targeted revisions.

The primary V2 goals are:

1. strengthen all four LLM prompts with operational, testable decision criteria;
2. harden and version the structured-output protocol already present in the repository;
3. make prompt and component replacement genuinely configuration-driven for ablations;
4. improve the interactive demo so that each internal decision is readable and auditable;
5. add tests that verify request payloads, schema contracts, component factories, and critical prompt semantics.

The source dataset remains:

```text
dataset/DAG.jsonl
```

Do not modify or rewrite the dataset.

---

## 1. Current implementation: preserve these parts

The current repository already implements or largely implements the following. Preserve them unless a change below explicitly requires refactoring:

- ordered DAG loading and validation;
- `N1 ... Nk` intent nodes plus system-level `END`;
- three discrete satisfaction states:
  - `unsatisfied`
  - `partially_satisfied`
  - `satisfied`
- controller prefix-closure normalization;
- monotonic satisfaction normalization;
- system termination only when `end_reachable` is true and all exposed nodes are satisfied;
- automatic next-backbone exposure when all exposed nodes are satisfied but END is unavailable;
- Easy, Medium, and Hard selection policies;
- Medium random-prefix selection;
- Clear and Abstract realization modes;
- seeded episode randomness;
- OpenRouter integration;
- Pydantic validation;
- append-only audit events;
- terminal interactive demo commands.

### Important structured-output clarification

The current `OpenRouterStructuredClient` already sends:

```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": schema_name,
        "strict": True,
        "schema": response_model.model_json_schema(),
    },
}
```

and uses Pydantic models with `extra="forbid"`.

Therefore, V2 must **not** replace this with plain JSON mode or prompt-only JSON instructions. The task is to make the existing structured-output path complete, configurable, versioned, testable, and robust.

---

## 2. Main V2 gaps to fix

### P0 — Prompt semantics are under-specified

The existing prompt files are too short to reliably define:

- what evidence permits an intent node to be exposed;
- the difference between semantic relevance and actual exposability;
- how clarification questions can expose nodes without satisfying them;
- what `unsatisfied`, `partially_satisfied`, and `satisfied` mean operationally;
- that satisfaction must be based on assistant contributions rather than the user's own disclosure;
- how partially satisfied nodes should be expressed in the next user message;
- what makes an Abstract realization indirect but still faithful;
- how to prevent leakage of unselected latent intents;
- how the initial user turn differs from a follow-up turn.

Replace the four prompts with the V2 prompts in Section 5.

### P0 — Structured output is not fully bound to configuration

The current request hard-codes `json_schema` and `strict=True`, while the model YAML also declares `structured_output`. This creates a configuration field that is not the actual source of runtime behavior.

V2 must:

- define a typed structured-output configuration;
- use it in every request;
- validate that strict JSON Schema mode is enabled at startup;
- attach each prompt to a named and versioned schema;
- log both prompt and schema hashes;
- add tests for the exact OpenRouter request body;
- explicitly handle refusal, truncation, empty content, invalid JSON, and schema failure;
- never silently downgrade to free-form output or plain JSON mode.

### P0 — Prompt and component ablations are not genuinely configuration-driven

The current simulator config contains component names, but the demo directly instantiates:

- `LLMController`
- `LLMSatisfactionUpdater`
- `DifficultySelectionPolicy`
- `DifficultyRealizationPolicy`
- `LLMUserRealizer`

The concrete classes also use hard-coded default prompt paths.

V2 must add:

- typed prompt configuration;
- a component registry/factory;
- construction of all components through the factory;
- prompt-path selection through YAML;
- optional CLI prompt overrides;
- startup validation for unknown component or schema names.

### P1 — Interactive audit output is too raw

The current console renderer prints a list of raw event dictionaries. V2 should render one consolidated turn panel that clearly shows:

- frontier before and after;
- candidates;
- raw and normalized controller decisions;
- newly exposed nodes;
- END reachability;
- satisfaction before and after;
- normalization violations;
- termination decision;
- automatic backbone exposure;
- unresolved queue;
- selected nodes;
- realization mode;
- generated user message;
- model latency, token use, and retry count.

### P1 — Test coverage must be expanded

Add tests for:

- strict structured-output request payloads;
- schema registry and schema hashing;
- prompt configuration and prompt overrides;
- component factory behavior;
- refusal and truncation handling;
- exact semantic output validation;
- no latent-intent leakage;
- satisfaction not being inferred from user statements alone;
- demo operation with a mock LLM and no API key.

---

## 3. Required architecture changes

Use or adapt the following structure:

```text
user_simulator/
├── configs/
│   ├── simulator.yaml
│   ├── models/
│   │   └── deepseek_v4_pro.yaml
│   └── prompts/
│       ├── controller_v2.yaml
│       ├── satisfaction_v2.yaml
│       ├── user_clear_v2.yaml
│       └── user_abstract_v2.yaml
├── src/user_simulator/
│   ├── factory.py
│   ├── config.py
│   ├── llm/
│   │   ├── base.py
│   │   ├── openrouter_client.py
│   │   ├── prompt.py
│   │   ├── schema_registry.py
│   │   ├── schema_utils.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── controller.py
│   │       ├── satisfaction.py
│   │       └── user_generation.py
│   ├── controller/
│   ├── satisfaction/
│   ├── policy/
│   ├── realizer/
│   ├── engine/
│   ├── audit/
│   └── cli/
└── tests/
    ├── unit/
    ├── integration/
    └── live/
```

Exact filenames may differ slightly, but the responsibilities must remain separate.

### 3.1 Separate wire schemas from normalized domain results

LLM wire schemas should live under:

```text
src/user_simulator/llm/schemas/
```

Domain-level results such as `NormalizedControllerResult`, `SelectionResult`, and `TurnResult` may remain under `domain/`.

This separation prevents an ablation or provider-specific schema change from unintentionally changing engine-domain types.

---

## 4. Typed configuration and genuine component replacement

### 4.1 Add prompt settings

Extend `SimulatorConfig` with:

```python
class PromptSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller: str
    satisfaction: str
    realizer_clear: str
    realizer_abstract: str
```

Example:

```yaml
prompts:
  controller: configs/prompts/controller_v2.yaml
  satisfaction: configs/prompts/satisfaction_v2.yaml
  realizer_clear: configs/prompts/user_clear_v2.yaml
  realizer_abstract: configs/prompts/user_abstract_v2.yaml
```

### 4.2 Replace untyped component dictionaries

Use a typed component config:

```python
class ComponentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller: str = "llm_controller"
    satisfaction_updater: str = "llm_satisfaction_updater"
    selection_policy: str = "difficulty_selection_v1"
    realization_policy: str = "difficulty_realization_v1"
    user_realizer: str = "llm_user_realizer"
```

### 4.3 Add a component factory

Create a registry/factory such as:

```python
CONTROLLER_REGISTRY = {
    "llm_controller": LLMController,
    "mock_controller": MockController,
}

SATISFACTION_REGISTRY = {
    "llm_satisfaction_updater": LLMSatisfactionUpdater,
    "mock_satisfaction_updater": MockSatisfactionUpdater,
}
```

The demo and future evaluation scripts must call a single builder:

```python
components = build_simulator_components(
    config=resolved_config,
    environment=environment,
    audit=audit,
)
```

Do not instantiate concrete LLM components directly inside `cli/demo.py`.

Unknown names must fail at startup with a clear configuration error.

### 4.4 Prompt objects must be injected

Prefer:

```python
controller = LLMController(
    client=client,
    generation=generation,
    prompt=PromptTemplate.load(config.prompts.controller),
)
```

over hard-coded prompt paths inside component constructors.

A constructor fallback may remain for backward compatibility, but normal runtime construction must use resolved configuration.

### 4.5 CLI overrides

Add optional overrides:

```text
--controller-prompt
--satisfaction-prompt
--clear-prompt
--abstract-prompt
--controller-component
--satisfaction-component
--realizer-component
--mock
```

A custom simulator YAML remains the preferred way to run large ablations.

Configuration precedence remains:

```text
default YAML < custom YAML < environment variables < CLI arguments
```

---

## 5. Replace the prompts with V2 prompts

Each prompt YAML must contain:

```yaml
name: ...
version: 2
schema_name: ...
schema_version: 2
system: |
  ...
user_template: |
  ...
```

`PromptTemplate` must parse and expose all four metadata fields.

At startup, verify that `schema_name` exists in the schema registry and that the configured `schema_version` matches the registered schema version.

---

### 5.1 `configs/prompts/controller_v2.yaml`

```yaml
name: controller_v2
version: 2
schema_name: controller_result
schema_version: 2

system: |
  You are the exposure controller of a reason-DAG user simulator.

  Your task is to decide how far the currently exposed frontier can advance
  after the latest assistant message. You are not evaluating user
  satisfaction here, and you must not classify the assistant message as an
  answer, clarification, or mixed action.

  DEFINITIONS

  - The current frontier is the highest-index intent node already exposed.
  - Candidate nodes are direct outgoing intent targets from the current
    frontier, sorted by node index.
  - A candidate is exposable only when the latest assistant message, interpreted
    in the full conversation context, creates a natural reason for the user to
    reveal or discuss that candidate intent next.
  - Exposure is not the same as satisfaction. A targeted clarification question
    may expose an intent even when it provides no useful solution.
  - A useful solution may expose a candidate when it meaningfully addresses the
    chain up to that candidate to at least a minimal useful degree. Complete
    satisfaction is not required.
  - Semantic relatedness alone is not enough. Do not expose a node merely
    because its hidden intent is related to the topic, would be useful to ask
    about, or appears in the candidate metadata.
  - Do not assume the assistant knows the hidden node content. Judge only what
    the visible conversation and latest assistant message support.

  EXPOSURE TEST

  For each candidate in order, ask:

  1. Does the latest assistant message directly address information that makes
     this intent a natural next concern?
  OR
  2. Does it ask a targeted, context-grounded question whose natural answer
     would reveal this intent?

  If neither is true, mark the candidate as not exposable.

  PREFIX CLOSURE

  - Evaluate candidates from lowest to highest node index.
  - Once one candidate is not exposable, every later candidate must also be
    marked not exposable.
  - Still return one decision for every candidate.
  - For candidates after the first failure, use a concise reason indicating that
    they are blocked by prefix closure.

  END

  - END is a system-level structural permission, not immediate termination.
  - END can be reachable only when an outgoing END edge exists and every ordered
    intent candidate is exposable.
  - Also require the latest assistant message to make ending the task
    conversationally plausible by meaningfully addressing the remaining need.
  - A message that only asks for more information should not make END reachable,
    even though it may expose intent nodes.
  - Do not terminate the episode yourself.

  OUTPUT

  - Return exactly one decision for every supplied candidate, in the same order.
  - Use only supplied node IDs.
  - Keep each reason short and evidence-based.
  - Return only the externally enforced structured result.
  - Do not include private reasoning, Markdown, or extra fields.

user_template: |
  Evaluate this controller state.

  {context}

  Required candidate order:
  {candidate_order}

  {correction}
```

#### Controller rendering changes

Pass `candidate_order` separately instead of requiring the model to infer it from a large JSON blob.

The context should clearly separate:

```text
VISIBLE CONVERSATION
LATEST ASSISTANT MESSAGE
CURRENT FRONTIER
EXPOSED NODE IDS
CURRENT SATISFACTION STATES
ORDERED CANDIDATE NODE DETAILS
HAS OUTGOING END EDGE
```

Do not send unrelated retained metadata to the controller.

---

### 5.2 `configs/prompts/satisfaction_v2.yaml`

```yaml
name: satisfaction_v2
version: 2
schema_name: satisfaction_update_result
schema_version: 2

system: |
  You are the satisfaction updater of a reason-DAG user simulator.

  Evaluate how well the assistant has addressed each currently exposed user
  intent across the complete conversation. Return one status for every exposed
  intent node.

  CRITICAL EVIDENCE RULE

  Satisfaction measures assistance provided by the assistant. The user's own
  message stating, explaining, or repeating an intent is not evidence that the
  intent has been satisfied.

  Use assistant contributions across the full history as evidence. The latest
  assistant message may add to earlier useful assistance. Do not judge only the
  latest message.

  LABEL DEFINITIONS

  unsatisfied:
  - The assistant has provided no useful help for the node's core intent;
  - or it only restates the request;
  - or it only asks for clarification without providing relevant help;
  - or the response is irrelevant, contradictory, or based on an unsupported
    assumption;
  - or any useful content is too weak to address a meaningful part of the need.

  partially_satisfied:
  - The assistant provides relevant and useful help for a meaningful part of
    the node;
  - but an important requested component, constraint, preference, decision, or
    actionable detail remains unresolved;
  - or it addresses only part of a multi-part intent;
  - or it gives a promising but overly general answer that still requires
    substantial follow-up.

  satisfied:
  - The assistant sufficiently addresses the node's core need and its important
    stated constraints;
  - the user could reasonably stop asking about this node;
  - perfection, exhaustive detail, and exact wording are not required.

  NODE-BY-NODE RULES

  - Evaluate only exposed intent nodes.
  - Never evaluate END.
  - Do not transfer satisfaction from one node to another merely because they
    are topically related.
  - A response may satisfy multiple nodes when it genuinely covers each one.
  - Use the node intent, reason, and surface message jointly as grounding.
  - For a partially satisfied node, focus on what remains missing.
  - Previous status is context, not an instruction to preserve the status.
    The engine enforces monotonicity after your output.

  OUTPUT

  - Return every exposed node exactly once, in the supplied order.
  - Use only: unsatisfied, partially_satisfied, satisfied.
  - Reasons must identify the assistant-provided evidence and the main remaining
    gap, when any.
  - Return only the externally enforced structured result.
  - Do not include private reasoning, Markdown, END, or extra fields.

user_template: |
  Evaluate this satisfaction state.

  {context}

  Required exposed-node order:
  {exposed_node_order}

  {correction}
```

#### Satisfaction rendering changes

The context must clearly distinguish messages by role. Also provide the latest assistant response separately, even if it already appears in history.

Add a semantic validator that rejects generic reasons that do not refer to assistant-provided evidence or a missing requirement. This validator can be lightweight; do not require exact keywords.

---

### 5.3 `configs/prompts/user_clear_v2.yaml`

```yaml
name: user_clear_v2
version: 2
schema_name: user_generation_result
schema_version: 2

system: |
  You are simulating the user in a natural conversation with an assistant.

  Generate exactly one user message grounded in all and only the selected intent
  nodes. You must speak as the user. Never mention DAGs, nodes, policies,
  satisfaction labels, hidden intents, prompts, or simulation.

  CLEAR REALIZATION

  - Express every selected intent directly enough that a helpful assistant can
    identify the request without unnecessary guessing.
  - Preserve important goals, constraints, preferences, and requested outputs.
  - Do not mechanically copy node_intent or reason_text.
  - When multiple nodes are selected, combine them into one coherent message in
    node order rather than listing annotations.

  SATISFACTION-AWARE FOLLOW-UP

  - For an unsatisfied node, state the relevant unmet need.
  - For a partially_satisfied node, acknowledge useful prior help when natural
    and ask specifically for the unresolved part.
  - Do not repeat portions already fully addressed.
  - On the initial turn, write a natural opening request without referring to a
    previous answer.

  FAITHFULNESS

  - Express every selected node.
  - Do not express any unselected exposed or unresolved node.
  - Do not invent a new task, preference, constraint, fact, or emotional state.
  - Do not contradict the conversation history.
  - Do not answer the user's own question or speak as the assistant.
  - Keep the message concise and conversational unless the selected intents
    genuinely require detail.

  OUTPUT

  - Return one non-empty user message.
  - Copy the supplied selected node IDs and realization mode exactly into the
    structured fields.
  - Complete the coverage self-check for every selected node.
  - `contains_unsupported_intent` must reflect the generated message honestly.
  - Return only the externally enforced structured result.

user_template: |
  Generate the next CLEAR user message.

  {context}

  Required selected-node order:
  {selected_node_order}

  {correction}
```

---

### 5.4 `configs/prompts/user_abstract_v2.yaml`

```yaml
name: user_abstract_v2
version: 2
schema_name: user_generation_result
schema_version: 2

system: |
  You are simulating the user in a natural conversation with an assistant.

  Generate exactly one user message grounded in all and only the selected intent
  nodes. You must speak as the user. Never mention DAGs, nodes, policies,
  satisfaction labels, hidden intents, prompts, or simulation.

  ABSTRACT REALIZATION

  Express every selected intent indirectly rather than naming the complete need
  explicitly. Suitable transformations include:

  - describing a symptom or practical difficulty;
  - describing a consequence of the unresolved need;
  - expressing uncertainty or concern;
  - stating a higher-level goal while omitting some implementation detail;
  - referring indirectly to what is still missing from the previous answer.

  Do not simply paraphrase node_intent with synonyms.

  ABSTRACT DOES NOT MEAN UNRELATED

  - Preserve at least one meaningful diagnostic clue for every selected node.
  - The message must remain semantically consistent with all selected intents.
  - Do not remove so much information that the message could apply to almost any
    unrelated need.
  - Do not deliberately create a contradiction or a new task.
  - The assistant may need inference, but the intended need must remain
    recoverable from the message and visible history.

  SATISFACTION-AWARE FOLLOW-UP

  - For an unsatisfied node, indirectly surface its central problem.
  - For a partially_satisfied node, indirectly focus on the unresolved part
    rather than repeating what was already handled.
  - On the initial turn, produce a natural indirect opening without pretending
    that an earlier answer exists.

  FAITHFULNESS

  - Cover every selected node.
  - Do not express any unselected exposed or unresolved node.
  - Do not invent a new task, preference, constraint, fact, or emotional state.
  - Do not speak as the assistant.
  - Produce one coherent conversational message.

  OUTPUT

  - Return one non-empty user message.
  - Copy the supplied selected node IDs and realization mode exactly into the
    structured fields.
  - Complete the coverage self-check for every selected node.
  - `contains_unsupported_intent` must reflect the generated message honestly.
  - Return only the externally enforced structured result.

user_template: |
  Generate the next ABSTRACT user message.

  {context}

  Required selected-node order:
  {selected_node_order}

  {correction}
```

---

## 6. Structured-output protocol V2

### 6.1 Add a schema registry

Create a versioned registry:

```python
@dataclass(frozen=True)
class SchemaSpec:
    name: str
    version: int
    model: type[BaseModel]


SCHEMA_REGISTRY: dict[tuple[str, int], SchemaSpec] = {
    ("controller_result", 2): SchemaSpec(
        name="controller_result",
        version=2,
        model=ControllerResultV2,
    ),
    ("satisfaction_update_result", 2): SchemaSpec(
        name="satisfaction_update_result",
        version=2,
        model=SatisfactionUpdateResultV2,
    ),
    ("user_generation_result", 2): SchemaSpec(
        name="user_generation_result",
        version=2,
        model=UserGenerationResultV2,
    ),
}
```

Prompt configuration is the source of the schema name and version. Component code must not hard-code strings such as:

```python
schema_name="controller_result"
```

without resolving the registered version.

### 6.2 Define strict wire schemas

All wire models must use:

```python
model_config = ConfigDict(extra="forbid", strict=True)
```

Use constraints where useful:

```python
reason: str = Field(min_length=1, max_length=300)
summary: str = Field(min_length=1, max_length=500)
user_message: str = Field(min_length=1, max_length=4000)
```

#### Controller V2 schema

```python
class CandidateExposureDecisionV2(StrictWireModel):
    node_id: str
    exposable: bool
    reason: str = Field(min_length=1, max_length=300)


class ControllerResultV2(StrictWireModel):
    decisions: list[CandidateExposureDecisionV2]
    end_reachable: bool
    summary: str = Field(min_length=1, max_length=500)
```

Semantic validation must require:

- candidate IDs exactly match the required candidates;
- output order exactly matches candidate order;
- no duplicate IDs;
- no unknown IDs;
- code-level prefix normalization;
- END forced false when no END edge exists;
- END forced false when the candidate prefix is incomplete.

### 6.3 Replace dynamic `coverage_check` dictionaries

Do not use:

```python
coverage_check: dict[str, bool]
```

for the wire protocol. Dynamic property names weaken the explicit schema and are harder to validate consistently across providers.

Use:

```python
class NodeCoverageDecisionV2(StrictWireModel):
    node_id: str
    covered: bool


class UserGenerationResultV2(StrictWireModel):
    user_message: str = Field(min_length=1, max_length=4000)
    selected_node_ids: list[str]
    realization_mode: RealizationMode
    coverage: list[NodeCoverageDecisionV2]
    contains_unsupported_intent: bool
    summary: str = Field(min_length=1, max_length=500)
```

Semantic validation must require:

- `selected_node_ids` exactly equal the policy-selected IDs in order;
- `coverage` contains the same IDs exactly once and in order;
- every `covered` is true;
- realization mode equals the system-selected mode;
- `contains_unsupported_intent` is false;
- user message is non-empty after stripping whitespace.

### 6.4 Generate canonical schemas and hashes

Before requests:

1. call `model.model_json_schema()`;
2. canonicalize it using sorted JSON keys;
3. recursively ensure object schemas set `additionalProperties: false` where appropriate;
4. compute:

```python
schema_hash = sha256(canonical_schema_json.encode()).hexdigest()
```

Do not mutate schemas to permit extra output.

Audit every request with:

```text
schema_name
schema_version
schema_hash
structured_output_type
strict
```

### 6.5 Use model configuration at runtime

Replace an untyped dictionary with:

```python
class StructuredOutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"] = "json_schema"
    strict: bool = True
    require_parameters: bool = True
```

The OpenRouter client must read these values from the model profile.

Startup must fail if:

```text
type != json_schema
strict is not true
require_parameters is not true
```

for the default V2 LLM components.

The request must still include OpenRouter provider routing:

```python
extra_body={
    "provider": {
        **profile.routing,
        "require_parameters": True,
    },
    ...
}
```

### 6.6 Explicit response-state handling

Before parsing content, inspect:

- missing choices;
- message refusal, when exposed by the SDK;
- empty content;
- `finish_reason`;
- truncation such as `finish_reason == "length"`;
- invalid JSON;
- Pydantic validation failure;
- provider/API errors.

A refusal or truncation is not a valid structured result and must be audited and retried according to policy.

Prefer:

```python
result = response_model.model_validate_json(content)
```

over manually calling `json.loads` followed by `model_validate`, unless SDK response handling requires otherwise.

### 6.7 Never downgrade silently

On failure, do not:

- switch to `json_object`;
- remove `strict=True`;
- parse Markdown fences;
- extract braces with regex;
- fill missing model fields with guessed values;
- accept unknown enum labels;
- fall back to free-form text.

After configured retries, raise a component-specific error and emit `episode_failed`.

Deterministic engine normalizations remain allowed:

- prefix closure;
- END invariants;
- satisfaction monotonicity.

Every normalization must be visible in audit logs.

---

## 7. Prompt and schema metadata

`PromptTemplate.render()` must return metadata containing:

```text
prompt_name
prompt_version
prompt_path
prompt_hash
schema_name
schema_version
schema_hash
```

`prompt_hash` should be computed from the complete rendered system and user messages, not only the YAML source.

The config snapshot must include resolved prompt paths and model profiles.

Changing prompt text without increasing `version` should produce a startup warning. A simple manifest can store known prompt hashes, or tests can snapshot expected hashes.

---

## 8. Controller and satisfaction context formatting

Do not place all inputs under an unlabeled `Episode state: {context}` block only.

Use deterministic, human-readable rendering. JSON is acceptable, but separate fields visibly.

### Controller context should include only:

- visible conversation history;
- latest assistant response;
- current frontier;
- exposed node IDs;
- current satisfaction states;
- ordered outgoing candidate node details;
- outgoing END-edge existence.

### Satisfaction context should include:

- visible conversation history with explicit roles;
- latest assistant response;
- exposed node details in required order;
- previous satisfaction states.

### Realizer context should include:

- selected node details;
- selected-node order;
- selected-node satisfaction states;
- unselected unresolved node IDs and details only when needed to enforce non-leakage;
- complete visible history;
- latest assistant response, or an explicit initial-turn marker;
- task summary and expectation when useful.

Do not send irrelevant metadata or survey fields by default.

---

## 9. Semantic validation and correction feedback

Keep structured schema validation and semantic validation separate.

When semantic validation fails, correction feedback must be specific.

Examples:

```text
Candidate order mismatch. Required: [N3, N4, N5].
Returned: [N4, N3, N5].
```

```text
Coverage IDs must exactly match selected IDs [N2, N3].
```

```text
The message self-check reports an unsupported intent. Regenerate using only
N2 and N3.
```

Audit:

```text
semantic_attempt
semantic_error
corrective_message
```

Do not expose private chain-of-thought. Reasons and summaries are concise audit justifications only.

---

## 10. Interactive demo V2

### 10.1 Add offline mock mode

The demo must support:

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --mock
```

Mock mode:

- requires no API key;
- exercises the full state machine;
- is deterministic;
- is suitable for CI and manual verification.

### 10.2 Prevent accidental latent-information leakage

The current debug demo may show the task summary before conversation starts. For a faithful assistant-facing view, hide latent task summary and expectation by default.

Add:

```text
--show-latent-summary
```

Only show latent information when explicitly requested for auditing.

The visible conversation area should contain only user and assistant messages. Audit panels must be visually labeled and kept separate.

### 10.3 Consolidated per-turn audit panel

Replace the raw event-list-only display with a readable panel or table.

Required sections:

```text
TURN
- turn index
- sample ID
- difficulty

CONTROLLER
- frontier before
- outgoing candidates
- raw decisions
- normalized decisions
- prefix violations
- newly exposed nodes
- END reachable
- frontier after

SATISFACTION
- status before
- proposed status
- applied status
- monotonicity violations

SYSTEM
- all exposed satisfied
- termination result
- auto-exposed backbone node

USER POLICY
- unresolved queue
- selected nodes
- selection rule
- realization mode
- generated message

LLM
- component/model
- schema name/version/hash
- prompt name/version/hash
- latency
- input/output tokens
- retry counts
- validation status
```

`/audit` should display this consolidated view for the latest turn.

Add optional commands:

```text
/raw-audit
/prompts
/schemas
```

- `/raw-audit`: raw latest-turn events;
- `/prompts`: prompt names, versions, and hashes, not hidden API keys;
- `/schemas`: schema names, versions, and hashes.

---

## 11. Audit log revisions

Every LLM event must record:

```text
component
model_id
model_profile
prompt_name
prompt_version
prompt_hash
schema_name
schema_version
schema_hash
request_id
provider
finish_reason
latency_seconds
input_tokens
output_tokens
transport_retry_count
semantic_retry_count
structured_validation_status
semantic_validation_status
normalization_violations
```

For `audit.level == summary`:

- do not store full rendered prompts;
- do not store complete latent node text;
- keep hashes and concise decisions.

For `audit.level == full`:

- store rendered prompts;
- store raw structured model responses;
- store complete component input context.

Never log the API key or authorization headers.

---

## 12. Required tests

The implementation is not complete until all tests below exist and pass.

### 12.1 Structured request tests

Using a fake or mocked OpenAI client, assert that every request contains:

```python
response_format["type"] == "json_schema"
response_format["json_schema"]["strict"] is True
response_format["json_schema"]["name"] == expected_schema_name
extra_body["provider"]["require_parameters"] is True
stream is False
```

Also assert that the schema contains:

```text
type: object
additionalProperties: false
required fields
enum restrictions
```

### 12.2 Structured response failure tests

Test:

- empty choices;
- empty content;
- invalid JSON;
- extra properties;
- invalid enum;
- missing required fields;
- refusal;
- `finish_reason == "length"`;
- retry exhaustion;
- no fallback to plain JSON or free text.

### 12.3 Prompt contract tests

Test prompt loading and rendering:

- all V2 prompts have name/version/schema name/schema version;
- placeholders are complete;
- unknown placeholders fail at startup;
- schema registry lookup succeeds;
- prompt hash changes when rendered content changes;
- prompt paths can be overridden through config and CLI.

### 12.4 Component-factory tests

Test:

- each configured default component is built;
- a mock component can replace each LLM component;
- unknown component names fail clearly;
- demo construction uses the factory rather than direct instantiation.

### 12.5 Core semantic regression tests

Add curated mock or deterministic cases covering:

1. **Clarification exposure without satisfaction**

   - a targeted question exposes downstream nodes;
   - satisfaction remains unchanged when no assistant help is provided.
2. **User disclosure is not satisfaction**

   - the user clearly states an intent;
   - the assistant only restates or asks a question;
   - node remains `unsatisfied`.
3. **Semantic relation is not exposure**

   - a candidate is topically related;
   - the assistant response does not address it or ask a targeted question;
   - candidate is not exposed.
4. **Partial satisfaction**

   - assistant covers one meaningful part but misses a key constraint;
   - status is `partially_satisfied`.
5. **Clear faithfulness**

   - generated message covers all selected nodes;
   - no unselected intent is introduced.
6. **Abstract faithfulness**

   - generated message is indirect;
   - each selected intent remains recoverable;
   - no new intent is introduced.
7. **Partially satisfied follow-up**

   - generated user message asks only for the unresolved portion.
8. **Initial turn**

   - no reference to an earlier assistant answer.

### 12.6 Engine tests

Retain and extend tests for:

- prefix closure;
- END normalization;
- monotonic satisfaction;
- termination;
- natural backbone exposure;
- random-prefix reproducibility;
- exact queue ordering;
- no END in exposed nodes or satisfaction map;
- terminal deadlock error.

### 12.7 Integration tests

Add a complete deterministic mock episode for each difficulty:

```text
easy
medium
hard
```

Verify the exact selected node sequence and realization sequence for a fixed seed.

### 12.8 Live tests

Live OpenRouter tests must be opt-in and skipped without `OPENROUTER_API_KEY`.

Test all three output schemas using the default:

```text
deepseek/deepseek-v4-pro
```

The live test should verify:

- strict structured output succeeds;
- schema validation succeeds;
- `require_parameters` is enabled;
- metadata includes token usage or records its absence safely;
- no free-form fallback occurs.

Keep live calls minimal.

---

## 13. Prompt smoke-test utility

Add:

```bash
python -m user_simulator.cli.prompt_smoke_test \
  --component all \
  --cases tests/fixtures/prompt_cases.jsonl \
  --output-dir runs/prompt_smoke
```

Each case should contain:

```json
{
  "case_id": "...",
  "component": "controller",
  "input": {},
  "expected_invariants": {}
}
```

The utility should report:

- schema-valid rate;
- semantic-valid rate;
- retry rate;
- controller prefix-violation rate;
- satisfaction label distribution;
- user-generation coverage-check pass rate;
- unsupported-intent self-check rate.

This is a diagnostic tool, not the final human evaluation.

---

## 14. Updated default configuration

Update `configs/simulator.yaml`:

```yaml
dataset_path: dataset/DAG.jsonl

components:
  controller: llm_controller
  satisfaction_updater: llm_satisfaction_updater
  selection_policy: difficulty_selection_v1
  realization_policy: difficulty_realization_v1
  user_realizer: llm_user_realizer

prompts:
  controller: configs/prompts/controller_v2.yaml
  satisfaction: configs/prompts/satisfaction_v2.yaml
  realizer_clear: configs/prompts/user_clear_v2.yaml
  realizer_abstract: configs/prompts/user_abstract_v2.yaml

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

Update the model profile to use a typed structured-output section:

```yaml
structured_output:
  type: json_schema
  strict: true
  require_parameters: true
```

Keep:

```yaml
model_id: deepseek/deepseek-v4-pro
```

as the default.

---

## 15. Migration requirements

Maintain backward compatibility where reasonable:

- existing `controller.yaml`, `satisfaction.yaml`, `user_clear.yaml`, and
  `user_abstract.yaml` may remain as V1 prompt files;
- V2 default config must point to V2 prompt files;
- old audit logs do not need migration;
- dataset format must not change;
- episode-policy semantics must not change;
- public base interfaces should remain stable unless the factory refactor
  requires a documented change.

Do not delete working mock components or existing tests.

---

## 16. Verification commands

The following commands must work from `user_simulator/`.

### Static and unit tests

```bash
pytest -q
```

### Dataset validation

```bash
python -m user_simulator.cli.validate_dataset \
  --dataset dataset/DAG.jsonl \
  --strict
```

### Offline demo

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --mock \
  --audit-level full
```

### Live demo

```bash
python -m user_simulator.cli.demo \
  --random-sample \
  --difficulty medium \
  --seed 42 \
  --audit-level full
```

### Prompt smoke test

```bash
python -m user_simulator.cli.prompt_smoke_test \
  --component all \
  --cases tests/fixtures/prompt_cases.jsonl \
  --output-dir runs/prompt_smoke
```

---

## 17. V2 acceptance criteria

V2 is complete only when all of the following are true:

### Prompts

- all four V2 prompts are implemented;
- prompts have name, version, schema name, and schema version;
- prompts use operational criteria rather than short subjective instructions;
- satisfaction explicitly ignores user disclosure as satisfaction evidence;
- Clear and Abstract modes both enforce selected-node faithfulness;
- prompt paths are configuration-driven.

### Structured outputs

- all LLM calls use strict OpenRouter JSON Schema;
- strict mode is controlled and validated through model configuration;
- `require_parameters=true` is enforced;
- schemas are versioned and registered;
- schema hashes are audited;
- dynamic `coverage_check` keys are replaced by typed coverage items;
- refusal, truncation, invalid JSON, and schema failures are handled;
- no free-form fallback exists;
- exact request-body tests pass.

### Ablations

- components are constructed through a registry/factory;
- changing component names in YAML changes runtime implementations;
- changing prompt paths in YAML changes runtime prompts;
- CLI overrides work;
- mock mode requires no API key.

### Audit and demo

- latest-turn audit is readable without inspecting raw JSON;
- prompt/schema/model metadata is visible;
- task summary is hidden from the assistant-facing view by default;
- all state transitions remain auditable.

### Regression

- dataset remains unchanged;
- Easy/Medium/Hard policies remain unchanged;
- prefix closure remains enforced;
- satisfaction remains three-state and monotonic by default;
- END remains system-only;
- natural backbone exposure remains active;
- all existing and new tests pass.

---

## 18. Deliverables

Codex should modify the repository and provide:

1. updated source code;
2. four V2 prompt YAML files;
3. updated simulator and model configuration;
4. schema registry and strict wire schemas;
5. component factory;
6. improved audit renderer;
7. offline mock demo mode;
8. prompt smoke-test utility;
9. expanded tests;
10. a concise changelog describing:
    - files changed;
    - behavior changed;
    - backward-compatibility notes;
    - test commands and results.

Do not claim completion without running the non-live test suite and reporting its results.
