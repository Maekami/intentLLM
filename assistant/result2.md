# ExperimentalResults2.md

## Current Baseline Results

This file records the second complete baseline run for the three backbone models: 54 conditions, each evaluated on the same 292 test samples. No development-set evolution or distillation was run.

### Metric Notes

- **Exposure Turns ↓**: average number of interaction turns until all evaluated hidden DAG intent nodes have been exposed.
- **Satisfaction Turns ↓**: average number of interaction turns until all evaluated hidden DAG intent nodes have been satisfied.
- Terminal DAG nodes are excluded. If all evaluated nodes have not been exposed/satisfied within the 20-turn budget, the corresponding episode metric is assigned 21.
- **Avg. Tokens ↓**: average assistant output tokens per episode, shown in **thousands of tokens** and rounded to two decimals.
- Token cells use **excluding thinking (including thinking)**. The main value counts only the final assistant text delivered to the simulated user; parentheses add hidden reasoning associated with that response. Internal method calls, action JSON, and guard output are excluded from these token metrics.
- Tokenizers: Qwen native tokenizer; Luna official `o200k_base`; Gemini the user-approved local MedGemma-27B tokenizer.
- **AITR**: not yet available.
- Lower is better for the first three metrics.

---

# 1. Qwen3.6-27B

| Method           | Easy Exposure ↓ | Easy Satisfaction ↓ | Easy Tokens ↓ | Easy AITR | Medium Exposure ↓ | Medium Satisfaction ↓ | Medium Tokens ↓ | Medium AITR | Hard Exposure ↓ | Hard Satisfaction ↓ | Hard Tokens ↓ | Hard AITR |
| ---------------- | ---------------: | -------------------: | -------------: | --------: | -----------------: | ---------------------: | ---------------: | ----------: | ---------------: | -------------------: | -------------: | --------: |
| Prompted Base    |            3.36 |                4.87 |   3.91 (3.91) |         — |              3.65 |                  4.96 |     3.86 (3.86) |           — |            4.62 |                6.48 |   4.35 (4.35) |         — |
| +Thinking        |            3.29 |                4.53 |   2.38 (5.91) |         — |              3.66 |                  5.11 |     2.57 (6.26) |           — |            4.38 |                6.25 |   2.81 (6.42) |         — |
| ExpRAG           |            3.41 |                4.51 |   5.16 (5.16) |         — |              3.56 |                  4.93 |     5.28 (5.28) |           — |            4.04 |                5.93 |   5.93 (5.93) |         — |
| ReMem            |            5.11 |                6.73 |   3.03 (3.03) |         — |              5.67 |                  7.43 |     3.09 (3.09) |           — |            6.57 |                8.63 |   3.27 (3.27) |         — |
| Trace2Skill-like |            3.13 |                4.32 |   4.42 (4.42) |         — |              3.49 |                  4.71 |     4.76 (4.76) |           — |            3.82 |                5.49 |   5.17 (5.17) |         — |
| Probing          |            7.21 |                8.85 |   1.66 (1.66) |         — |              7.46 |                  9.24 |     1.60 (1.60) |           — |            9.07 |               11.37 |   1.57 (1.57) |         — |
| Ours             |               — |                   — |             — |         — |                 — |                     — |               — |           — |               — |                   — |             — |         — |

---

# 2. Gemini 3.6 Flash

| Method           | Easy Exposure ↓ | Easy Satisfaction ↓ | Easy Tokens ↓ | Easy AITR | Medium Exposure ↓ | Medium Satisfaction ↓ | Medium Tokens ↓ | Medium AITR | Hard Exposure ↓ | Hard Satisfaction ↓ | Hard Tokens ↓ | Hard AITR |
| ---------------- | ---------------: | -------------------: | -------------: | --------: | -----------------: | ---------------------: | ---------------: | ----------: | ---------------: | -------------------: | -------------: | --------: |
| Prompted Base    |            3.62 |                4.82 |   2.65 (2.65) |         — |              3.78 |                  5.14 |     2.81 (2.81) |           — |            4.62 |                6.75 |   3.30 (3.30) |         — |
| +Thinking        |            3.46 |                4.49 |   2.68 (6.31) |         — |              3.85 |                  5.15 |     2.98 (7.15) |           — |            4.50 |                6.30 |   3.11 (8.18) |         — |
| ExpRAG           |            3.27 |                4.20 |   3.58 (3.58) |         — |              3.52 |                  4.68 |     4.01 (4.01) |           — |            3.93 |                5.64 |   4.36 (4.36) |         — |
| ReMem            |            3.33 |                4.38 |   2.97 (2.97) |         — |              3.52 |                  4.88 |     3.04 (3.04) |           — |            3.94 |                5.51 |   3.24 (3.24) |         — |
| Trace2Skill-like |            3.57 |                4.86 |   3.12 (3.12) |         — |              3.76 |                  5.01 |     3.06 (3.06) |           — |            4.12 |                6.06 |   3.25 (3.25) |         — |
| Probing          |            5.28 |                6.71 |   1.86 (1.86) |         — |              5.83 |                  7.66 |     1.85 (1.85) |           — |            7.43 |                9.89 |   2.00 (2.00) |         — |
| Ours             |               — |                   — |             — |         — |                 — |                     — |               — |           — |               — |                   — |             — |         — |

---

# 3. GPT-5.6-Luna

| Method           | Easy Exposure ↓ | Easy Satisfaction ↓ | Easy Tokens ↓ | Easy AITR | Medium Exposure ↓ | Medium Satisfaction ↓ | Medium Tokens ↓ | Medium AITR | Hard Exposure ↓ | Hard Satisfaction ↓ | Hard Tokens ↓ | Hard AITR |
| ---------------- | ---------------: | -------------------: | -------------: | --------: | -----------------: | ---------------------: | ---------------: | ----------: | ---------------: | -------------------: | -------------: | --------: |
| Prompted Base    |            3.53 |                4.53 |   2.83 (2.83) |         — |              3.68 |                  4.90 |     2.81 (2.81) |           — |            4.39 |                6.16 |   2.80 (2.80) |         — |
| +Thinking        |            3.46 |                4.65 |   2.85 (5.65) |         — |              3.67 |                  5.13 |     2.97 (5.99) |           — |            4.42 |                6.22 |   3.00 (6.37) |         — |
| ExpRAG           |            3.33 |                4.40 |   3.22 (3.22) |         — |              3.66 |                  4.82 |     3.20 (3.20) |           — |            3.93 |                5.56 |   3.31 (3.31) |         — |
| ReMem            |            3.46 |                4.44 |   2.78 (2.78) |         — |              3.53 |                  4.80 |     2.77 (2.77) |           — |            4.16 |                5.61 |   2.59 (2.59) |         — |
| Trace2Skill-like |            3.33 |                4.47 |   3.22 (3.22) |         — |              3.84 |                  5.24 |     3.28 (3.28) |           — |            4.20 |                5.86 |   3.40 (3.40) |         — |
| Probing          |            4.47 |                5.61 |   1.75 (1.75) |         — |              4.77 |                  6.30 |     1.79 (1.79) |           — |            5.53 |                7.37 |   1.82 (1.82) |         — |
| Ours             |               — |                   — |             — |         — |                 — |                     — |               — |           — |               — |                   — |             — |         — |

---

# 4. Compact Machine-Readable View

The following representation may be easier for scripts or Codex to parse. Both token fields use thousands of tokens; `tokens` excludes thinking.

```yaml
metrics:
  exposure_turns: lower_is_better
  satisfaction_turns: lower_is_better
  avg_tokens: lower_is_better
  avg_tokens_including_thinking: lower_is_better
  aitr: pending

token_unit: thousand

models:
  Qwen3.6-27B:
    Prompted Base:
      easy:   {exposure: 3.36, satisfaction: 4.87, tokens: 3.91, tokens_including_thinking: 3.91}
      medium: {exposure: 3.65, satisfaction: 4.96, tokens: 3.86, tokens_including_thinking: 3.86}
      hard:   {exposure: 4.62, satisfaction: 6.48, tokens: 4.35, tokens_including_thinking: 4.35}
    +Thinking:
      easy:   {exposure: 3.29, satisfaction: 4.53, tokens: 2.38, tokens_including_thinking: 5.91}
      medium: {exposure: 3.66, satisfaction: 5.11, tokens: 2.57, tokens_including_thinking: 6.26}
      hard:   {exposure: 4.38, satisfaction: 6.25, tokens: 2.81, tokens_including_thinking: 6.42}
    ExpRAG:
      easy:   {exposure: 3.41, satisfaction: 4.51, tokens: 5.16, tokens_including_thinking: 5.16}
      medium: {exposure: 3.56, satisfaction: 4.93, tokens: 5.28, tokens_including_thinking: 5.28}
      hard:   {exposure: 4.04, satisfaction: 5.93, tokens: 5.93, tokens_including_thinking: 5.93}
    ReMem:
      easy:   {exposure: 5.11, satisfaction: 6.73, tokens: 3.03, tokens_including_thinking: 3.03}
      medium: {exposure: 5.67, satisfaction: 7.43, tokens: 3.09, tokens_including_thinking: 3.09}
      hard:   {exposure: 6.57, satisfaction: 8.63, tokens: 3.27, tokens_including_thinking: 3.27}
    Trace2Skill-like:
      easy:   {exposure: 3.13, satisfaction: 4.32, tokens: 4.42, tokens_including_thinking: 4.42}
      medium: {exposure: 3.49, satisfaction: 4.71, tokens: 4.76, tokens_including_thinking: 4.76}
      hard:   {exposure: 3.82, satisfaction: 5.49, tokens: 5.17, tokens_including_thinking: 5.17}
    Probing:
      easy:   {exposure: 7.21, satisfaction: 8.85, tokens: 1.66, tokens_including_thinking: 1.66}
      medium: {exposure: 7.46, satisfaction: 9.24, tokens: 1.60, tokens_including_thinking: 1.60}
      hard:   {exposure: 9.07, satisfaction: 11.37, tokens: 1.57, tokens_including_thinking: 1.57}

  Gemini-3.6-Flash:
    Prompted Base:
      easy:   {exposure: 3.62, satisfaction: 4.82, tokens: 2.65, tokens_including_thinking: 2.65}
      medium: {exposure: 3.78, satisfaction: 5.14, tokens: 2.81, tokens_including_thinking: 2.81}
      hard:   {exposure: 4.62, satisfaction: 6.75, tokens: 3.30, tokens_including_thinking: 3.30}
    +Thinking:
      easy:   {exposure: 3.46, satisfaction: 4.49, tokens: 2.68, tokens_including_thinking: 6.31}
      medium: {exposure: 3.85, satisfaction: 5.15, tokens: 2.98, tokens_including_thinking: 7.15}
      hard:   {exposure: 4.50, satisfaction: 6.30, tokens: 3.11, tokens_including_thinking: 8.18}
    ExpRAG:
      easy:   {exposure: 3.27, satisfaction: 4.20, tokens: 3.58, tokens_including_thinking: 3.58}
      medium: {exposure: 3.52, satisfaction: 4.68, tokens: 4.01, tokens_including_thinking: 4.01}
      hard:   {exposure: 3.93, satisfaction: 5.64, tokens: 4.36, tokens_including_thinking: 4.36}
    ReMem:
      easy:   {exposure: 3.33, satisfaction: 4.38, tokens: 2.97, tokens_including_thinking: 2.97}
      medium: {exposure: 3.52, satisfaction: 4.88, tokens: 3.04, tokens_including_thinking: 3.04}
      hard:   {exposure: 3.94, satisfaction: 5.51, tokens: 3.24, tokens_including_thinking: 3.24}
    Trace2Skill-like:
      easy:   {exposure: 3.57, satisfaction: 4.86, tokens: 3.12, tokens_including_thinking: 3.12}
      medium: {exposure: 3.76, satisfaction: 5.01, tokens: 3.06, tokens_including_thinking: 3.06}
      hard:   {exposure: 4.12, satisfaction: 6.06, tokens: 3.25, tokens_including_thinking: 3.25}
    Probing:
      easy:   {exposure: 5.28, satisfaction: 6.71, tokens: 1.86, tokens_including_thinking: 1.86}
      medium: {exposure: 5.83, satisfaction: 7.66, tokens: 1.85, tokens_including_thinking: 1.85}
      hard:   {exposure: 7.43, satisfaction: 9.89, tokens: 2.00, tokens_including_thinking: 2.00}

  GPT-5.6-Luna:
    Prompted Base:
      easy:   {exposure: 3.53, satisfaction: 4.53, tokens: 2.83, tokens_including_thinking: 2.83}
      medium: {exposure: 3.68, satisfaction: 4.90, tokens: 2.81, tokens_including_thinking: 2.81}
      hard:   {exposure: 4.39, satisfaction: 6.16, tokens: 2.80, tokens_including_thinking: 2.80}
    +Thinking:
      easy:   {exposure: 3.46, satisfaction: 4.65, tokens: 2.85, tokens_including_thinking: 5.65}
      medium: {exposure: 3.67, satisfaction: 5.13, tokens: 2.97, tokens_including_thinking: 5.99}
      hard:   {exposure: 4.42, satisfaction: 6.22, tokens: 3.00, tokens_including_thinking: 6.37}
    ExpRAG:
      easy:   {exposure: 3.33, satisfaction: 4.40, tokens: 3.22, tokens_including_thinking: 3.22}
      medium: {exposure: 3.66, satisfaction: 4.82, tokens: 3.20, tokens_including_thinking: 3.20}
      hard:   {exposure: 3.93, satisfaction: 5.56, tokens: 3.31, tokens_including_thinking: 3.31}
    ReMem:
      easy:   {exposure: 3.46, satisfaction: 4.44, tokens: 2.78, tokens_including_thinking: 2.78}
      medium: {exposure: 3.53, satisfaction: 4.80, tokens: 2.77, tokens_including_thinking: 2.77}
      hard:   {exposure: 4.16, satisfaction: 5.61, tokens: 2.59, tokens_including_thinking: 2.59}
    Trace2Skill-like:
      easy:   {exposure: 3.33, satisfaction: 4.47, tokens: 3.22, tokens_including_thinking: 3.22}
      medium: {exposure: 3.84, satisfaction: 5.24, tokens: 3.28, tokens_including_thinking: 3.28}
      hard:   {exposure: 4.20, satisfaction: 5.86, tokens: 3.40, tokens_including_thinking: 3.40}
    Probing:
      easy:   {exposure: 4.47, satisfaction: 5.61, tokens: 1.75, tokens_including_thinking: 1.75}
      medium: {exposure: 4.77, satisfaction: 6.30, tokens: 1.79, tokens_including_thinking: 1.79}
      hard:   {exposure: 5.53, satisfaction: 7.37, tokens: 1.82, tokens_including_thinking: 1.82}

```

---

# 5. Status

All 54 baseline conditions are complete and audited, with 292 evaluated episodes per condition (15,768 in total): 15,077 successful episodes, 691 turn-limit failures, and 0 final infrastructure failures. Turn-limit failures were retained without retry. `Ours` and `AITR` are currently pending.

User-authorized targeted recovery was applied to Probing format failures in Qwen easy/medium/hard and Gemini easy/hard. For Luna Probing medium, only `user97_task1_conversation1` was recovered and subsequently restarted twice; its final attempt used at most 10 schema-valid candidates per turn, executing the last candidate if all were guard-rejected. This exception was triggered only on turn 1, and the sample completed in 3 turns. Other outcomes, archived attempts, and recorded costs were retained.

**Cumulative known recorded API cost: $467.383656**, including failed attempts and recoveries. This is a lower bound, not a complete invoice total: historical Luna billing gaps, one Gemini Probing hard request with unknown cost, and potentially unreturned usage at the two Luna interruptions remain documented. Local Qwen API cost is recorded as zero; hardware cost is not imputed.

Source records: [full metrics, per-condition costs, and key assignments](../baseline_round2_20261004/results.csv), [completion and exception summary](../baseline_round2_20261004/completion_summary.json), and [54-condition audit](../baseline_round2_20261004/completion_audit.json). Experiments used the frozen settings; the user-confirmed external simulator-profile change is recorded in the audit.
