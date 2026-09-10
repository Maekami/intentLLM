
# ExperimentalResults.md

## Current Baseline Results

This file records the current experimental results for the three backbone models.

### Metric Notes

- **Exposure Turns ↓**: average number of interaction turns until all hidden DAG intent nodes have been exposed.
- **Satisfaction Turns ↓**: average number of interaction turns until all hidden DAG intent nodes have been satisfied.
- **Avg. Tokens ↓**: average assistant output tokens per episode.
- **AITR**: not yet available.
- For `+Thinking`, only the non-parenthesized token value is kept here, as requested.
- Lower is better for the first three metrics.

---

# 1. Qwen3.6-27B

| Method           | Easy Exposure ↓ | Easy Satisfaction ↓ | Easy Tokens ↓ | Easy AITR | Medium Exposure ↓ | Medium Satisfaction ↓ | Medium Tokens ↓ | Medium AITR | Hard Exposure ↓ | Hard Satisfaction ↓ | Hard Tokens ↓ | Hard AITR |
| ---------------- | ---------------: | -------------------: | -------------: | --------: | -----------------: | ---------------------: | ---------------: | ----------: | ---------------: | -------------------: | -------------: | --------: |
| Prompted Base    |             3.43 |                 4.51 |           3.61 |        — |               3.68 |                   4.94 |             3.81 |          — |             4.08 |                 5.48 |           4.17 |        — |
| +Thinking        |             3.39 |                 4.50 |           2.49 |        — |               3.75 |                   4.97 |             2.70 |          — |             4.39 |                 5.90 |           2.84 |        — |
| ExpRAG           |             3.39 |                 4.46 |           5.09 |        — |               3.57 |                   4.64 |             5.17 |          — |             4.02 |                 5.58 |           5.79 |        — |
| ReMem            |             5.03 |                 6.29 |           3.03 |        — |               5.32 |                   6.74 |             3.12 |          — |             6.36 |                 8.23 |           3.42 |        — |
| Trace2Skill-like |             3.31 |                 4.32 |           4.65 |        — |               3.36 |                   4.59 |             4.86 |          — |             3.61 |                 5.10 |           5.20 |        — |
| Probing          |             6.54 |                 8.07 |           1.91 |        — |               6.99 |                   8.78 |             1.72 |          — |             8.32 |                10.57 |           1.74 |        — |
| Ours             |               — |                   — |             — |        — |                 — |                     — |               — |          — |               — |                   — |             — |        — |

---

# 2. Gemini 3.6 Flash

| Method           | Easy Exposure ↓ | Easy Satisfaction ↓ | Easy Tokens ↓ | Easy AITR | Medium Exposure ↓ | Medium Satisfaction ↓ | Medium Tokens ↓ | Medium AITR | Hard Exposure ↓ | Hard Satisfaction ↓ | Hard Tokens ↓ | Hard AITR |
| ---------------- | ---------------: | -------------------: | -------------: | --------: | -----------------: | ---------------------: | ---------------: | ----------: | ---------------: | -------------------: | -------------: | --------: |
| Prompted Base    |             3.66 |                 4.70 |           2.66 |        — |               3.89 |                   5.12 |             2.82 |          — |             4.33 |                 5.86 |           3.14 |        — |
| +Thinking        |             3.65 |                 4.72 |           2.87 |        — |               3.91 |                   5.06 |             2.80 |          — |             4.35 |                 5.78 |           3.02 |        — |
| ExpRAG           |             3.34 |                 4.27 |           3.81 |        — |               3.61 |                   4.71 |             4.03 |          — |             4.04 |                 5.47 |           4.29 |        — |
| ReMem            |             3.47 |                 4.41 |           3.03 |        — |               3.67 |                   4.75 |             3.22 |          — |             4.12 |                 5.47 |           3.63 |        — |
| Trace2Skill-like |             3.39 |                 4.42 |           3.27 |        — |               3.51 |                   4.68 |             3.05 |          — |             3.93 |                 5.39 |           3.22 |        — |
| Probing          |             5.28 |                 6.52 |           1.83 |        — |               5.88 |                   7.28 |             1.93 |          — |             6.73 |                 8.66 |           2.16 |        — |
| Ours             |               — |                   — |             — |        — |                 — |                     — |               — |          — |               — |                   — |             — |        — |

---

# 3. GPT-5.6-Luna

| Method           | Easy Exposure ↓ | Easy Satisfaction ↓ | Easy Tokens ↓ | Easy AITR | Medium Exposure ↓ | Medium Satisfaction ↓ | Medium Tokens ↓ | Medium AITR | Hard Exposure ↓ | Hard Satisfaction ↓ | Hard Tokens ↓ | Hard AITR |
| ---------------- | ---------------: | -------------------: | -------------: | --------: | -----------------: | ---------------------: | ---------------: | ----------: | ---------------: | -------------------: | -------------: | --------: |
| Prompted Base    |             3.34 |                 4.43 |           2.91 |        — |               3.59 |                   4.71 |             2.82 |          — |             3.69 |                 5.09 |           2.71 |        — |
| +Thinking        |             3.42 |                 4.47 |           2.98 |        — |               3.49 |                   4.60 |             3.04 |          — |             3.75 |                 5.22 |           2.92 |        — |
| ExpRAG           |             3.30 |                 4.30 |           3.28 |        — |               3.55 |                   4.80 |             3.42 |          — |             3.98 |                 5.45 |           3.45 |        — |
| ReMem            |             3.47 |                 4.46 |           2.77 |        — |               3.67 |                   4.73 |             2.74 |          — |             3.95 |                 5.43 |           2.84 |        — |
| Trace2Skill-like |             3.57 |                 4.57 |           3.40 |        — |               3.60 |                   4.87 |             3.43 |          — |             3.88 |                 5.30 |           3.29 |        — |
| Probing          |             4.47 |                 5.55 |           1.87 |        — |               4.86 |                   6.07 |             1.88 |          — |             5.28 |                 6.89 |           1.85 |        — |
| Ours             |               — |                   — |             — |        — |                 — |                     — |               — |          — |               — |                   — |             — |        — |

---

# 4. Compact Machine-Readable View

The following representation may be easier for scripts or Codex to parse.

```yaml
metrics:
  exposure_turns: lower_is_better
  satisfaction_turns: lower_is_better
  avg_tokens: lower_is_better
  aitr: pending

models:
  Qwen3.6-27B:
    Prompted Base:
      easy:   {exposure: 3.43, satisfaction: 4.51, tokens: 3.61}
      medium: {exposure: 3.68, satisfaction: 4.94, tokens: 3.81}
      hard:   {exposure: 4.08, satisfaction: 5.48, tokens: 4.17}
    +Thinking:
      easy:   {exposure: 3.39, satisfaction: 4.50, tokens: 2.49}
      medium: {exposure: 3.75, satisfaction: 4.97, tokens: 2.70}
      hard:   {exposure: 4.39, satisfaction: 5.90, tokens: 2.84}
    ExpRAG:
      easy:   {exposure: 3.39, satisfaction: 4.46, tokens: 5.09}
      medium: {exposure: 3.57, satisfaction: 4.64, tokens: 5.17}
      hard:   {exposure: 4.02, satisfaction: 5.58, tokens: 5.79}
    ReMem:
      easy:   {exposure: 5.03, satisfaction: 6.29, tokens: 3.03}
      medium: {exposure: 5.32, satisfaction: 6.74, tokens: 3.12}
      hard:   {exposure: 6.36, satisfaction: 8.23, tokens: 3.42}
    Trace2Skill-like:
      easy:   {exposure: 3.31, satisfaction: 4.32, tokens: 4.65}
      medium: {exposure: 3.36, satisfaction: 4.59, tokens: 4.86}
      hard:   {exposure: 3.61, satisfaction: 5.10, tokens: 5.20}
    Probing:
      easy:   {exposure: 6.54, satisfaction: 8.07, tokens: 1.91}
      medium: {exposure: 6.99, satisfaction: 8.78, tokens: 1.72}
      hard:   {exposure: 8.32, satisfaction: 10.57, tokens: 1.74}

  Gemini-3.6-Flash:
    Prompted Base:
      easy:   {exposure: 3.66, satisfaction: 4.70, tokens: 2.66}
      medium: {exposure: 3.89, satisfaction: 5.12, tokens: 2.82}
      hard:   {exposure: 4.33, satisfaction: 5.86, tokens: 3.14}
    +Thinking:
      easy:   {exposure: 3.65, satisfaction: 4.72, tokens: 2.87}
      medium: {exposure: 3.91, satisfaction: 5.06, tokens: 2.80}
      hard:   {exposure: 4.35, satisfaction: 5.78, tokens: 3.02}
    ExpRAG:
      easy:   {exposure: 3.34, satisfaction: 4.27, tokens: 3.81}
      medium: {exposure: 3.61, satisfaction: 4.71, tokens: 4.03}
      hard:   {exposure: 4.04, satisfaction: 5.47, tokens: 4.29}
    ReMem:
      easy:   {exposure: 3.47, satisfaction: 4.41, tokens: 3.03}
      medium: {exposure: 3.67, satisfaction: 4.75, tokens: 3.22}
      hard:   {exposure: 4.12, satisfaction: 5.47, tokens: 3.63}
    Trace2Skill-like:
      easy:   {exposure: 3.39, satisfaction: 4.42, tokens: 3.27}
      medium: {exposure: 3.51, satisfaction: 4.68, tokens: 3.05}
      hard:   {exposure: 3.93, satisfaction: 5.39, tokens: 3.22}
    Probing:
      easy:   {exposure: 5.28, satisfaction: 6.52, tokens: 1.83}
      medium: {exposure: 5.88, satisfaction: 7.28, tokens: 1.93}
      hard:   {exposure: 6.73, satisfaction: 8.66, tokens: 2.16}

  GPT-5.6-Luna:
    Prompted Base:
      easy:   {exposure: 3.34, satisfaction: 4.43, tokens: 2.91}
      medium: {exposure: 3.59, satisfaction: 4.71, tokens: 2.82}
      hard:   {exposure: 3.69, satisfaction: 5.09, tokens: 2.71}
    +Thinking:
      easy:   {exposure: 3.42, satisfaction: 4.47, tokens: 2.98}
      medium: {exposure: 3.49, satisfaction: 4.60, tokens: 3.04}
      hard:   {exposure: 3.75, satisfaction: 5.22, tokens: 2.92}
    ExpRAG:
      easy:   {exposure: 3.30, satisfaction: 4.30, tokens: 3.28}
      medium: {exposure: 3.55, satisfaction: 4.80, tokens: 3.42}
      hard:   {exposure: 3.98, satisfaction: 5.45, tokens: 3.45}
    ReMem:
      easy:   {exposure: 3.47, satisfaction: 4.46, tokens: 2.77}
      medium: {exposure: 3.67, satisfaction: 4.73, tokens: 2.74}
      hard:   {exposure: 3.95, satisfaction: 5.43, tokens: 2.84}
    Trace2Skill-like:
      easy:   {exposure: 3.57, satisfaction: 4.57, tokens: 3.40}
      medium: {exposure: 3.60, satisfaction: 4.87, tokens: 3.43}
      hard:   {exposure: 3.88, satisfaction: 5.30, tokens: 3.29}
    Probing:
      easy:   {exposure: 4.47, satisfaction: 5.55, tokens: 1.87}
      medium: {exposure: 4.86, satisfaction: 6.07, tokens: 1.88}
      hard:   {exposure: 5.28, satisfaction: 6.89, tokens: 1.85}
```

---

# 5. Status

`Ours` and `AITR` are currently pending.

When new results are available, update this file without changing the metric definitions or baseline values unless an experiment is rerun.
