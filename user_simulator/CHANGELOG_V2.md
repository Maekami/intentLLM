# V2 Changelog

## Behavior

- Added operational V2 controller, satisfaction, clear-realizer, and
  abstract-realizer prompts.
- Added registered V2 wire schemas with strict Pydantic validation, canonical
  JSON Schema generation, and schema hashing.
- Bound JSON Schema type, strictness, and `require_parameters` to typed model
  configuration.
- Added explicit handling for missing choices, refusals, empty output,
  truncation, invalid JSON, schema violations, and retry exhaustion.
- Added ordered semantic validation with specific correction feedback.
- Added a registry-based component factory and configuration-driven prompt
  injection.
- Added deterministic `--mock` demo mode, prompt/component CLI overrides, and
  latent-summary hiding by default.
- Replaced raw-only turn display with a consolidated audit panel and added
  `/raw-audit`, `/prompts`, and `/schemas`.
- Added the offline prompt smoke-test utility and fixture cases.

## Compatibility

- Episode ordering, prefix closure, monotonic satisfaction, END handling,
  backbone exposure, and difficulty policies are unchanged.
- V1 prompt files remain in the repository; V2 is the default.
- Public protocol method shapes remain compatible, with schema version metadata
  added as an optional argument.
- The source dataset format and contents are unchanged.

## Verification

```bash
ruff format --check src tests
ruff check src tests
pytest -q
python -m user_simulator.cli.validate_dataset --dataset dataset/DAG.jsonl
python -m user_simulator.cli.prompt_smoke_test \
  --component all \
  --cases tests/fixtures/prompt_cases.jsonl \
  --output-dir runs/prompt_smoke
```
