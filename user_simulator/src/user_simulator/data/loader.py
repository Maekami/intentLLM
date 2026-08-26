import json
import random
from pathlib import Path

from pydantic import ValidationError

from user_simulator.data.validator import DatasetValidator
from user_simulator.domain.dag import Sample
from user_simulator.exceptions import DatasetValidationError


class DatasetLoader:
    def __init__(
        self,
        path: str | Path = "dataset/DAG.jsonl",
    ) -> None:
        self.path = Path(path)
        self.validator = DatasetValidator()
        self._cache: list[Sample] | None = None

    def load_all_samples(self) -> list[Sample]:
        if self._cache is not None:
            return list(self._cache)
        samples: list[Sample] = []
        try:
            with self.path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        samples.append(Sample.model_validate_json(line))
                    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                        raise DatasetValidationError(
                            f"{self.path}:{line_number}: invalid sample: {exc}"
                        ) from exc
        except OSError as exc:
            raise DatasetValidationError(f"cannot read {self.path}: {exc}") from exc
        report = self.validator.validate_samples(samples)
        report.raise_for_errors()
        self._cache = samples
        return list(samples)

    def load_sample_by_id(self, sample_id: str) -> Sample:
        for sample in self.load_all_samples():
            if sample.sample_id == sample_id:
                return sample
        raise DatasetValidationError(f"unknown sample_id: {sample_id}")

    def sample_randomly(self, seed: int) -> Sample:
        samples = self.load_all_samples()
        if not samples:
            raise DatasetValidationError("dataset is empty")
        return random.Random(seed).choice(samples)
