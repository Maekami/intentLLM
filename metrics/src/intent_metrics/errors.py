class MetricsError(Exception):
    """Base error for trace loading and metric evaluation."""


class TraceLoadError(MetricsError):
    """Raised when a pipeline run cannot be reconstructed faithfully."""


class MetricDataError(MetricsError):
    """Raised when a required canonical metric field is absent or invalid."""


class AITREvaluationError(MetricsError):
    """Raised when no valid AITR judgment can be obtained."""


class CacheError(MetricsError):
    """Raised when an AITR cache is malformed or cannot be used safely."""
