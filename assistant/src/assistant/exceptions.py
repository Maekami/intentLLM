class AssistantError(Exception):
    """Base assistant error."""


class ConfigurationError(AssistantError):
    pass


class ModelRequestError(AssistantError):
    pass


class OpenRouterRequestError(ModelRequestError):
    """Backward-compatible OpenRouter-specific request error."""

    pass


class InvalidModelResponseError(AssistantError):
    pass
