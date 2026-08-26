class SimulatorError(Exception):
    """Base simulator error."""


class DatasetValidationError(SimulatorError):
    pass


class ConfigurationError(SimulatorError):
    pass


class GraphInvariantError(SimulatorError):
    pass


class StructuredOutputError(SimulatorError):
    pass


class ControllerOutputError(StructuredOutputError):
    pass


class SatisfactionOutputError(StructuredOutputError):
    pass


class UserGenerationError(StructuredOutputError):
    pass


class InvalidTerminalStateError(SimulatorError):
    pass


class EpisodeTurnLimitError(SimulatorError):
    pass


class ModelRequestError(SimulatorError):
    pass


class OpenRouterRequestError(ModelRequestError):
    pass
