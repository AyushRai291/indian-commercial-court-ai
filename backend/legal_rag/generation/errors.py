"""Generation-layer failures that are safe to map to generic HTTP errors."""


class GenerationError(RuntimeError):
    """Base class for grounded generation failures."""


class ProviderUnavailableError(GenerationError):
    """The configured model provider could not complete a request."""


class MalformedModelResponseError(GenerationError):
    """The provider did not return the required structured payload."""


class CitationIntegrityError(GenerationError):
    """Answer citations did not match the supplied request-local evidence IDs."""
