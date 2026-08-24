"""Verifier failures separated by client input and provider availability."""


class VerificationError(RuntimeError):
    """Base class for claim-verification failures."""


class InvalidVerificationRequestError(VerificationError):
    """The answer and supplied evidence failed structural citation checks."""


class MalformedVerifierResponseError(VerificationError):
    """The provider result did not exactly match the requested claim batch."""


class VerificationProviderUnavailableError(VerificationError):
    """The configured provider could not complete semantic verification."""
