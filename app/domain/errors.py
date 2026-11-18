"""Domain errors. Inbound adapters map each one to an HTTP status code."""


class DomainError(Exception):
    """Base class for every business rule violation."""


class InvalidInputError(DomainError):
    """Input that breaks a business rule (HTTP 422)."""


class InvalidTrainingFileError(InvalidInputError):
    """Uploaded training CSV with a wrong extension, size or columns (HTTP 422)."""


class AuthenticationError(DomainError):
    """Missing, invalid or expired credentials (HTTP 401)."""


class PermissionDeniedError(DomainError):
    """Authenticated user without the required role (HTTP 403)."""


class NotFoundError(DomainError):
    """Requested resource does not exist (HTTP 404)."""


class EmailAlreadyRegisteredError(DomainError):
    """Registration with an email that already exists (HTTP 409)."""


class RetrainInProgressError(DomainError):
    """A retraining job is already running (HTTP 409)."""


class ModelNotAvailableError(DomainError):
    """No active model version can serve predictions (HTTP 503)."""


class AdminProvisioningError(DomainError):
    """The configured admin email belongs to a non-admin account (startup fails)."""
