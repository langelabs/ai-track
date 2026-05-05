class ModelNotFound(Exception):
    """Raised when a requested model cannot be found."""

    def __init__(self, model: str) -> None:
        """Initialize the error for a missing model.

        Parameters:
            model: The model identifier that could not be resolved.
        """
        super().__init__(f"Model {model} not found.")
