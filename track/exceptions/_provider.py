class ProviderNotSupported(Exception):
    def __init__(self, provider:str):
        self.add_note(f"Provider {provider} not supported.")
