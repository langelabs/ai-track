from openai import Client, AsyncClient

from .__base import AiProvider

class LocalProvider(AiProvider):
    def get_client(self) -> Client:
        pass # todo

    def get_async_client(self) -> AsyncClient:
        pass # todo

    def download(self, model_dir:str|None = None) -> bool:
        pass # todo

    def load(self, model_dir:str|None = None) -> bool:
        pass # todo



