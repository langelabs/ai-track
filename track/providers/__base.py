"""Provider contracts for OpenAI-compatible backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from track.contracts import AiModel
from openai import Client, AsyncClient


class AiProvider(ABC):
    """Define the shared interface for model providers."""

    def __init__(self, model:AiModel, api_key:str|None=None) -> None:
        self.model = model
        self._api_key = api_key

        self.loaded:bool = False
        self.downloaded:bool = False

    @abstractmethod
    def get_client(self) -> Client:
        """
        Return the client for this model.
        :return: OpenAI-compatible client for the model
        """

    @abstractmethod
    def get_async_client(self) -> AsyncClient:
        """
        Return the asynchronous client for the model.
        :return: OpenAI-compatible client for the model
        """

    @abstractmethod
    async def download(self, model_dir:str|None = None) -> bool:
        """
        Download a model from HuggingFace.
        :param model_dir: the directory to save the model
        :return: Boolean indicating whether the model was downloaded
        """

    @abstractmethod
    async def load(self, model_dir:str|None = None) -> bool:
        """
        Load the model into compute.
        :param model_dir: Optional directory where the model is saved.
        :return: Boolean indicating whether the model was loaded
        """