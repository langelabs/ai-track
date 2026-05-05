from track.contracts import AiModel
from providers import AiProvider, OpenRouterProvider, LocalProvider
from track.exceptions import ProviderNotSupported


class AiHub:

    def __init__(self,
                 hugging_face_secret: str | None = None,
                 openrouter_secret: str | None = None,
                 model_dir: str | None = None):
        self._models: dict[AiModel, AiProvider] = {}

        self.model_dir = model_dir

        # secrets
        self._hugging_face_secret = hugging_face_secret
        self._openrouter_secret = openrouter_secret

    @property
    def models(self) -> list[AiModel]:
        return list(self._models.keys())

    def add_model(self, model: AiModel):
        if model.provider == "local":
            self._models[model] = LocalProvider(model)
        elif model.provider == "openrouter":
            self._models[model] = OpenRouterProvider(model, api_key=self._openrouter_secret)
        else:
            raise ProviderNotSupported(model.provider)

    def remove_model(self, model: AiModel):
        self._models.pop(model)

    async def load_model(self, model: AiModel):
        await self._models[model].load()

    def get_client(self, model: AiModel):
        return self._models[model].get_client()

    def get_async_client(self, model: AiModel):
        return self._models[model].get_client()
