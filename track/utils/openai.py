from openai import Client, AsyncClient

def get_openai_client(base:str, api_key:str) -> Client:
    return Client(api_key=api_key, base_url=base)

def get_async_openai_client(base:str, api_key:str) -> AsyncClient:
    return AsyncClient(api_key=api_key, base_url=base)