from langchain_ollama import ChatOllama
from pydantic import BaseModel


class LLMClient(BaseModel):
    model_name: str
    temperature: float | None = None
    base_url: str | None = None
    num_ctx: int | None = None
    keep_alive: bool | None = None

def get_llm_client(client: LLMClient) -> ChatOllama:
    return ChatOllama(
        model=client.model_name,
        temperature=client.temperature,
        base_url=client.base_url,
        num_ctx=client.num_ctx,
        keep_alive=client.keep_alive
    )

class LLMService(BaseModel):
    llm_client: LLMClient
    def get_llm_client(self) -> ChatOllama:
        try:
            return get_llm_client(self.llm_client)
        except LLMException as e:
            raise LLMException(f"Failed to get LLM client: {e}")
        

    def invoke(self, messages: list[tuple[str, str]]) -> str:
        try:
            return self.get_llm_client().invoke(messages).content()
        except LLMException as e:
            raise LLMException(f"Failed to invoke LLM: {e}")


class LLMException(Exception):
    pass


# messages = [
#     ("system", "You are a helpful assistant that translates from English to French. Translate the user sentence into French."),
#     ("user", "I am hoping to build a chatbot that can help me with my homework.")
# ]



