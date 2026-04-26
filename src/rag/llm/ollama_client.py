import logging
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for the local Ollama chat model.

    This is config only. It does not create or invoke the model.
    """

    model_name: str = "llama3.1:8b"
    temperature: float = 0.0
    base_url: str | None = None
    timeout: int | None = None
    num_ctx: int | None = None
    keep_alive: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """
    Normalized response returned by our app-facing LLM client.

    Upper layers should depend on this instead of depending directly
    on LangChain's raw response object.
    """

    content: str
    model_name: str
    raw_response: Any | None = None


class LLMException(Exception):
    """Raised when the LLM client cannot complete an operation."""


def build_ollama_chat_client(config: LLMConfig) -> ChatOllama:
    """
    Factory function for creating the raw LangChain ChatOllama client.

    This is the only place where ChatOllama should be instantiated.
    """

    try:
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
            timeout=config.timeout,
            num_ctx=config.num_ctx,
            keep_alive=config.keep_alive,
        )
    except Exception as exc:
        raise LLMException(
            f"Failed to build Ollama chat client for model '{config.model_name}'."
        ) from exc


class OllamaLLMClient:
    """
    App-facing LLM adapter.

    This class hides LangChain/Ollama details from the rest of the app.
    The query layer should use this class, not ChatOllama directly.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = build_ollama_chat_client(self.config)

    def invoke_messages(
        self,
        messages: list[tuple[str, str]] | list[BaseMessage],
    ) -> LLMResponse:
        """
        Main generation method.

        Receives already-prepared messages and returns a normalized response.
        It does not build prompts, retrieve documents, or know anything about RAG.
        """

        try:
            logger.info("Invoking Ollama model: %s", self.config.model_name)

            response = self._client.invoke(messages)

            content = getattr(response, "content", None)

            if not isinstance(content, str):
                raise LLMException(
                    f"Unexpected LLM response format from model '{self.config.model_name}'."
                )

            return LLMResponse(
                content=content,
                model_name=self.config.model_name,
                raw_response=response,
            )

        except LLMException:
            raise
        except Exception as exc:
            raise LLMException(
                f"Failed to invoke Ollama model '{self.config.model_name}'."
            ) from exc

    def invoke_prompt(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """
        Convenience method for simple calls.

        The main application should still prefer invoke_messages(),
        because RAG prompts will usually be assembled as structured messages.
        """

        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]

        return self.invoke_messages(messages)

    def health_check(self) -> bool:
        """
        Verifies that Ollama and the configured model are reachable.

        This can later be used by a FastAPI /health endpoint.
        """

        try:
            response = self.invoke_prompt(
                system_prompt="You are a health-check assistant. Reply with only: ok",
                user_prompt="health check",
            )

            return bool(response.content.strip())

        except Exception as exc:
            logger.warning(
                "Ollama health check failed for model '%s': %s",
                self.config.model_name,
                exc,
            )
            return False