import os
from contextlib import contextmanager, asynccontextmanager
from typing import Iterator, AsyncIterator

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_weaviate import WeaviateVectorStore
from langchain_postgres import PGVectorStore, PGEngine

from backend.configuration import BaseConfiguration
from backend.constants import DOCS_INDEX_NAME
from backend.embeddings import get_embeddings_model


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


@contextmanager
def make_weaviate_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Iterator[BaseRetriever]:
    """Caution: use 0.0.3 version of WeaviateVectorStore, there are bugs with 0.0.4"""

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.classes.init.Auth.api_key(
            os.environ.get("WEAVIATE_API_KEY", "not_provided")
        ),
        skip_init_checks=True,
    ) as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding_model,
            attributes=["source", "title"],
        )
        search_kwargs = {**configuration.search_kwargs, "return_uuids": True}
        yield store.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "weaviate":
            with make_weaviate_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )


@asynccontextmanager
async def amake_retriever(
    config: RunnableConfig,
) -> AsyncIterator[BaseRetriever]:
    """Create a retriever for the agent asynchronously, based on the current configuration."""

    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = get_embeddings_model()

    table_name = os.environ["VECTOR_TABLE_NAME"]
    pg_engine = PGEngine.from_connection_string(url=os.environ["VECTOR_DB_URL"])

    vectorstore = await PGVectorStore.create(
        engine=pg_engine,
        table_name=table_name,
        embedding_service=embedding_model,
    )
    search_kwargs = {**configuration.search_kwargs, "return_uuids": True}
    yield vectorstore.as_retriever(search_kwargs=search_kwargs)
