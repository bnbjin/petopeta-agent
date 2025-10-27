"""Researcher graph used in the conversational retrieval system as a subgraph.

This module defines the core structure and functionality of the researcher graph,
which is responsible for generating search queries and retrieving relevant documents.
"""

from typing import cast, Literal, Union, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from langgraph.graph import END, START, StateGraph
from langchain_community.retrievers import TavilySearchAPIRetriever

from backend import retrieval
from backend.retrieval_graph.configuration import AgentConfiguration
from backend.retrieval_graph.researcher_graph.state import QueryState, ResearcherState
from backend.utils import load_chat_model


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> Command[Literal["retrieve_documents"]]:
    """Generate search queries based on the question (a step in the research plan).

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(TypedDict):
        queries: list[str]

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Response)
    messages = [
        {"role": "system", "content": configuration.generate_queries_system_prompt},
        {"role": "ai", "content": f"<pet-information> {state.pet} </pet-information>"},
        {"role": "human", "content": state.question},
    ]
    response = cast(
        Response, await model.ainvoke(messages, {"tags": ["langsmith:nostream"]})
    )

    return Command(
        goto=[
            Send("retrieve_documents", QueryState(query=q)) for q in response["queries"]
        ]
    )


async def retrieve_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, Union[list[Document], str]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration with the retriever used to fetch documents.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """

    docs = []

    # async with retrieval.amake_retriever(config) as library_retriever:
    #     library_docs = await library_retriever.ainvoke(state.query, config)
    #     docs.extend(library_docs)

    web_retriever = TavilySearchAPIRetriever(k=3)
    retrival_chain = web_retriever
    web_docs = await retrival_chain.ainvoke(state.query, config)
    docs.extend(web_docs)

    # with retrieval.make_retriever(config) as library_retriever:
    #     library_docs = await library_retriever.ainvoke(state.query, config)
    #     docs.extend(library_docs)

    return {"documents": docs}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_documents", QueryState(query=query)) for query in state.queries
    ]


# Define the graph
builder = StateGraph(ResearcherState)

builder.add_node(generate_queries)
builder.add_node(retrieve_documents)

builder.add_edge(START, "generate_queries")
builder.add_edge("retrieve_documents", END)

graph = builder.compile()
graph.name = "ResearcherGraph"
