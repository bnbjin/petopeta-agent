"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans to answer user questions,
conducting research, and formulating responses.
"""

from typing import Literal, TypedDict, cast, Union

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from backend.retrieval_graph.configuration import AgentConfiguration
from backend.retrieval_graph.researcher_graph.graph import graph as researcher_graph
from backend.retrieval_graph.pet_manager.filter_graph import graph as pet_filter_graph
from backend.retrieval_graph.state import (
    AgentState,
    InputState,
    Router,
    Pet,
)
from backend.utils import format_docs, load_chat_model


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> Command[
    Literal["get_and_update_pet_info", "ask_for_more_info", "respond_to_general_query"]
]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages

    router = cast(
        Router,
        await model.with_structured_output(Router).ainvoke(messages),
    )

    goto = None
    match router["type"]:
        case "health":
            goto = "get_and_update_pet_info"
        case "behavior":
            goto = "get_and_update_pet_info"
        case "disease":
            goto = "get_and_update_pet_info"
        case "more-info":
            goto = "ask_for_more_info"
        case _:
            goto = "respond_to_general_query"

    return Command(
        update={"router": router},
        goto=goto,
    )


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.more_info_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.general_system_prompt.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def get_and_update_pet_info(
    state: AgentState,
    *,
    config: RunnableConfig,
) -> dict[str, list[Pet]]:
    """filter and update pet info."""

    response = await pet_filter_graph.ainvoke({"messages": state.messages})
    target_pets = response.get("result_pets", [])
    return {"pets": target_pets}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Union[list[str], str]]:
    """Create a step-by-step research plan for answering a pet-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    if len(state.pets) == 0:
        return {"steps": [], "documents": "delete"}

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Plan)
    messages = [
        {
            "role": "system",
            "content": configuration.research_plan_system_prompt,
        },
        {
            "role": "ai",
            "content": f"<pet-information> {state.pets[0] if state.pets else 'no pet found information'} </pet-information>",
        },
    ] + state.messages
    response = cast(
        Plan, await model.ainvoke(messages, {"tags": ["langsmith:nostream"]})
    )
    return {
        "steps": response["steps"],
        "documents": "delete",
        # "query": state.messages[-1].content,
    }


async def conduct_research(
    state: AgentState,
) -> Command[Literal["respond"]]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.
    
    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        Command[Literal["respond"]]: A command to update the state with the retrieved documents and removes the completed step.
        If add conduct_research to the literal, error will occur. Could be a bug.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """

    if len(state.steps) == 0:
        return Command(
            update={"documents": [], "steps": []},
            goto="respond",
        )

    result = await researcher_graph.ainvoke(
        {"question": state.steps[0], "pet": state.pets[0]}
    )

    return Command(
        update={"documents": result["documents"], "steps": state.steps[1:]},
        goto="conduct_research",
    )


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Union[list[BaseMessage], str]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)

    top_k = 20
    context = format_docs(state.documents[:top_k])
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "ai",
            "content": f"<pet-information> {state.pets[0] if state.pets else 'no pet found information'} </pet-information>",
        },
    ] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response], "answer": response.content}


# Define the graph
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)

builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(get_and_update_pet_info)
builder.add_node(create_research_plan)
builder.add_node(conduct_research)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_edge("ask_for_more_info", END)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("get_and_update_pet_info", "create_research_plan")
builder.add_edge("create_research_plan", "conduct_research")
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "PetoPeta"
