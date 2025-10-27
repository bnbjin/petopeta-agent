from typing import Annotated
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig

from backend.utils import load_chat_model
from backend.retrieval_graph.pet_manager.tools import (
    add_or_update_pet,
    get_pets,
    delete_pet,
)
from backend.retrieval_graph.configuration import AgentConfiguration
from backend.retrieval_graph.state import AgentState, PetList

configuration = AgentConfiguration()
model = load_chat_model(configuration.query_model)
tools = [
    get_pets,
    add_or_update_pet,
    delete_pet,
]

llm_input_trimmer = trim_messages(
    # token_counter=count_tokens_approximately,
    token_counter=len,
    max_tokens=32,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)


async def get_pet_manager_graph():
    async def prepare_model_inputs(
        state: AgentState,
        *,
        config: RunnableConfig,
    ):
        messages_trimmed = await llm_input_trimmer.ainvoke(state.get("messages", []))

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=configuration.get_and_update_pet_info_system_prompt
                ),
                *messages_trimmed,
            ]
        )

        return prompt

    graph = create_react_agent(
        model,
        tools,
        prompt=prepare_model_inputs,
    )
    graph.name = "PetInformationManager"
    return graph
