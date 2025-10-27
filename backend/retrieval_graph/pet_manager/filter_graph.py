"""
This graph is used to filter the pets specified by the user from the store before the research.

1. Get all the pets recorded in the database.
2. Filter the pets recorded in the database.
3. Filter the pets not recorded in the database.
4. Add the new pets to the database.
5. Assemble the filtered pets.
"""

from typing import List, Dict, cast
from dataclasses import dataclass, field
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage

from backend.retrieval_graph.configuration import AgentConfiguration
from backend.retrieval_graph.state import InputState, Pet, PetList
from backend.retrieval_graph.pet_manager.tools import get_pets, add_or_update_pet
from backend.utils import load_chat_model
from backend.prompts_local.en import (
    FILTER_PETS_RECORDED_SYSTEM_PROMPT_STR,
    FILTER_PETS_RECORDED_AI_PROMPT_STR,
    FILTER_PETS_NOT_RECORDED_SYSTEM_PROMPT_STR,
)


@dataclass(kw_only=True)
class PetInformationFilterState(InputState):
    """State of the pet information filter graph."""

    pets_recorded: List[Dict] = field(default_factory=list)

    target_pets_recorded: List[Dict] = field(default_factory=list)

    new_pets: List[Pet] = field(default_factory=list)

    result_pets: List[Pet] = field(default_factory=list)


async def get_all_recorded_pets(
    state: PetInformationFilterState,
    *,
    config: RunnableConfig,
) -> Dict[str, List[Dict]]:
    """Get all the pets information of the user given in the config recorded."""

    pets = await get_pets.ainvoke({}, config=config)

    return {"pets_recorded": pets}


async def filter_pets_recorded(
    state: PetInformationFilterState,
    *,
    config: RunnableConfig,
) -> Dict[str, List[Dict]]:
    """Filter those pets information recorded in the store and specified by the user."""

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    pets = state.pets_recorded

    response = await model.with_structured_output(PetList).ainvoke(
        [
            SystemMessage(content=FILTER_PETS_RECORDED_SYSTEM_PROMPT_STR),
            AIMessage(
                content=FILTER_PETS_RECORDED_AI_PROMPT_STR.format(pets_recorded=pets)
            ),
            *state.messages,
        ]
    )
    pet_list = cast(PetList, response)

    return {"target_pets_recorded": pet_list.pets}


async def filter_pets_not_recorded(
    state: PetInformationFilterState,
    *,
    config: RunnableConfig,
) -> Dict[str, List[Dict]]:
    """Filter those pets information not recorded in the store and specified by the user."""

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)

    pets = state.pets_recorded

    response = await model.with_structured_output(PetList).ainvoke(
        [
            SystemMessage(content=FILTER_PETS_NOT_RECORDED_SYSTEM_PROMPT_STR),
            AIMessage(
                content=FILTER_PETS_RECORDED_AI_PROMPT_STR.format(pets_recorded=pets)
            ),
            *state.messages,
        ]
    )
    pet_list = cast(PetList, response)

    return {"new_pets": pet_list.pets}


async def add_new_pets_to_storage(
    state: PetInformationFilterState,
    *,
    config: RunnableConfig,
) -> Dict:
    """Add the new pets to the store with a tool."""

    for pet in state.new_pets:
        is_valid = []
        for must_have_key in ("name", "species"):
            is_valid.append(pet.get(must_have_key, "") != "")
        if all(is_valid):
            await add_or_update_pet.ainvoke(dict(pet), config=config)
    return {}


async def assembile_filter_pets(
    state: PetInformationFilterState,
    *,
    config: RunnableConfig,
) -> Dict:
    """Assemble the filtered pets."""

    result_pets = []
    result_pets.extend(state.target_pets_recorded)
    result_pets.extend([pet for pet in state.new_pets if pet.get("species", "") != ""])

    return {
        "result_pets": result_pets,
    }


builder = StateGraph(PetInformationFilterState)

builder.add_node(get_all_recorded_pets)
builder.add_node(filter_pets_recorded)
builder.add_node(filter_pets_not_recorded)
builder.add_node(add_new_pets_to_storage)
builder.add_node(assembile_filter_pets)

builder.add_edge(START, "get_all_recorded_pets")
builder.add_edge("get_all_recorded_pets", "filter_pets_recorded")
builder.add_edge("filter_pets_recorded", "filter_pets_not_recorded")
builder.add_edge("filter_pets_not_recorded", "add_new_pets_to_storage")
builder.add_edge("add_new_pets_to_storage", "assembile_filter_pets")
builder.add_edge("assembile_filter_pets", END)

graph = builder.compile()
graph.name = "PetInformationFilterGraph"
