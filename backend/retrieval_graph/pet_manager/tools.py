from typing import Annotated, Optional, List, cast, Dict
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.config import get_store

from backend.prompts_local.en import *


# TODO 现在对Store的访问可能有Race Condition的隐患，后续需要加上类似的读写锁的办法控制。


@tool(description=TOOL_ADD_PET_DESCRIPTION)
async def add_or_update_pet(
    name: Annotated[Optional[str], PET_NAME_DESCRIPTION],
    species: Annotated[Optional[str], PET_SPECIES_DESCRIPTION],
    breed: Annotated[Optional[str], PET_BREED_DESCRIPTION],
    gender: Annotated[Optional[str], PET_GENDER_DESCRIPTION],
    age: Annotated[Optional[int], PET_AGE_DESCRIPTION],
    weight: Annotated[Optional[int], PET_WEIGHT_DESCRIPTION],
    extra_condition: Annotated[Optional[str], PET_EXTRA_CONDITION_DESCRIPTION],
    *,
    config: RunnableConfig,
    # store: Annotated[BaseStore, InjectedStore()],
):
    store = get_store()

    user_id = config.get("metadata", {}).get("user_id")
    if not user_id:
        return

    namespace = ("pets", user_id)

    pets = await store.asearch(namespace)

    cur_pet = {
        "name": name,
        "species": species,
        "breed": breed,
        "gender": gender,
        "age": age,
        "weight": weight,
        "extra_condition": extra_condition,
    }

    if cur_pet["name"] and cur_pet["species"]:
        await store.aput(namespace, f"pet_{name}", cur_pet)


@tool(description=TOOL_GET_PETS_DESCRIPTION)
async def get_pets(*, config: RunnableConfig) -> List[Dict]:
    user_id = config.get("metadata", {}).get("user_id")
    if not user_id:
        return []

    store = get_store()

    namespace = ("pets", user_id)

    pets = await store.asearch(namespace)

    result = [pet.value for pet in pets]

    return result


@tool(description=TOOL_DELETE_PET_DESCRIPTION)
async def delete_pet(
    name: Annotated[Optional[str], PET_NAME_DESCRIPTION],
    species: Annotated[Optional[str], PET_SPECIES_DESCRIPTION],
    breed: Annotated[Optional[str], PET_BREED_DESCRIPTION],
    age: Annotated[Optional[int], PET_AGE_DESCRIPTION],
    *,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    user_id = config.get("metadata", {}).get("user_id")
    if not user_id:
        return NO_PET_FOUND_STR

    namespace = ("pets", user_id)

    pets = await store.asearch(namespace)
    matches = []
    index = 0
    for index, pet in enumerate(pets):
        pet = cast(dict, pet)
        for kd, vd in {"name": name, "species": species, "breed": breed, "age": age}:
            matches.append(vd == pet.get(kd, None))
        if all(matches):
            break
    if not all(matches):
        return NO_PET_FOUND_STR

    await store.adelete(namespace, f"pet_{pets[index]['name']}")

    return PET_DELETED_STR
