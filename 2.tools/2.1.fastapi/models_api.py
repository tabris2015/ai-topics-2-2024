from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
import random


class RaceEnum(str, Enum):
    orc = "ORC"
    elf = "ELF"
    human = "HUMAN"
    goblin = "GOBLIN"

class Guild(BaseModel):
    id: int
    name: str
    realm: str
    created: datetime

class Character(BaseModel):
    id: int
    name: str
    level: int
    race: RaceEnum
    hp: int
    damage: int | None = None # opcional
    guild: Guild

class CharacterCreate(BaseModel):
    name: str
    level: int
    race: RaceEnum
    hp: int
    damage: int
    guild_id: int

app = FastAPI(title="validacion")

guilds = []
characters = []

@app.post("/guilds", status_code=201)
def create_guild(guild: Guild) -> list[Guild]:
    guilds.append(guild)
    return guilds

@app.post("/characters", status_code=201)
def create_character(character: CharacterCreate):
    id = random.randint(0, 9999)
    guilds_found = [g for g in guilds if g.id == character.guild_id]
    if not guilds_found:
        raise HTTPException(status_code=404, detail="guild not found")
    guild = guilds_found[0]
    new_character = Character(id=id, guild=guild, **character.model_dump(exclude=["guild_id"]))

    characters.append(new_character)
    return characters

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("models_api:app", reload=True, port=8008)