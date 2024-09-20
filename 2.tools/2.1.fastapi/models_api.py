from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class RaceEnum(str, Enum):
    orc = "ORC"
    elf = "ELF"
    human = "HUMAN"
    goblin = "GOBLIN"

class Guild(BaseModel):
    name: str
    realm: str
    created: datetime

class Character(BaseModel):
    name: str
    level: int
    race: RaceEnum
    hp: int
    damage: int | None = None # opcional
    guild: Guild

app = FastAPI(title="validacion")

guilds = []
characters = []

@app.post("/guilds", status_code=201)
def create_guild(guild: Guild):
    guilds.append(guild)
    return guilds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("models_api:app", reload=True, port=8008)