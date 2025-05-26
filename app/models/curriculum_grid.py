from pydantic import BaseModel
from typing import List, Dict

class Discipline(BaseModel):
    name: str
    period: int
    prerequisites: List[str] = []

class CurriculumGrid(BaseModel):
    disciplines: Dict[str, Discipline]