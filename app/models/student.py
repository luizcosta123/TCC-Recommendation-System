from pydantic import BaseModel, Field
from typing import Dict

class DisciplinesDetails(BaseModel):
    approved: bool
    difficulty: int = Field(ge=1, le=5)
    interest: int = Field(ge=1, le=5)
    importance: int = Field(ge=1, le=5)

class Student(BaseModel):
    disciplines: Dict[str, DisciplinesDetails]