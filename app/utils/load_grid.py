import json
from pathlib import Path
from app.models.curriculum_grid import CurriculumGrid

def load_grid(path: str = "app/data/grid.json") -> CurriculumGrid:
    with open(Path(path), encoding="utf-8") as f:
        data = json.load(f)
    return CurriculumGrid(disciplines=data)

