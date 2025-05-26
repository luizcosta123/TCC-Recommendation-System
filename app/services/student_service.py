from app.models.student import Student
import pandas as pd

def student_vector(student: Student, columns: list[str]) -> pd.DataFrame:
    linha = {}

    for col in columns:
        for codigo, info in student.disciplines.items():
            if f" - {codigo}" in col:
                if "Você foi aprovado" in col:
                    linha[col] = int(info.approved)
                elif "Dificuldade" in col:
                    linha[col] = info.difficulty
                elif "Interesse" in col:
                    linha[col] = info.interest
                elif "Importância" in col:
                    linha[col] = info.importance
                break
        else:
            linha[col] = 0

    return pd.DataFrame([linha])