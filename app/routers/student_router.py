import joblib
from fastapi import APIRouter
from app.models.student import Student
from app.services.student_service import recomendar_disciplinas, student_vector
from app.utils import load_grid

router = APIRouter()

@router.post("/cluster")
def classificar_aluno(student: Student):
    model = joblib.load("kmeans/models/kmeans_model.pkl")
    scaler = joblib.load("kmeans/models/kmeans_scaler.pkl")
    with open("kmeans/data/columns.txt", encoding="utf-8") as f:
        columns = f.read().splitlines()

    vector = student_vector(student, columns)
    vector_transformado = scaler.transform(vector)
    cluster = model.predict(vector_transformado)[0]

    print(model.predict(vector_transformado))

    return {"cluster": int(cluster)}