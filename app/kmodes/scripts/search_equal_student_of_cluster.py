import pickle
import pandas as pd
import numpy as np

with open('../training/models/modelo_kmodes.pkl', 'rb') as f:
    kmodes_model = pickle.load(f)

df_new_student = pd.read_csv('../data/new_student_2.csv', header=None, nrows=1)
new_student_numpy = df_new_student.to_numpy()

df_training_students = pd.read_csv('../data/form_responses.csv', header=None)
column_headers = df_training_students.iloc[0]

df_training_students = df_training_students.drop(index=0).reset_index(drop=True)
training_students_numpy = df_training_students.to_numpy()

training_student_clusters = kmodes_model.predict(training_students_numpy)
df_training_students['cluster'] = training_student_clusters

predicted_cluster = kmodes_model.predict(new_student_numpy)[0]
print(f"Cluster Previsto: {predicted_cluster}")

same_cluster_students = df_training_students[df_training_students['cluster'] == predicted_cluster]

if not same_cluster_students.empty:
    same_cluster_students_numpy = same_cluster_students.drop('cluster', axis=1).to_numpy()

    distances = np.sum(same_cluster_students_numpy != new_student_numpy, axis=1)
    min_distance_local_index = np.argmin(distances)
    min_distance = distances[min_distance_local_index]

    most_similar_student_index = same_cluster_students.index[min_distance_local_index]

    print(f"Aluno Mais Similar: {most_similar_student_index}")
    print(f"Distância (respostas diferentes): {min_distance} de {new_student_numpy.shape[1]} atributos")

    new_student_responses = new_student_numpy.flatten()
    similar_student_responses = df_training_students.loc[most_similar_student_index].drop('cluster').to_numpy()

    num_courses = new_student_responses.shape[0] // 4

    courses = [column_headers[i * 4].split('\n')[0] for i in range(num_courses)]

    def extract_approvals(respostas):
        return [respostas[i * 4] for i in range(num_courses)]

    new_approvals = extract_approvals(new_student_responses)
    similar_approvals = extract_approvals(similar_student_responses)

    df_comparison = pd.DataFrame({
        'Disciplina': courses,
        'Novo Aluno': new_approvals,
        'Aluno Similar': similar_approvals
    })

    df_comparison.to_csv('../results/comparison_of_students.csv', index=False, encoding='utf-8-sig')
