from sentence_transformers import SentenceTransformer
import numpy as np


def digest(text) -> None:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text)


def vector_similarity(vector1, vector2):
    # ==>> Multiplicación sin usar la librería numpy <<==
    # result = 0
    # for i in range(0, len(vector1)):
    #     result = result + (vector1[i] * vector2[i])
    # return result
    return np.dot(np.squeeze(np.array(vector1)), np.squeeze(np.array(vector2)))


phrase1 = 'Apple es el nombre de la fruta manzana en inglés'
embedding1 = digest(phrase1)
phrase2 = 'Apple ha creado un nuevo Iphone'
embedding2 = digest(phrase2)
phrase3 = 'Manzana es una fruta'
embedding3 = digest(phrase3)
phrase4 = 'Los Iphone de Apple son caros'
embedding4 = digest(phrase4)

print("Longitud embedding1: ", len(embedding1))
print("Longitud embedding2: ", len(embedding2))

print("****** Mismos Embeddings *******")
print("Grado de similitud, e1, e1: ", vector_similarity(embedding1, embedding1))

print("****** Embedding 1 *******")
print("Grado de similitud, e1, e2: ", vector_similarity(embedding1, embedding2))
print("Grado de similitud, e1, e3: ", vector_similarity(embedding1, embedding3))
print("Grado de similitud, e1, e4: ", vector_similarity(embedding1, embedding2))

print("****** Embedding 2 *******")
print("Grado de similitud, e2, e3: ", vector_similarity(embedding2, embedding3))
print("Grado de similitud, e2, e4: ", vector_similarity(embedding2, embedding4))

print("****** Embedding 3 *******")
print("Grado de similitud, e3, e4: ", vector_similarity(embedding3, embedding4))








