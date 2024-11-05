'''
Hernández Jiménez Erick Yael
Patiño Flores Samuel
Robert Garayzar Arturo

Descripción:
Programa que compara el clasificador KNN con distintos métodos de validación.
Buscando el valor mas óptimo de K.
'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

class ClasificadorKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_entrenamiento = []
        self.y_entrenamiento = []

    def ajustar(self, X_entrenamiento, y_entrenamiento):
        self.X_entrenamiento = X_entrenamiento
        self.y_entrenamiento = y_entrenamiento

    def predecir(self, X_prueba):
        return [self.predecir_individual(x) for x in X_prueba]

    def predecir_individual(self, x):
        distancias = [(self.distancia_euclidiana(x, x_ent), y) for x_ent, y in zip(self.X_entrenamiento, self.y_entrenamiento)]
        vecinos_cercanos = sorted(distancias, key=lambda dist: dist[0])[:self.k]
        etiquetas_vecinos = [etiqueta for _, etiqueta in vecinos_cercanos]
        etiqueta_comun = Counter(etiquetas_vecinos).most_common(1)[0][0]
        return etiqueta_comun

    def distancia_euclidiana(self, punto1, punto2):
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(punto1, punto2)))

# Función para evaluar KNN con distintos métodos de validación
def evaluar_knn(X, y, max_k=10):
    mejor_k = {}
    mejor_precision = {}
    mejor_matriz_confusion = {}

    # Validación Cruzada Estratificada 10-Fold
    kfold_estratificado = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    precisiones_kfold = []
    matrices_confusion_kfold = []
    for k in range(1, max_k + 1):
        knn = ClasificadorKNN(k=k)
        precisiones_fold = []
        confusiones_fold = []
        for indices_entrenamiento, indices_prueba in kfold_estratificado.split(X, y):
            knn.ajustar(X[indices_entrenamiento], y[indices_entrenamiento])
            y_pred = knn.predecir(X[indices_prueba])
            precisiones_fold.append(accuracy_score(y[indices_prueba], y_pred))
            confusiones_fold.append(confusion_matrix(y[indices_prueba], y_pred))

        precision_media = np.mean(precisiones_fold)
        matriz_confusion_media = sum(confusiones_fold)
        precisiones_kfold.append(precision_media)
        matrices_confusion_kfold.append(matriz_confusion_media)
        print(f"10-Fold CV - K={k}: Accuracy={precision_media:.4f}")

    mejor_kfold_k = np.argmax(precisiones_kfold) + 1
    mejor_k["10-Fold"] = mejor_kfold_k
    mejor_precision["10-Fold"] = max(precisiones_kfold)
    mejor_matriz_confusion["10-Fold"] = matrices_confusion_kfold[mejor_kfold_k - 1]

    # Validación Cruzada Leave-One-Out
    loo = LeaveOneOut()
    precisiones_loo = []
    matrices_confusion_loo = []
    for k in range(1, max_k + 1):
        knn = ClasificadorKNN(k=k)
        precisiones = []
        confusiones = []
        for indices_entrenamiento, indice_prueba in loo.split(X):
            knn.ajustar(X[indices_entrenamiento], y[indices_entrenamiento])
            y_pred = knn.predecir([X[indice_prueba][0]])[0]
            precisiones.append(accuracy_score([y[indice_prueba][0]], [y_pred]))
            confusiones.append(confusion_matrix([y[indice_prueba][0]], [y_pred], labels=np.unique(y)))

        precision_media = np.mean(precisiones)
        matriz_confusion_media = sum(confusiones)
        precisiones_loo.append(precision_media)
        matrices_confusion_loo.append(matriz_confusion_media)
        print(f"LOO CV - K={k}: Accuracy={precision_media:.4f}")

    mejor_loo_k = np.argmax(precisiones_loo) + 1
    mejor_k["LOO"] = mejor_loo_k
    mejor_precision["LOO"] = max(precisiones_loo)
    mejor_matriz_confusion["LOO"] = matrices_confusion_loo[mejor_loo_k - 1]

    # Validación Hold-Out 70/30 Estratificada
    precisiones_holdout = []
    matrices_confusion_holdout = []
    for k in range(1, max_k + 1):
        knn = ClasificadorKNN(k=k)
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        knn.ajustar(X_entrenamiento, y_entrenamiento)
        y_pred = knn.predecir(X_prueba)
        precision = accuracy_score(y_prueba, y_pred)
        matriz_confusion = confusion_matrix(y_prueba, y_pred)
        precisiones_holdout.append(precision)
        matrices_confusion_holdout.append(matriz_confusion)
        print(f"Hold-Out 70/30 - K={k}: Accuracy={precision:.4f}")

    mejor_holdout_k = np.argmax(precisiones_holdout) + 1
    mejor_k["Hold-Out"] = mejor_holdout_k
    mejor_precision["Hold-Out"] = max(precisiones_holdout)
    mejor_matriz_confusion["Hold-Out"] = matrices_confusion_holdout[mejor_holdout_k - 1]

    # Determinar el método con la mayor precisión promedio
    mejor_metodo = max(mejor_precision, key=mejor_precision.get)
    k_optimo = mejor_k[mejor_metodo]
    matriz_confusion_optima = mejor_matriz_confusion[mejor_metodo]
    print(f"\nMétodo más óptimo: {mejor_metodo}")
    print(f"Valor óptimo de K: {k_optimo} con una precisión de {mejor_precision[mejor_metodo]:.4f}")

    # Mostrar la matriz de confusión usando matplotlib
    mostrar_matriz_confusion(matriz_confusion_optima, mejor_metodo, k_optimo)

    return mejor_metodo, k_optimo, matriz_confusion_optima

# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion(matriz_confusion, metodo, k_optimo):
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión - {metodo} con K={k_optimo}')
    plt.show()

# Cargar los datasets y evaluar
def cargar_datos_iris():
    datos = load_iris()
    return datos.data, datos.target

def cargar_datos_vino():
    datos = load_wine()
    return datos.data, datos.target

def cargar_datos_cancer():
    datos = load_breast_cancer()
    return datos.data, datos.target

# Ejemplo de uso con los datasets
if __name__ == "__main__":
    conjuntos_datos = {
        "Iris": cargar_datos_iris(),
        "Wine": cargar_datos_vino(),
        "Breast Cancer": cargar_datos_cancer()
    }

    for nombre_datos, (X, y) in conjuntos_datos.items():
        print(f"\nDataset {nombre_datos}:")
        mejor_metodo, k_optimo, matriz_confusion_optima = evaluar_knn(X, y, max_k=10)
        print(f"El método más óptimo para el dataset {nombre_datos} es {mejor_metodo} con K={k_optimo}\n")
