import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio para la función seleccionar_features_por_peso.
    Retorna:
        input_dict (dict): Diccionario con X (DataFrame), y (Series) y porcentaje_acumulado.
        output_esperado (np.ndarray): Array de nombres de columnas que suman el peso indicado.
    """
    # 1. Configuración aleatoria
    n_muestras = np.random.randint(100, 200)
    # Creamos 10 columnas en total: 3 informativas, 2 redundantes y 5 repetidas/ruido
    n_features = 10
    porcentaje_acumulado = np.random.choice([0.70, 0.85, 0.95])

    X_raw, y_raw = make_classification(
        n_samples=n_muestras,
        n_features=n_features,
        n_informative=3,
        n_redundant=2,
        random_state=42
    )

    nombres_cols = [f"feat_{i:02d}" for i in range(n_features)]
    X = pd.DataFrame(X_raw, columns=nombres_cols)
    y = pd.Series(y_raw)

    # 2. Calcular el output esperado (Lógica interna)
    # Entrenamos el modelo para obtener importancias
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # Crear un DataFrame de importancias para ordenar
    importancias = pd.Series(rf.feature_importances_, index=nombres_cols).sort_values(ascending=False)

    # Calcular suma acumulada
    importancia_acumulada = np.cumsum(importancias.values)

    # Encontrar el índice donde se supera el umbral
    # Buscamos el primer índice donde la suma es >= porcentaje_acumulado
    indice_corte = np.where(importancia_acumulada >= porcentaje_acumulado)[0][0]

    # Seleccionamos los nombres de las columnas hasta ese índice (inclusive)
    output_esperado = importancias.index[:indice_corte + 1].values

    # 3. Formatear el input
    input_dict = {
        "X": X,
        "y": y,
        "porcentaje_acumulado": porcentaje_acumulado
    }

    return input_dict, output_esperado

# --- Ejemplo de Inspección ---
inp, out = generar_caso_de_uso_preparar_datos()

print(f"--- PARÁMETROS ---")
print(f"Porcentaje deseado: {inp['porcentaje_acumulado'] * 100}%")

print("\n--- INPUT (X) ---")
print(f"Dimensiones de la matriz X: {inp['X'].shape}")
print(f"Primeras 3 filas:\n{inp['X'].head(3)}")

print("\n--- OUTPUT ESPERADO ---")
print(f"Columnas seleccionadas: {out}")