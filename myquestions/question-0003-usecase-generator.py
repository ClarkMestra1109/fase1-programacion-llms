import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def generar_caso_de_uso_datos():
    """
    Genera un caso de uso aleatorio para la función identificar_variables_redundantes.
    Retorna:
        input_dict (dict): Diccionario con el DataFrame y el umbral.
        output_esperado (list): Lista de columnas con VIF > umbral.
    """
    # 1. Configuración aleatoria
    n_filas = np.random.randint(50, 100)
    umbral = np.random.choice([5.0, 10.0])

    # 2. Crear variables base (independientes)
    col1 = np.random.normal(0, 1, n_filas)
    col2 = np.random.normal(5, 2, n_filas)

    # 3. Crear variables redundantes (multicolinealidad)
    # col3 es casi una combinación lineal de col1 y col2
    col3 = col1 * 2 + col2 * 0.5 + np.random.normal(0, 0.01, n_filas)

    # col4 es una copia con muy poco ruido de col1
    col4 = col1 * 1.5 + np.random.normal(0, 0.01, n_filas)

    df_input = pd.DataFrame({
        'feature_A': col1,
        'feature_B': col2,
        'feature_C': col3,
        'feature_D': col4
    })

    # 4. Calcular el VIF real para el output esperado
    # El VIF se calcula sobre el DataFrame (usualmente con constante)
    X = add_constant(df_input)
    vifs = []
    for i in range(len(df_input.columns)):
        # El índice en X es i+1 porque la columna 0 es la 'const'
        vif = variance_inflation_factor(X.values, i + 1)
        vifs.append(vif)

    output_esperado = [
        df_input.columns[i]
        for i, v in enumerate(vifs) if v > umbral
    ]

    input_dict = {
        "df": df_input,
        "umbral": umbral
    }

    return input_dict, output_esperado

# --- Ejemplo de visualización ---
inp, out = generar_caso_de_uso_datos()

print(f"--- INPUT ---")

print(f"--- CONFIGURACIÓN ---")
print(f"Umbral VIF: {inp['umbral']}")

print("\n--- DATAFRAME (Primeras 5 filas) ---")
print(inp["df"].head())

print("\n--- OUTPUT ESPERADO (Variables Redundantes) ---")
print(out)
