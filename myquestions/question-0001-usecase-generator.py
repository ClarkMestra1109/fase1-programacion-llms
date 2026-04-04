import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio.
    El 'input' contiene DataFrames reales listos para usar.
    """
    # Configuraciones aleatorias
    n_filas = np.random.randint(5, 10) # Pocas filas para que sea legible al imprimir
    cols = ['ventas', 'clientes', 'temperatura']
    alpha = 0.05

    # 1. Crear datos de entrenamiento (Base)
    df_entrenamiento = pd.DataFrame({
        col: np.random.uniform(10, 100, n_filas).round(2) for col in cols
    })

    # 2. Crear datos actuales (con posibilidad de drift)
    df_actual = df_entrenamiento.copy()
    cols_con_drift = []

    for col in cols:
        # Decisión aleatoria: ¿esta columna tendrá drift?
        if np.random.choice([True, False]):
            # Aplicamos un cambio drástico (sumar 500) para asegurar que el test lo detecte
            df_actual[col] = (df_actual[col] + 500).round(2)
            cols_con_drift.append(col)
        else:
            # Cambio mínimo que no debería activar el drift
            df_actual[col] = (df_actual[col] + np.random.normal(0, 1, n_filas)).round(2)

    # 3. Calcular el output esperado usando la lógica de KS
    # (Lo calculamos aquí para que el generador sea "el maestro" que sabe la verdad)
    output_real = []
    for col in cols:
        _, p_value = ks_2samp(df_entrenamiento[col], df_actual[col])
        if p_value < alpha:
            output_real.append(col)

    # El input es un diccionario que CONTIENE los objetos DataFrame
    input_dict = {
        "df_entrenamiento": df_entrenamiento,
        "df_actual": df_actual,
        "alpha": alpha
    }

    return input_dict, output_real

# --- Demostración de cómo se ve el input ---
inp, out = generar_caso_de_uso_preparar_datos()

print("--- INPUT: DATAFRAME ENTRENAMIENTO ---")
print(inp["df_entrenamiento"])
print("\n--- INPUT: DATAFRAME ACTUAL ---")
print(inp["df_actual"])
print(f"\nAlpha: {inp['alpha']}")
print(f"\n--- OUTPUT ESPERADO (Columnas con Drift) ---")
print(out)