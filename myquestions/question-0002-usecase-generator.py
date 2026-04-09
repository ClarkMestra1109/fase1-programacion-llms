import pandas as pd
import numpy as np

def generar_caso_de_uso_crear_datos():
    """
    Genera un caso de uso aleatorio para la función codificar_ciclo_temporal.
    Retorna:
        input_dict (dict): Diccionario con el DataFrame y el nombre de la columna.
        output_esperado (pd.DataFrame): DataFrame transformado con hora_sin y hora_cos.
    """
    # 1. Configuración aleatoria
    n_filas = np.random.randint(5, 12)
    nombre_col = np.random.choice(['hora_dia', 'hour', 'momento_registro'])

    # 2. Crear el DataFrame de entrada con horas aleatorias (0-23)
    horas = np.random.randint(0, 24, n_filas)
    df_input = pd.DataFrame({nombre_col: horas})

    # Guardamos una copia para el input antes de transformar
    input_dict = {
        "df": df_input.copy(),
        "col_hora": nombre_col
    }

    # 3. Calcular el output esperado
    # La fórmula es: sin(2 * pi * x / max_val)
    df_output = df_input.copy()
    max_periodo = 24

    df_output['hora_sin'] = np.sin(2 * np.pi * df_output[nombre_col] / max_periodo)
    df_output['hora_cos'] = np.cos(2 * np.pi * df_output[nombre_col] / max_periodo)

    # Eliminamos la columna original según el requerimiento
    df_output = df_output.drop(columns=[nombre_col])

    return input_dict, df_output

# --- Demostración del caso de uso ---
inp, out = generar_caso_de_uso_crear_datos()

print("--- INPUT (DataFrame Original) ---")
print(inp["df"])
print(f"\nColumna a transformar: {inp['col_hora']}")

print("\n--- OUTPUT ESPERADO (Codificación Cíclica) ---")
print(out)

# Verificación rápida de la naturaleza cíclica:
# Si graficáramos sin vs cos, obtendríamos puntos en un círculo unitario.
