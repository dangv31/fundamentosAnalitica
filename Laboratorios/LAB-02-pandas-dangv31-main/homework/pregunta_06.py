import pandas as pd
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en los archivos `tbl0.tsv`, `tbl1.tsv` y 
`tbl2.tsv`. En este laboratorio solo puede utilizar las funciones y 
librerias de pandas para resolver las preguntas.
"""


def pregunta_06():
    """
    Retorne una lista con los valores unicos de la columna `c4` del archivo
    `tbl1.csv` en mayusculas y ordenados alfabéticamente.

    Rta/
    ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    """
    df = pd.read_csv('files/input/tbl1.tsv', sep='\t')
    valores_unicos_c4 = df['c4'].unique().tolist()
    valores_unicos_c4 = [valor.upper() for valor in valores_unicos_c4]
    valores_unicos_c4.sort()
    return valores_unicos_c4
if __name__ == "__main__":
    print(pregunta_06())
