"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_10():
    """
    Retorne una lista de tuplas contengan por cada tupla, la letra de la
    columna 1 y la cantidad de elementos de las columnas 4 y 5.

    Rta/
    [('E', 3, 5),
     ('A', 3, 4),
     ...
     ('E', 2, 3),
     ('E', 3, 3)]


    """
    count = []
    with open("files/input/data.csv", "r", encoding="utf-8") as file:
        for line in file:
            column = line.split("\t")
            column4 = len(column[3].split(","))
            column5 = len(column[4].split(","))
            count.append((column[0], column4, column5))
    return count
if __name__ == "__main__":
    print(pregunta_10())
