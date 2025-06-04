"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_05():
    """
    Retorne una lista de tuplas con el valor maximo y minimo de la columna 2
    por cada letra de la columa 1.

    Rta/
    [('A', 9, 2), ('B', 9, 1), ('C', 9, 0), ('D', 8, 3), ('E', 9, 1)]

    """
    maxMin = {}
    with open("files/input/data.csv", "r", encoding="utf-8") as file:
        for line in file:
            column = line.split("\t")
            if column[0] not in maxMin:
                maxMin[column[0]] = [int(column[1]), int(column[1])]
            else:
                maxMin[column[0][0]] = [max(maxMin[column[0][0]][0], int(column[1])),
                                        min(maxMin[column[0][0]][1], int(column[1]))]
    return sorted([(key, value[0], value[1]) for key, value in maxMin.items()])

if __name__ == "__main__":
    print(pregunta_05())