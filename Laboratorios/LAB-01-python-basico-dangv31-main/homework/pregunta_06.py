"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_06():
    """
    La columna 5 codifica un diccionario donde cada cadena de tres letras
    corresponde a una clave y el valor despues del caracter `:` corresponde al
    valor asociado a la clave. Por cada clave, obtenga el valor asociado mas
    pequeño y el valor asociado mas grande computados sobre todo el archivo.

    Rta/
    [('aaa', 1, 9),
     ('bbb', 1, 9),
     ('ccc', 1, 10),
     ('ddd', 0, 9),
     ('eee', 1, 7),
     ('fff', 0, 9),
     ('ggg', 3, 10),
     ('hhh', 0, 9),
     ('iii', 0, 9),
     ('jjj', 5, 17)]

    """
    minMax = {}
    with open("files/input/data.csv", "r", encoding="utf-8") as file:
        for line in file:
            column = line.split("\t")
            column = column[4].split(",")
            for val in column: 
                key, value = val.split(":")
                if key not in minMax:
                    minMax[key] = [int(value), int(value)]
                else:
                    minMax[key][0] = min(minMax[key][0], int(value))
                    minMax[key][1] = max(minMax[key][1], int(value))
    return sorted([(key, value[0], value[1]) for key, value in minMax.items()])

if __name__ == "__main__":
    print(pregunta_06())