"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta. Los
datos requeridos se encuentran en el archivo data.csv. En este laboratorio
solo puede utilizar las funciones y librerias basicas de python. No puede
utilizar pandas, numpy o scipy.
"""


def pregunta_11():
    """
    Retorne un diccionario que contengan la suma de la columna 2 para cada
    letra de la columna 4, ordenadas alfabeticamente.

    Rta/
    {'a': 122, 'b': 49, 'c': 91, 'd': 73, 'e': 86, 'f': 134, 'g': 35}


    """
    count = {}
    with open("files/input/data.csv", "r", encoding = "utf-8") as file:
        for line in file:
            column = line.split("\t")
            column4 = column[3].split(",")
            for val in column4:
                count[val] = int(column[1]) if val not in count else count[val] + int(column[1])
    return dict(sorted(count.items()))
if __name__ == "__main__":
    print(pregunta_11())