"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""

# pylint: disable=import-outside-toplevel
import pandas as pd
import re


def pregunta_01():
    """
    Construya y retorne un dataframe de Pandas a partir del archivo
    'files/input/clusters_report.txt'. Los requierimientos son los siguientes:

    - El dataframe tiene la misma estructura que el archivo original.
    - Los nombres de las columnas deben ser en minusculas, reemplazando los
      espacios por guiones bajos.
    - Las palabras clave deben estar separadas por coma y con un solo
      espacio entre palabra y palabra.


    """
    with open("files/input/clusters_report.txt", "r", encoding="utf-8") as f:
      lines = f.readlines()

    lines = lines[4:]  # Saltar encabezado

    data = []
    cluster = None
    cantidad = None
    porcentaje = None
    palabras = ''

    for line in lines:
        line = line.rstrip()
        line_strip = line.lstrip()

        if line_strip and line_strip[0].isdigit():
            # Guardar la fila anterior
            if cluster is not None:
                palabras = ' '.join(palabras.split())
                palabras = palabras.replace(' ,', ',').replace(',', ', ')
                palabras = ' '.join(palabras.split())  # limpiar dobles espacios otra vez
                palabras = palabras.rstrip('.')  # quitar punto final
                data.append([cluster, cantidad, porcentaje, palabras])

            partes = line_strip.split()
            cluster = int(partes[0])
            cantidad = int(partes[1])

            # Detectar si el símbolo % está separado
            if partes[3] == '%':
                porcentaje = float(partes[2].replace(',', '.'))
                palabras = ' '.join(partes[4:])
            else:
                porcentaje = float(partes[2].replace(',', '.').replace('%', ''))
                palabras = ' '.join(partes[3:])
        else:
            palabras += ' ' + line_strip

    # Guardar la última fila
    if cluster is not None:
        palabras = ' '.join(palabras.split())
        palabras = palabras.replace(' ,', ',').replace(',', ', ')
        palabras = ' '.join(palabras.split())
        palabras = palabras.rstrip('.')
        data.append([cluster, cantidad, porcentaje, palabras])

    # Crear DataFrame
    df = pd.DataFrame(data, columns=[
        'cluster',
        'cantidad_de_palabras_clave',
        'porcentaje_de_palabras_clave',
        'principales_palabras_clave'
    ])

    return df
df = pregunta_01()
asd = df.principales_palabras_clave.to_list()[0]
print(asd)

