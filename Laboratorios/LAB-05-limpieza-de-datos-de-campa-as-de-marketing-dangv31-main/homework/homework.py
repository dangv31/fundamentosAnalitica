"""
Escriba el codigo que ejecute la accion solicitada.
"""

# pylint: disable=import-outside-toplevel

import zipfile
import glob
import pandas as pd
import os
from datetime import datetime

def clean_campaign_data():
    """
    En esta tarea se le pide que limpie los datos de una campaña de
    marketing realizada por un banco, la cual tiene como fin la
    recolección de datos de clientes para ofrecerls un préstamo.

    La información recolectada se encuentra en la carpeta
    files/input/ en varios archivos csv.zip comprimidos para ahorrar
    espacio en disco.

    Usted debe procesar directamente los archivos comprimidos (sin
    descomprimirlos). Se desea partir la data en tres archivos csv
    (sin comprimir): client.csv, campaign.csv y economics.csv.
    Cada archivo debe tener las columnas indicadas.

    Los tres archivos generados se almacenarán en la carpeta files/output/.

    client.csv:
    - client_id
    - age
    - job: se debe cambiar el "." por "" y el "-" por "_"
    - marital
    - education: se debe cambiar "." por "_" y "unknown" por pd.NA
    - credit_default: convertir a "yes" a 1 y cualquier otro valor a 0
    - mortage: convertir a "yes" a 1 y cualquier otro valor a 0

    campaign.csv:
    - client_id
    - number_contacts
    - contact_duration
    - previous_campaing_contacts
    - previous_outcome: cmabiar "success" por 1, y cualquier otro valor a 0
    - campaign_outcome: cambiar "yes" por 1 y cualquier otro valor a 0
    - last_contact_day: crear un valor con el formato "YYYY-MM-DD",
        combinando los campos "day" y "month" con el año 2022.

    economics.csv:
    - client_id
    - const_price_idx
    - eurobor_three_months



    """
    output_path = "files/output/"
    zip_folder = "files/input/*"

    dataframes = []
    
    for zip_path in glob.glob(zip_folder):
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_name in z.namelist():
                with z.open(file_name) as f:
                    df = pd.read_csv(f)
                    dataframes.append(df)
    full_df = pd.concat(dataframes, ignore_index=True)

    # Client.csv
    client_df = full_df[[ "client_id", "age", "job", "marital", "education", "credit_default", "mortgage"]].copy()
    client_df["job"] = client_df["job"].str.replace(".", "").str.replace("-", "_")
    client_df["education"] = client_df["education"].str.replace(".", "_")
    client_df["education"] = client_df["education"].replace("unknown", pd.NA)
    client_df["credit_default"] = (client_df["credit_default"] == "yes").astype(int)
    client_df["mortgage"] = (client_df["mortgage"] == "yes").astype(int)

    # Campaign.csv
    campaign_df = full_df[["client_id", "number_contacts", "contact_duration","previous_campaign_contacts", "previous_outcome","campaign_outcome", "day", "month"]].copy()

    campaign_df["previous_outcome"] = (campaign_df["previous_outcome"] == "success").astype(int)
    campaign_df["campaign_outcome"] = (campaign_df["campaign_outcome"] == "yes").astype(int)

    campaign_df["last_contact_date"] = campaign_df.apply(lambda row: datetime.strptime(f"2022-{row['month']}-{int(row['day'])}", "%Y-%b-%d").strftime("%Y-%m-%d"), axis=1)

    campaign_df.drop(columns=["day", "month"], inplace=True)

    # Economics.csv
    economics_df = full_df[["client_id", "cons_price_idx", "euribor_three_months"]].copy()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    client_df.to_csv(os.path.join(output_path, "client.csv"), index=False)
    campaign_df.to_csv(os.path.join(output_path, "campaign.csv"), index=False)
    economics_df.to_csv(os.path.join(output_path, "economics.csv"), index=False)


if __name__ == "__main__":
    clean_campaign_data()
