# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix

import json
import gzip
import pickle
import os


def pregunta_1():
    def load_data(path):
        return pd.read_csv(path, index_col=False, compression="zip")
    def clean_data(df):
        df = df.rename(columns={"default payment next month": "default"})
        df = df.drop(columns=["ID"])
        df = df.dropna()
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
        return df
    def create_pipeline():
        cat_features = ["SEX", "EDUCATION", "MARRIAGE"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ],
            remainder="passthrough"
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        return pipeline
    def optimize_pipeline(pipeline, x_train, y_train):
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
        }
        scorer = make_scorer(balanced_accuracy_score)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scorer,
            cv=10,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(x_train, y_train)
        return grid_search
    def save_model(model, output_path="files/models/model.pkl.gz"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with gzip.open(output_path, "wb") as f:
            pickle.dump(model, f)
    
    def evaluate_and_save_metrics(model, x_train, y_train, x_test, y_test, output_path="files/output/metrics.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        metrics = []
        for dataset_name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            y_pred = model.predict(x)
            result = {
                "type": "metrics",  # ← AÑADIDO
                "dataset": dataset_name,
                "precision": precision_score(y, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1_score": f1_score(y, y_pred)
            }
            metrics.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")


    def add_confusion_matrices(model, x_train, y_train, x_test, y_test, output_path="files/output/metrics.json"):
        with open(output_path, "r", encoding="utf-8") as f:
            metrics = [json.loads(line) for line in f]

        for dataset_name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            y_pred = model.predict(x)
            cm = confusion_matrix(y, y_pred, labels=[0, 1])
            cm_dict = {
                "type": "cm_matrix",
                "dataset": dataset_name,
                "true_0": {
                    "predicted_0": int(cm[0][0]),
                    "predicted_1": int(cm[0][1])
                },
                "true_1": {
                    "predicted_0": int(cm[1][0]),
                    "predicted_1": int(cm[1][1])
                }
            }
            metrics.append(cm_dict)
        with open(output_path, "w", encoding="utf-8") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")


    input_path = "files/input/"
    train = load_data(input_path + "train_data.csv.zip")
    test = load_data(input_path + "test_data.csv.zip")
    train = clean_data(train)
    test = clean_data(test)

    x_test = test.drop(columns=["default"])
    y_test = test["default"]

    x_train = train.drop(columns=["default"])
    y_train = train["default"]

    pipeline = create_pipeline()
    estimator = optimize_pipeline(pipeline, x_train, y_train)

    save_model(estimator)
    evaluate_and_save_metrics(estimator, x_train, y_train, x_test, y_test)
    add_confusion_matrices(estimator, x_train, y_train, x_test, y_test)

    

if __name__ == "__main__":
    pregunta_1()