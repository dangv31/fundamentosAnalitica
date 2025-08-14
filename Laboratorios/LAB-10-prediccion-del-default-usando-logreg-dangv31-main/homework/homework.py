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
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
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
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
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
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
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
import pickle
import gzip
import os
import json

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score, confusion_matrix




def pregunta_1():
    def load_data(path):
        return pd.read_csv(path, index_col=False, compression="zip")
    
    def clean_data(df):
        df = df.rename(columns={"default payment next month": "default"})
        df.drop(columns=["ID"], inplace=True, errors='ignore')
        df = df.loc[df["MARRIAGE"] != 0]
        df = df.loc[df["EDUCATION"] != 0]
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [1, 2, 3] else 4)
        return df

    def make_pipeline(k=23):
        cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]   
        transformer = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
            remainder="passthrough",
        )
        pipeline = Pipeline([
            ('preprocessor', transformer),
            ('scaler', MinMaxScaler()),
            ('feature_selection', SelectKBest(score_func=f_classif, k=k)),
            ('classifier', LogisticRegression(max_iter=500, random_state=42))
        ])
        return pipeline
    
    def pipeline_optimizer(pipeline, x_train, y_train):
        param_grid = {
            'feature_selection__k': range(1, 11),
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear'],
            "classifier__max_iter": [100, 200]
        }
        grid_search =  GridSearchCV(
            pipeline,
            param_grid,
            cv=10,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        grid_search.fit(x_train, y_train)
        return grid_search


    def save_data(modelo, ruta="files/models/model.pkl.gz"):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with gzip.open(ruta, "wb") as f:
            pickle.dump(modelo, f)

    def eval_metrics(y_train_true, y_test_true, y_train_pred, y_test_pred):
        def metrics_dict(y_true, y_pred, dataset):
            return {
                "type": "metrics",
                "dataset": dataset,
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
                "recall": round(recall_score(y_true, y_pred), 4),
                "f1_score": round(f1_score(y_true, y_pred), 4),
            }

        metrics_train = metrics_dict(y_train_true, y_train_pred, "train")
        metrics_test = metrics_dict(y_test_true, y_test_pred, "test")
        
        return [metrics_train, metrics_test]

    def save_metrics(metricas, ruta="files/output/metrics.json", append=False):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        mode = "a" if append else "w"
        with open(ruta, mode) as f:
            for m in metricas:
                f.write(json.dumps(m) + "\n")

    def confusion_matrix_json(y_true, y_pred, dataset):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            "type": "cm_matrix",
            "dataset": dataset,
            "true_0": {
                "predicted_0": int(cm[0][0]),
                "predicted_1": int(cm[0][1]),
            },
            "true_1": {
                "predicted_0": int(cm[1][0]),
                "predicted_1": int(cm[1][1]),
            },
        }

    train = load_data("files/input/train_data.csv.zip")
    test = load_data("files/input/test_data.csv.zip")

    train = clean_data(train)
    test = clean_data(test)

    x_test = test.drop(columns=["default"])
    y_test = test["default"]

    x_train = train.drop(columns=["default"])
    y_train = train["default"]

    pipeline = make_pipeline(k=23)
    estimator = pipeline_optimizer(pipeline, x_train, y_train)

    save_data(estimator)

    y_train_pred = estimator.best_estimator_.predict(x_train)
    y_test_pred = estimator.best_estimator_.predict(x_test)

    metricas = eval_metrics(y_train, y_test, y_train_pred, y_test_pred)
    save_metrics(metricas)

    cm_train = confusion_matrix_json(y_train, y_train_pred, "train")
    cm_test = confusion_matrix_json(y_test, y_test_pred, "test")
    save_metrics([cm_train, cm_test], append=True)


if __name__ == "__main__":
    pregunta_1()