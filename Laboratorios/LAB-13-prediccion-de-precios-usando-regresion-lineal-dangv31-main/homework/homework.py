#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
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
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import pickle
import gzip
import os
import json
from glob import glob

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def pregunta_1():
    def load_data():
        train = pd.read_csv('files/input/train_data.csv.zip', index_col=False, compression='zip')
        test = pd.read_csv('files/input/test_data.csv.zip', index_col=False, compression='zip')
        return train, test

    def clean_data(df):
        df = df.copy()
        df['Age'] = 2021 - df['Year']
        df.drop(columns=['Year', 'Car_Name'], inplace=True)
        return df

    def split_data(df):
        x = df.drop(columns=['Present_Price'])
        y = df['Present_Price']
        return x, y

    def make_pipeline(x_train):
        cat_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
        num_cols = [col for col in x_train.columns if col not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), cat_cols),
                ('scaler', MinMaxScaler(), num_cols)
            ]
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('classifier', LinearRegression())
        ])

        return pipeline

    def optimize_parameters(pipeline, x_train, y_train):
        param_grid = {
            'feature_selection__k': range(1, 25),
            'classifier__fit_intercept': [True, False],
            'classifier__positive': [True, False]
        }

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=10,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            refit=True,
            verbose=1
        )

        grid.fit(x_train, y_train)
        return grid

    def clean_model_directory(path="files/models/"):
        if os.path.exists(path):
            for file in glob(f"{path}/*"):
                os.remove(file)
            os.rmdir(path)
        os.makedirs(path, exist_ok=True)

    def save_model(model, path="files/models/model.pkl.gz"):
        clean_model_directory(os.path.dirname(path))
        with gzip.open(path, "wb") as f:
            pickle.dump(model, f)

    def calculate_metrics(dataset_type, y_true, y_pred):
        return {
            "type": "metrics",
            "dataset": dataset_type,
            "r2": round(r2_score(y_true, y_pred), 4),
            "mse": round(mean_squared_error(y_true, y_pred), 4),
            "mad": round(median_absolute_error(y_true, y_pred), 4)
        }

    def save_metrics(metrics, path="files/output/metrics.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for metric in metrics:
                f.write(json.dumps(metric) + "\n")

    train_df, test_df = load_data()
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    x_train, y_train = split_data(train_df)
    x_test, y_test = split_data(test_df)

    pipeline = make_pipeline(x_train)
    estimator = optimize_parameters(pipeline, x_train, y_train)
    save_model(estimator)

    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    metrics = [
        calculate_metrics("train", y_train, y_train_pred),
        calculate_metrics("test", y_test, y_test_pred)
    ]

    save_metrics(metrics)


if __name__ == "__main__":
    pregunta_1()


