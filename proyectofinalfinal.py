# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 08:35:31 2025

@author: zegar
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Análisis de Datos", layout="wide")

st.title("Análisis de Datos Estadísticos")

# Cargar datos
dataset_path = st.text_input("Ruta del archivo CSV:", "dataset_estadistica.csv")

if dataset_path:
    try:
        dataset = pd.read_csv(dataset_path)

        st.subheader("📊 Información Original del Dataset")
        st.write(dataset.info())
        st.write(dataset.head())

        # Eliminar duplicados
        dataset = dataset.drop_duplicates()

        # Imputación
        numeric_cols = ['Edad','Ingreso_Mensual', 'Horas_Estudio_Semanal']
        categorical_cols = ['Nivel_Educativo','Genero']

        numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

        categorical_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

        st.subheader("✅ Datos Limpios")
        st.write(dataset.info())
        st.write(dataset.head())

        # Estadística descriptiva
        st.subheader("📈 Estadística Descriptiva")
        st.write(dataset.describe())

        st.write("**Media**")
        st.write(dataset.mean(numeric_only=True))

        st.write("**Mediana**")
        st.write(dataset.median(numeric_only=True))

        st.write("**Moda**")
        st.write(dataset.mode(numeric_only=True))

        st.write("**Asimetría (Skewness)**")
        st.write(dataset.skew(numeric_only=True))

        st.write("**Curtosis (Kurtosis)**")
        st.write(dataset.kurtosis(numeric_only=True))

        # Visualización
        st.subheader("📉 Gráficos")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Distribución de Edad")
            fig, ax = plt.subplots()
            sns.histplot(dataset["Edad"], kde=True, ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("Ingreso Mensual")
            fig, ax = plt.subplots()
            sns.boxplot(x=dataset["Ingreso_Mensual"], ax=ax)
            st.pyplot(fig)

        st.info("En resumen, las distribuciones de Edad y Satisfacción de Vida son simétricas. Ingreso Mensual y Horas de Estudio presentan asimetría positiva.")

    except Exception as e:
        st.error(f"Error al cargar el dataset: {e}")
