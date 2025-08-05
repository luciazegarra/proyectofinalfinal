# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 08:35:31 2025

@author: zegar
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Configuración inicial
st.set_page_config(page_title="Análisis Estadístico", layout="centered")
st.title("📊 Análisis de Satisfacción de Vida")

# 2. Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_estadistica.csv")

try:
    dataset = cargar_datos()

    st.subheader("📊 Información Original del Dataset")
    st.write(dataset.head())

    # Eliminar duplicados
    dataset = dataset.drop_duplicates()

    # Imputación
    numeric_cols = ['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal']
    categorical_cols = ['Nivel_Educativo', 'Genero']

    numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

    categorical_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

    st.subheader("✅ Datos Limpios")
    st.write(dataset.head())

    # Estadística descriptiva
    st.subheader("📈 Estadística Descriptiva")
    st.write("**Resumen:**")
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

    st.subheader("📊 Visualización de Distribuciones y Relaciones")

    # Conteo de género - Gráfico de torta
    st.markdown("### 🎂 Distribución de Género")
    gender_counts = dataset['Genero'].value_counts()
    
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.pie(
        gender_counts,
        labels=gender_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('pastel')
    )
    ax1.set_title('Distribución de Género')
    ax1.axis('equal')  # Círculo perfecto
    st.pyplot(fig1)
    
    st.info("Según el análisis, la distribución de la variable Género está notablemente equilibrada entre las categorías Masculino, Femenino y Otro.")
    
    # Gráfico de barras - Nivel Educativo
    st.markdown("### 🎓 Distribución de Nivel Educativo")
    nivel_educativo_percentages = dataset['Nivel_Educativo'].value_counts(normalize=True) * 100
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=nivel_educativo_percentages.index,
        y=nivel_educativo_percentages.values,
        palette='viridis',
        ax=ax2
    )
    ax2.set_title('Distribución de Nivel Educativo')
    ax2.set_xlabel('Nivel Educativo')
    ax2.set_ylabel('Porcentaje (%)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Añadir etiquetas
    for p in ax2.patches:
        percentage = f'{p.get_height():.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax2.annotate(percentage, (x, y), ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.info("El 64% de la población cuenta con educación superior (técnica, universitaria o posgrado), mientras que el 36% restante corresponde a niveles de educación secundaria y primaria.")
    
    # Histogramas con KDE
    st.markdown("### 📈 Distribuciones de Variables Numéricas")
    
    ds_num = dataset.select_dtypes(include=['float64', 'int64'])
    
    for columna in ds_num.columns:
        st.write(f"Distribución de **{columna}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(dataset[columna], kde=True, color='skyblue', ax=ax)
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de {columna}")
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Boxplots
    st.markdown("### 📦 Boxplots de Variables Numéricas")
    
    for columna in ds_num.columns:
        st.write(f"Boxplot de **{columna}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=dataset[columna], color='lightgreen', ax=ax)
        ax.set_title(f"Boxplot de {columna}")
        ax.set_xlabel(columna)
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'dataset_estadistica.csv'.")
except Exception as e:
    st.error(f"⚠️ Ocurrió un error inesperado: {e}")