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

    # 3. Limpieza de datos
    dataset = dataset.drop_duplicates()

    numeric_cols = ['Edad', 'Ingreso_Mensual', 'Horas_Estudio_Semanal']
    categorical_cols = ['Nivel_Educativo', 'Genero']

    numeric_imputer = SimpleImputer(strategy="mean")
    dataset[numeric_cols] = numeric_imputer.fit_transform(dataset[numeric_cols])

    categorical_imputer = SimpleImputer(strategy="most_frequent")
    dataset[categorical_cols] = categorical_imputer.fit_transform(dataset[categorical_cols])

    st.subheader("✅ Datos Limpios")
    st.write(dataset.head())
    
    st.download_button(
    label="📥 Descargar base limpia (sin codificar)",
    data=dataset.to_csv(index=False),
    file_name="dataset_limpio.csv",
    mime="text/csv"
)

    # 4. Estadística descriptiva
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

    #5. VISUALIZACIÓN DE DISTRIBUCIONES Y RELACIONES
    st.subheader("📊 Visualización de Distribuciones y Relaciones")

    # 🎂 Distribución de Género
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
    ax1.axis('equal')
    st.pyplot(fig1)
    st.info("📝 **Conclusión:** La distribución de género es bastante equilibrada, aunque puede observarse una ligera predominancia de alguno de los grupos según el caso. Este equilibrio permite un análisis representativo de la población estudiada.")

    # 🎓 Distribución de Nivel Educativo
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
    for p in ax2.patches:
        percentage = f'{p.get_height():.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax2.annotate(percentage, (x, y), ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(fig2)
    st.info("📝 **Conclusión:** Una mayoría significativa de la población cuenta con educación superior. Esto puede influir en variables como el ingreso mensual o la satisfacción de vida, destacando la importancia de la formación académica.")

    # 📈 Histogramas con KDE
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

        skewness = dataset[columna].skew()
        if skewness > 0.5:
            forma = "asimétrica positiva (cola hacia la derecha)"
        elif skewness < -0.5:
            forma = "asimétrica negativa (cola hacia la izquierda)"
        else:
            forma = "aproximadamente simétrica"
        st.info(f"📝 **Conclusión:** La distribución de **{columna}** es {forma}.")

    # 📦 Boxplots
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

        st.info(f"📝 **Conclusión:** El boxplot de **{columna}** permite visualizar la presencia de posibles valores atípicos y la dispersión de los datos.")

except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'dataset_estadistica.csv'.")
except Exception as e:
    st.error(f"⚠️ Ocurrió un error inesperado: {e}")
    

# ---------------------------------------------
# 🔁 Selección de modelo y predicción interactiva
# ---------------------------------------------
st.subheader("🧠 Predicción Interactiva con Selección de Modelo")

# Entradas del usuario
edad = st.slider("Edad", int(dataset["Edad"].min()), int(dataset["Edad"].max()), int(dataset["Edad"].mean()))
ingreso = st.slider("Ingreso Mensual", int(dataset["Ingreso_Mensual"].min()), int(dataset["Ingreso_Mensual"].max()), int(dataset["Ingreso_Mensual"].mean()))
horas_estudio = st.slider("Horas de Estudio Semanal", 0, 80, int(dataset["Horas_Estudio_Semanal"].mean()))

# Elegir modelo
modelo_seleccionado = st.selectbox("📌 Selecciona el modelo de predicción", ["Regresión Lineal", "KNN Clasificador"])

# ----------------------------------
# Preprocesamiento de entrada
# ----------------------------------
# Datos originales codificados
x = dataset.drop(['ID_Persona', 'Satisfaccion_Vida'], axis=1)
x = pd.get_dummies(x, drop_first=True)
x_columns = x.columns
y = dataset['Satisfaccion_Vida']

# Escalar datos
scaler = StandardScaler()
x_escalado = scaler.fit_transform(x)

# División de datos
x_train_escalado, x_test_escalado, y_train, y_test = train_test_split(
    x_escalado, y, test_size=0.2, random_state=42
)

# Crear input_df para predicción
input_data = pd.DataFrame({
    'Edad': [edad],
    'Ingreso_Mensual': [ingreso],
    'Horas_Estudio_Semanal': [horas_estudio],
    'Nivel_Educativo': [dataset['Nivel_Educativo'].mode()[0]],  # Valor más frecuente para columnas faltantes
    'Genero': [dataset['Genero'].mode()[0]]
})
input_encoded = pd.get_dummies(input_data)
for col in x_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[x_columns]
input_scaled = scaler.transform(input_encoded)

# ----------------------------------
# Regresión Lineal
# ----------------------------------
if modelo_seleccionado == "Regresión Lineal":
    modelo_rl = LinearRegression()
    modelo_rl.fit(x_train_escalado, y_train)
    pred = modelo_rl.predict(input_scaled)[0]
    st.success(f"🔮 Predicción de satisfacción de vida (Regresión): **{pred:.2f}**")

# ----------------------------------
# KNN Clasificador
# ----------------------------------
elif modelo_seleccionado == "KNN Clasificador":
    modelo_knn = KNeighborsClassifier(n_neighbors=3)
    modelo_knn.fit(x_train_escalado, y_train)
    pred = modelo_knn.predict(input_scaled)[0]
    st.success(f"🔮 Predicción de satisfacción de vida (KNN): **{pred}**")

    # Visualización 2D (Edad vs Ingreso)
    st.markdown("### 📊 Visualización KNN - Edad vs Ingreso Mensual")
    fig_knn, ax_knn = plt.subplots(figsize=(8, 6))
    scatter = ax_knn.scatter(
        dataset['Edad'], dataset['Ingreso_Mensual'],
        c=dataset['Satisfaccion_Vida'],
        cmap='viridis', edgecolor='k', alpha=0.7
    )
    ax_knn.scatter(
        edad, ingreso,
        color="red", marker='X', s=120,
        label="Nuevo dato ingresado"
    )
    cbar = plt.colorbar(scatter, ax=ax_knn)
    cbar.set_label("Satisfacción de Vida")
    ax_knn.set_xlabel("Edad")
    ax_knn.set_ylabel("Ingreso Mensual")
    ax_knn.set_title(f"KNN - Predicción: {pred}")
    ax_knn.legend()
    ax_knn.grid(True)
    st.pyplot(fig_knn)
