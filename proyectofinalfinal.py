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
from sklearn.metrics import r2_score, mean_squared_error
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
    st.info("📝 **la base original tiene 1010 datos donde las se eliminaron 10 duplicados y se reemplazaron los datos faltantes por valores promedio y valores mas repetidos:** .")
   
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

    # 5. VISUALIZACIÓN DE DISTRIBUCIONES Y RELACIONES
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
        ax2.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha='center', va='bottom')
    st.pyplot(fig2)
    
    st.info("📝 **Conclusión:** Una mayoría significativa de la población cuenta con educación superior. Esto puede influir en variables como el ingreso mensual o la satisfacción de vida, destacando la importancia de la formación académica.")
    
    # 📈 Histogramas con KDE
    st.markdown("### 📈 Distribuciones de Variables Numéricas")
    ds_num = dataset.select_dtypes(include=['float64', 'int64'])

    for columna in ds_num.columns:
        st.write(f"Distribución de **{columna}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(dataset[columna], kde=True, color='skyblue', ax=ax)
        ax.set_title(f"Distribución de {columna}")
        ax.grid(True)
        st.pyplot(fig)
        st.info(f"📝 **Conclusión:** La distribución de **{columna}** es {forma}.")
    # 📦 Boxplots
    st.markdown("### 📦 Boxplots de Variables Numéricas")
    for columna in ds_num.columns:
        st.write(f"Boxplot de **{columna}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=dataset[columna], color='lightgreen', ax=ax)
        ax.set_title(f"Boxplot de {columna}")
        ax.grid(True)
        st.pyplot(fig)
        st.info(f"📝 **Conclusión:** El boxplot de **{columna}** permite visualizar la presencia de posibles valores atípicos y la dispersión de los datos.")

    # 🧊 Matriz de Correlación
    st.markdown("### 🧊 Matriz de Correlación")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(dataset.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Matriz de Correlación")
    st.pyplot(fig_corr)

    # 🔗 Pairplot
    st.markdown("### 🔗 Relaciones entre Variables Numéricas")
    try:
        pairplot_data = dataset[numeric_cols + ['Satisfaccion_Vida']].dropna().sample(n=200, random_state=1)
        pairplot_fig = sns.pairplot(pairplot_data)
        st.pyplot(pairplot_fig.figure)
    except Exception as e:
        st.warning(f"No se pudo generar el pairplot: {e}")

    # ---------------------------------------------
    # 🔁 Predicción Interactiva con Modelos
    # ---------------------------------------------
    st.subheader("🧠 Predicción Interactiva con Selección de Modelo")

    # Entradas del usuario
    edad = st.slider("Edad", int(dataset["Edad"].min()), int(dataset["Edad"].max()), int(dataset["Edad"].mean()))
    ingreso = st.slider("Ingreso Mensual", int(dataset["Ingreso_Mensual"].min()), int(dataset["Ingreso_Mensual"].max()), int(dataset["Ingreso_Mensual"].mean()))
    horas_estudio = st.slider("Horas de Estudio Semanal", 0, 80, int(dataset["Horas_Estudio_Semanal"].mean()))

    modelo_seleccionado = st.selectbox("📌 Selecciona el modelo de predicción", ["Regresión Lineal", "KNN Clasificador"])

    # Preprocesamiento
    x = dataset.drop(['ID_Persona', 'Satisfaccion_Vida'], axis=1)
    x = pd.get_dummies(x, drop_first=True)
    x_columns = x.columns
    y = dataset['Satisfaccion_Vida']

    scaler = StandardScaler()
    x_escalado = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_escalado, y, test_size=0.2, random_state=42)

    # Input para predicción
    input_data = pd.DataFrame({
        'Edad': [edad],
        'Ingreso_Mensual': [ingreso],
        'Horas_Estudio_Semanal': [horas_estudio],
        'Nivel_Educativo': [dataset['Nivel_Educativo'].mode()[0]],
        'Genero': [dataset['Genero'].mode()[0]]
    })
    input_encoded = pd.get_dummies(input_data)
    for col in x_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[x_columns]
    input_scaled = scaler.transform(input_encoded)

    # 🔹 Modelo Regresión
    if modelo_seleccionado == "Regresión Lineal":
        modelo = LinearRegression()
        modelo.fit(x_train, y_train)
        y_pred = modelo.predict(x_test)
        pred = modelo.predict(input_scaled)[0]

        st.success(f"🔮 Predicción (Regresión): **{pred:.2f}**")
        st.write("**Intercepto del modelo:**", round(modelo.intercept_, 2))
        st.success(f"🔹 R² Score (conjunto de prueba): {r2_score(y_test, y_pred):.4f}")
        st.info(f"🔸 MSE: {mean_squared_error(y_test, y_pred):.4f}")
        st.info("El valor de correlación multiple de Pearson es bastante bajo, por lo que se dice que las variables no tienen realación entre si")

        # Comparación real vs predicho
        st.markdown("### 📉 Comparación: Predicción vs Valores Reales")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(y_test, y_pred, color='green', alpha=0.6)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        ax3.set_xlabel("Valores reales")
        ax3.set_ylabel("Predicción")
        ax3.grid(True)
        st.pyplot(fig3)

    # 🔹 Modelo KNN
    elif modelo_seleccionado == "KNN Clasificador":
        modelo_knn = KNeighborsClassifier(n_neighbors=3)
        modelo_knn.fit(x_train, y_train)
        pred = modelo_knn.predict(input_scaled)[0]
        st.success(f"🔮 Predicción (KNN): **{pred}**")

        # Visualización 2D
        st.markdown("### 📊 Visualización KNN - Edad vs Ingreso Mensual")
        fig_knn, ax_knn = plt.subplots(figsize=(8, 6))
        scatter = ax_knn.scatter(
            dataset['Edad'], dataset['Ingreso_Mensual'],
            c=dataset['Satisfaccion_Vida'],
            cmap='viridis', edgecolor='k', alpha=0.7
        )
        ax_knn.scatter(edad, ingreso, color="red", marker='X', s=120, label="Nuevo dato")
        plt.colorbar(scatter, ax=ax_knn, label="Satisfacción de Vida")
        ax_knn.set_xlabel("Edad")
        ax_knn.set_ylabel("Ingreso Mensual")
        ax_knn.legend()
        st.pyplot(fig_knn)
        st.info("🔸 Conclusión: ")

except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'dataset_estadistica.csv'.")
except Exception as e:
    st.error(f"⚠️ Error inesperado: {e}")
