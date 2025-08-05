# 📊 Análisis de Satisfacción de Vida

Aplicación interactiva desarrollada con **Python** y **Streamlit** para el análisis estadístico, visualización de datos y predicción de la satisfacción de vida utilizando modelos de **Regresión Lineal** y **KNN**.

---

## 💻 Tecnologías Utilizadas

- **Lenguaje de programación:** Python  
- **Entorno de desarrollo:** Spyder, Google Colab  
- **Gestión de datos:** CSV (dataset_estadistica.csv), Pandas  
- **Visualización:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Interfaz web:** Streamlit  
- **Control de versiones:** GitHub  
- **Hosting / Deploy:** Streamlit Cloud  
- **Otras herramientas:** Numpy  

---

## 🗂️ Estructura del Proyecto

- **Carga y limpieza de datos**  
  - Lectura del dataset CSV.  
  - Eliminación de duplicados.  
  - Imputación de valores nulos (media para numéricos, moda para categóricos).  
  - Descarga del dataset limpio.  

- **Análisis estadístico**  
  - Estadística descriptiva (media, mediana, moda, asimetría, curtosis).  
  - Visualizaciones:  
    - Gráfico de torta (Distribución de género).  
    - Gráfico de barras (Nivel educativo).  
    - Histogramas + KDE.  
    - Boxplots.  
    - Matriz de correlación.  
    - Pairplot de relaciones.  

- **Predicción interactiva**  
  - Entrada de datos por sliders (edad, ingreso mensual, horas de estudio).  
  - Selección de modelo:  
    - Regresión Lineal.  
    - KNN Clasificador.  
  - Visualización de predicciones y comparación con datos reales.  

---

## 🧠 Lógica de Funcionamiento o Flujo Principal

1. **Inicio**: El usuario abre la app en Streamlit.  
   ![Pantalla Inicio](https://via.placeholder.com/800x400?text=Pantalla+de+Inicio)  

2. **Carga de datos**: El sistema lee `dataset_estadistica.csv`.  
   ![Vista previa dataset](https://via.placeholder.com/800x400?text=Vista+Previo+Dataset)  

3. **Limpieza**: Se eliminan duplicados y se reemplazan nulos.  

4. **Visualización inicial**: El usuario ve tablas y gráficos del dataset.  
https://proyectofinalfinalgrupo.streamlit.app/#estadistica-descriptiva

5. **Predicción**:  
   - El usuario ingresa sus datos mediante sliders.  
   - Selecciona un modelo predictivo.  
   - Obtiene el resultado y gráficas comparativas.  
   ![Interfaz de predicción](https://via.placeholder.com/800x400?text=Interfaz+Prediccion)  

6. **Descarga**: El usuario puede exportar el dataset limpio.  

---

## 🧪 Pruebas Realizadas

- **Funcionales**:  
  - Verificación de carga y limpieza del dataset.  
  - Validación de entradas y outputs en modelos predictivos.  
  - Comprobación de que las gráficas se generan correctamente.  

- **De usuario**:  
  - Simulación de diferentes valores para evaluar coherencia en predicciones.  
  - Validación de usabilidad en la interfaz.  

- **De rendimiento**:  
  - Test con datasets de diferentes tamaños para medir estabilidad.  

---

## 🧾 Conclusiones del Grupo

- **Aprendizajes**:  
  - Uso de Pandas y Numpy para procesamiento de datos.  
  - Creación de dashboards interactivos con Streamlit.  
  - Implementación de modelos de Machine Learning básicos.  

- **Dificultades superadas**:  
  - Manejo de valores nulos y codificación de variables categóricas.  
  - Sincronización de columnas en datos de predicción y entrenamiento.  

- **Mejoras futuras**:  
  - Añadir más modelos predictivos.  
  - Permitir carga de datasets personalizados.  
  - Mejorar visualización para dispositivos móviles.  

---

## 🚀 Ejecución

Para ejecutar el proyecto en local:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run app.py
