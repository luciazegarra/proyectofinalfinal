# 📊 Análisis de Satisfacción de Vida

Aplicación interactiva desarrollada con **Python** y **Streamlit** para el análisis estadístico, visualización de datos y predicción de la satisfacción de vida utilizando modelos de **Regresión Lineal** y **KNN**.

# 🏫 Nombre de la Institución:
Universidad Privada de Santa Cruz - UPSA

# 👨‍🏫 Materia / Asignatura:
Python e IA

# 👩‍🏫 Docente Responsable:

**Tito Zúñiga**

# 📅 Fecha de entrega:

11/08/2025

# 🧑‍🤝‍🧑 Integrantes del Grupo

1. **Alvarez Salvatierra Marina Virginia** 2946502.  

2. **Barca Magarzo Carmen**1878909.  
 
3. **Cabrera Machuca Maria Gladys** 6441400.  

4. **Zegarra Uriona Lucia Fernanda** 7856544.  

# 📌 Título del Proyecto
Análisis Descriptivo y Predicción de Tendencias Estadísticas con IA

# INTRODUCCIÓN:
En la era digital actual, el análisis de datos se ha convertido en una herramienta esencial para la toma de decisiones en múltiples disciplinas. La estadística descriptiva, como base del análisis estadístico, permite resumir, organizar y visualizar grandes volúmenes de información de manera efectiva. Con el auge de lenguajes de programación como Python y el desarrollo de herramientas de inteligencia artificial (IA), el procesamiento y análisis de datos ha alcanzado nuevos niveles de eficiencia, precisión y automatización. Este trabajo explora la integración del lenguaje Python y técnicas de IA en la aplicación de la estadística descriptiva, destacando su utilidad, versatilidad y ventajas frente a métodos tradicionales.

# OBJETIVOS:
## 🎯 Objetivo General del Proyecto
Aplicar técnicas.estadísticas e inteligencia artificial para describir, visualizar y predecir comportamientos en un conjunto de  datos, desarrollando competencias analíticas en los integrantes del grupo
## Objetivos específicos: 

 - [ ] Calcular medidas de tendencia central, dispersión y forma
 - [ ] Visualizar distribuciones de datos y correlaciones. 
 - [ ] Aplicar modelos de clasificación o regresión para predecir una variable objetivo
 - [ ] Detectar patrones ocultos en los datos usando técnicas estadísticas y de IA

# 🧩 Descripción del Proyecto

Este proyecto  tiene como finalidad el desarrollo de un sistema analítico que permita realizar un estudio estadístico descriptivo y predictivo utilizando Python y técnicas básicas de inteligencia artificial. El sistema está diseñado para analizar un conjunto de datos reales, generar visualizaciones claras e interpretar patrones y tendencias presentes en la información. Además, incorpora modelos predictivos simples que permiten anticipar comportamientos futuros con base en los datos históricos.
El proyecto está orientado a estudiantes, docentes, analistas de datos principiantes y cualquier persona interesada en adquirir habilidades prácticas en estadística y programación aplicada. A través de este sistema, los usuarios podrán explorar datos de manera intuitiva, detectar valores atípicos, calcular medidas estadísticas clave y generar visualizaciones personalizadas. Asimismo, podrán aplicar modelos de predicción automatizados para apoyar la toma de decisiones informadas.
El desarrollo del sistema se apoya en librerías de Python como Pandas, NumPy, Matplotlib, Seaborn y Scikit-learn. Su implementación no solo facilita el análisis estadístico, sino que también promueve el pensamiento crítico, la interpretación de resultados y el uso responsable de la inteligencia artificial en contextos educativos y prácticos.

# 🌐 Enlace al Proyecto Web
## 🔗 URL del proyecto:

Se creo una pagina web mediante el uso de repositorio de github junto con la base de datos proporcionada de csv, ese repositorio se hizo correr en la pagina de steamlit, a continuación se deja el enlace de la pagina creada
https://proyectofinalfinalgrupo.streamlit.app/#datos-limpios

también se trabajo en la Google colab para la generación de los códigos y visualización previa de los códigos y a continuación se puede ver el enlace:

https://colab.research.google.com/drive/1vUxvNgE6V5TmjdBOj7m2mWftJxPVP0dJ?usp=sharing

# 🗂️ Estructura del Proyecto

1. Módulo de Carga y Limpieza de Datos

Primeramente se limpió la base de datos, haciendo una eliminación de los datos repetidos y la eliminación de los datos en blanco o nan que se tenia, los datos nan se reemplazaron para las variables cuantitativas con el promedio de la variable y para la variable cualitativa se reemplazo con el datos más frecuente
  - Carga de Dataset: Lectura del archivo dataset_estadistica.csv desde el directorio del proyecto.
 - Eliminación de duplicados: Se eliminan registros repetidos para garantizar la calidad de los datos.
 - Imputación de valores nulos:
 -Variables numéricas → reemplazo por la media.
  -Variables categóricas → reemplazo por el valor más frecuente.
 - Exportación de datos limpios: Opción para descargar el dataset procesado en formato CSV.

2. Módulo de Estadística Descriptiva
Ya con la base de datos limpia se procedió a sacar primeramente la estadística descriptiva de dos formas, una mediante la tabla ya establecida de spyder, pero esta tabla no tiene algunas de los parámetros estadísticos, por lo tanto se saco igual todos los parámetros estadísticos.
En la segunda parte del trabajo se realizo los gráficos de los datos, mediante gráficos de tortas, barras, histogramas y boxplot para poder comprender la información

 -Presentación de resumen estadístico general (media, mediana, moda, asimetría, curtosis).
 -Visualizaciones gráficas:
o	Distribución de género (gráfico de pastel).
o	Distribución del nivel educativo (barras con porcentajes).
o	Histogramas con KDE y Boxplots para variables numéricas.
o	Matriz de correlación y Pairplot para analizar relaciones entre variables.

3. Módulo de Predicción Interactiva
Permite al usuario seleccionar entre dos modelos:
•	Regresión Lineal Múltiple: Predice el valor de satisfacción de vida.
•	KNN Clasificador: Clasifica un nuevo caso en base a los vecinos más cercanos.
Funciones principales:
•	Ingreso de datos por sliders (Edad, Ingreso Mensual, Horas de Estudio Semanal).
•	Visualización de métricas de rendimiento (R², MSE).
•	Gráficos comparativos de valores reales vs. predichos.
•	Mapa de dispersión para mostrar ubicación del nuevo dato en el dataset.

4. Pantallas desarrolladas en Streamlit
• Pantalla de carga y vista previa del dataset.
• Pantalla de estadística descriptiva con tablas y gráficos.
• Pantalla de visualización de distribuciones (histogramas, boxplots, correlación).
• Pantalla de predicción interactiva con selección de modelo y visualización de resultados


## 🧠 Lógica de Funcionamiento o Flujo Principal

1. **Inicio**: El usuario abre la app en Streamlit.  

2. **Carga de datos**: El sistema lee `dataset_estadistica.csv`.  
 
3. **Limpieza**: Se eliminan duplicados y se reemplazan nulos.  

4. **Visualización inicial**: El usuario ve tablas y gráficos del dataset.  
5. **Predicción**:  
   - El usuario ingresa sus datos mediante sliders.  
   - Selecciona un modelo predictivo.  
   - Obtiene el resultado y gráficas comparativas.  
 6. **Interacción final**:
 •	El usuario analiza resultados, interpreta gráficas y, si lo desea, descarga el dataset limpio para uso posterior 

---

📷 Capturas de Pantalla
Incluir al menos 3 capturas del sistema funcionando

1. **Captura de pantalla 1** Codigo en spyder funcionando, codigo solo para que corra en spyder y no en steamlit


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d842bb2b-eab4-422c-80b3-57174455e246" />

pestaña de imagenes

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7654f88a-e7bc-4712-b988-79590f819201" />


2. **Captura de pantalla 1** Codigo en spyder para que funcione en steamlit

<img width="916" height="767" alt="image" src="https://github.com/user-attachments/assets/dc06ebb9-67b9-43c5-9c4e-e4d1e3b64708" />

3. **Captura de github** Codigo en spyder para que funcione en steamlit

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8faadb37-a271-487d-9045-9876f1bcced6" />

4. **Captura de steamlit** pagina de steamlit

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/81c7ebc1-4b74-4fcc-9e6f-226b4a507116" />


   
5.  **Captura de Google Colab** pagina de steamlit

<img width="1048" height="440" alt="image" src="https://github.com/user-attachments/assets/9564d002-5c99-4328-88f1-bf82200c0252" />

<img width="1003" height="631" alt="image" src="https://github.com/user-attachments/assets/54b5fb42-e470-4918-ab0c-e32e14fe8b64" />


---

## 🧪 Pruebas Realizadas

- **Funcionales**:
  -Se trabajo primero en google Colab para la creación de los codigos
  -Se trabajo en spyder para la creacción de los codigos y verificar que todos los codigos esten funcionando antes de subir cambiar los codigos para adptarlos para ser cargados en github y que pueda funcionar en   streamlit, se adjunta el enlace de Google Colab para la verificación y el documento de spyder sin codigos de steamlit.
  - Se realizo la cerificación de carga y limpieza del dataset.
  - Comprobación de que las gráficas se generan correctamente.  
  - Validación de entradas y outputs en modelos predictivos.
  - Se genero el codigo para github y se verifico el funcionamiento en streamlit, como se puede ver en el link adjunto:
    
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
  - Uso de github y la pagina de steamlit

- **Mejoras futuras**:  
  - Añadir más modelos predictivos.  
  - Permitir carga de datasets personalizados.  
  - Mejorar visualización para dispositivos móviles y markdown

---

# 📎 Anexo: Código Fuente
- Repositorio GitHub: https://github.com/luciazegarra/proyectofinalfinal
- https://proyectofinalfinalgrupo.streamlit.app/#datos-limpios
- https://colab.research.google.com/drive/1vUxvNgE6V5TmjdBOj7m2mWftJxPVP0dJ?usp=sharing
- También se adjunta en un archivo comprimido (.zip o .rar)
  https://drive.google.com/drive/folders/1vc1zURZB32218vV2uKtBHq6empGVmFPw?usp=sharing

  
