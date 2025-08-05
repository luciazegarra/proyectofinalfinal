# 📊 Análisis de Satisfacción de Vida

Aplicación interactiva desarrollada con **Python** y **Streamlit** para el análisis estadístico, visualización de datos y predicción de la satisfacción de vida utilizando modelos de **Regresión Lineal** y **KNN**.

#🏫 Nombre de la Institución:
Universidad Privada de Santa Cruz - UPSA

#👨‍🏫 Materia / Asignatura:
Python e IA

#👩‍🏫 Docente Responsable:

**Tito Zúñiga**

#📅 Fecha de entrega:

11/08/2025
#🧑‍🤝‍🧑 Integrantes del Grupo

Nº	Nombre completo	C.I. / Matrícula	Rol en el proyecto
1	Alvarez Salvatierra Marina Virginia	2946502	
2	Barca Magarzo Carmen Silvia		
3	Cabrera Machuca Maria Gladys		
4	Zegarra Uriona Lucia Fernanda	7856544	

#📌 Título del Proyecto
Análisis Descriptivo y Predicción de Tendencias Estadísticas con IA

# INTRODUCCIÓN:
En la era digital actual, el análisis de datos se ha convertido en una herramienta esencial para la toma de decisiones en múltiples disciplinas. La estadística descriptiva, como base del análisis estadístico, permite resumir, organizar y visualizar grandes volúmenes de información de manera efectiva. Con el auge de lenguajes de programación como Python y el desarrollo de herramientas de inteligencia artificial (IA), el procesamiento y análisis de datos ha alcanzado nuevos niveles de eficiencia, precisión y automatización. Este trabajo explora la integración del lenguaje Python y técnicas de IA en la aplicación de la estadística descriptiva, destacando su utilidad, versatilidad y ventajas frente a métodos tradicionales.

#**OBJETIVOS:**
##Objetivo general:
•	Aplicar técnicas.estadísticas e inteligencia artificial para describir, visualizar y predecir comportamientos en un conjunto de  datos, desarrollando competencias analíticas en los integrantes del grupo
##Objetivos específicos:
•	Calcular medidas de tendencia central, dispersión y forma.
•	Visualizar distribuciones de datos y correlaciones.
•	Aplicar modelos de clasificación o regresión para predecir una variable objetivo 
•	Detectar patrones ocultos en los datos usando técnicas estadísticas y de IA 

#**DESCRIPCIÓN DEL TRABAJO:**
Este trabajo consistió en una revisión teórico-práctica sobre el papel que desempeñan Python y la inteligencia artificial en el análisis de datos desde la perspectiva de la estadística descriptiva. En primer lugar, se abordaron los fundamentos de esta rama de la estadística, para luego introducir las herramientas tecnológicas que permiten una aplicación más eficiente de sus conceptos. A lo largo del desarrollo, se ilustró cómo el uso de Python y de algoritmos básicos de IA facilita la exploración, visualización y síntesis de la información. Finalmente, se reflexionó sobre el impacto de estas tecnologías en el ámbito estadístico y sus posibles aplicaciones futuras.

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

#🌐 Enlace al Proyecto Web
#🔗 URL del proyecto:


Se creo una pagina web mediante el uso de repositorio de github junto con la base de datos proporcionada de csv, ese repositorio se hizo correr en la pagina de steamlit, a continuación se deja el enlace de la pagina creada
https://proyectofinalfinalgrupo.streamlit.app/#datos-limpios

también se trabajo en la Google colab para la generación de los códigos y visualización previa de los códigos y a continuación se puede ver el enlace:

https://colab.research.google.com/drive/1vUxvNgE6V5TmjdBOj7m2mWftJxPVP0dJ?usp=sharing

---

## 🗂️ Estructura del Proyecto

- **Carga y limpieza de datos**
Primeramente se limpio la base de datos, haciendo una eliminación de los datos repetidos y la eliminación de los datos en blanco o nan que se tenia, los datos nan se reemplazaron para las variables cuantitativas con el promedio de la variable y para la variable cualitativa se reemplazo con el datos mas frecuente.  
  - Lectura del dataset CSV.  
  - Eliminación de duplicados.  
  - Imputación de valores nulos (media para numéricos, moda para categóricos).
    
Se añadio la opcion de descarga de la base de datos ya limpia
      - Descarga del dataset limpio.  

- **Análisis estadístico**
Ya con la base de datos limpia se procedió a sacar primeramente la estadística descriptiva de dos formas, una mediante la tabla ya establecida de spyder, pero esta tabla no tiene algunas de los parámetros estadísticos, por lo tanto se saco igual todos los parámetros estadísticos.
En la segunda parte del trabajo se realizo los gráficos de los datos, mediante gráficos de tortas, barras, histogramas y boxplot para poder comprender la información

  - Estadística descriptiva (media, mediana, moda, asimetría, curtosis).  
  - Visualizaciones:  
    - Gráfico de torta (Distribución de género).  
    - Gráfico de barras (Nivel educativo).  
    - Histogramas + KDE.  
    - Boxplots.  
    - Matriz de correlación.  
    - Pairplot de relaciones.  

#**3. Módulo de Predicción Interactiva**
Permite al usuario seleccionar entre dos modelos:
•	Regresión Lineal Múltiple: Predice el valor de satisfacción de vida.
•	KNN Clasificador: Clasifica un nuevo caso en base a los vecinos más cercanos.
Funciones principales:
•	Ingreso de datos por sliders (Edad, Ingreso Mensual, Horas de Estudio Semanal).
•	Visualización de métricas de rendimiento (R², MSE).
•	Gráficos comparativos de valores reales vs. predichos.
•	Mapa de dispersión para mostrar ubicación del nuevo dato en el dataset.

  - Entrada de datos por sliders (edad, ingreso mensual, horas de estudio).  
  - Selección de modelo:  
    - Regresión Lineal.  
    - KNN Clasificador.  
  - Visualización de predicciones y comparación con datos reales.

4. Pantallas desarrolladas en Streamlit
   
•	Pantalla de carga y vista previa del dataset.
•	Pantalla de estadística descriptiva con tablas y gráficos.
•	Pantalla de visualización de distribuciones (histogramas, boxplots, correlación).
•	Pantalla de predicción interactiva con selección de modelo y visualización de resultados

---

## 🧠 Lógica de Funcionamiento o Flujo Principal

1. **Inicio**: El usuario abre la app en Streamlit.  

2. **Carga de datos**: El sistema lee `dataset_estadistica.csv`.  
 
3. **Limpieza**: Se eliminan duplicados y se reemplazan nulos.  

4. **Visualización inicial**: El usuario ve tablas y gráficos del dataset.  
5. **Predicción**:  
   - El usuario ingresa sus datos mediante sliders.  
   - Selecciona un modelo predictivo.  
   - Obtiene el resultado y gráficas comparativas.  
 6. **Descarga**: El usuario puede exportar el dataset limpio.  

---

## 🧪 Pruebas Realizadas

- **Funcionales**:
  -Se trabajo primero en google Colab para la creación de los codigos
  -Se trabajo en spyder para la creacción de los codigos y verificar que todos los codigos esten funcionando antes de subir cambiar los codigos para adptarlos para ser cargados en github y que pueda funcionar en   streamlit, se adjunta el enlace de Google Colab para la verificación y el documento de spyder sin codigos de steamlit.
  - Se realizo la cerificación de carga y limpieza del dataset.
  - - Comprobación de que las gráficas se generan correctamente.  
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

#📎 Anexo: Código Fuente
- Repositorio GitHub: https://github.com/luciazegarra/proyectofinalfinal
- https://proyectofinalfinalgrupo.streamlit.app/#datos-limpios
- https://colab.research.google.com/drive/1vUxvNgE6V5TmjdBOj7m2mWftJxPVP0dJ?usp=sharing
- También se adjunta en un archivo comprimido (.zip o .rar)
  https://drive.google.com/drive/folders/1vc1zURZB32218vV2uKtBHq6empGVmFPw?usp=sharing

  
