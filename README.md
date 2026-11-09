# Predicción del volumen de registros consulares

## Descripción del proyecto

Este proyecto desarrolla un modelo de aprendizaje automático para predecir el volumen mensual de registros por oficina consular.

El trabajo sigue la metodología CRISP-DM y utiliza datos históricos para identificar patrones temporales y estimar el comportamiento de los registros.

## Objetivo

Construir una herramienta predictiva que permita estimar el volumen mensual de registros consulares por oficina, como apoyo a la planificación y toma de decisiones.

## Metodología

Se desarrollaron las siguientes etapas:

1. Comprensión del negocio.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelado.
5. Evaluación.
6. Despliegue.

## Modelo seleccionado

Se utilizó Random Forest Regressor con ajuste de hiperparámetros mediante validación cruzada temporal.

Métricas obtenidas:

- MAE en validación cruzada: 41.26
- MAE en prueba: 41.81
- RMSE en prueba: 88.25
- R² en prueba: 0.9249

## Variables utilizadas

- Año
- Número del mes
- Trimestre
- Variables estacionales seno y coseno
- Rezagos de 1, 2, 3, 6 y 12 meses
- Promedios móviles de 3 y 6 meses
- Oficina de circunscripción consular

## Archivos del proyecto

- `app.py`: aplicación web desarrollada con Streamlit.
- `modelo_random_forest.joblib`: modelo entrenado y serializado.
- `requirements.txt`: dependencias necesarias.
- `README.md`: documentación del proyecto.

## Instalación

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Iniciar la aplicación:

```bash
streamlit run app.py
```

## Uso de la aplicación

Seleccionar una oficina consular, indicar el año y el mes que se desea estimar e ingresar los registros reales de los doce meses anteriores.

La aplicación genera una estimación del volumen mensual mediante el modelo entrenado.

## Consideraciones

Las predicciones dependen de la calidad y disponibilidad de los datos históricos. El modelo presenta mayores dificultades para estimar valores excepcionalmente altos.

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
