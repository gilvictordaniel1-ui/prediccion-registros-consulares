import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Predicción de registros consulares",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_random_forest.joblib")

st.title("Predicción del volumen de registros consulares")
st.write(
    "Herramienta de predicción mensual por oficina consular, "
    "desarrollada mediante aprendizaje automático."
)

try:
    paquete = cargar_modelo()
    modelo = paquete["modelo"]
    columnas = paquete["columnas"]
    oficinas = paquete["oficinas"]
except Exception as e:
    st.error(f"No fue posible cargar el modelo: {e}")
    st.stop()

st.info(
    "Selecciona la oficina, el año y el mes que deseas estimar. "
    "Debes ingresar los registros reales de los 12 meses anteriores."
)

with st.form("formulario_prediccion"):

    oficina = st.selectbox("Oficina consular", oficinas)

    col1, col2 = st.columns(2)

    with col1:
        anio = st.number_input(
            "Año de predicción",
            min_value=2014,
            max_value=2100,
            value=2027,
            step=1
        )

    with col2:
        nombres_meses = [
            "Enero", "Febrero", "Marzo", "Abril",
            "Mayo", "Junio", "Julio", "Agosto",
            "Septiembre", "Octubre", "Noviembre", "Diciembre"
        ]

        mes = st.selectbox(
            "Mes de predicción",
            range(1, 13),
            format_func=lambda x: nombres_meses[x - 1]
        )

    st.subheader("Registros históricos")
    st.caption(
        "Ingresa valores reales. El primer campo corresponde "
        "al mes inmediatamente anterior."
    )

    historial = []

    for i in range(1, 13):
        valor = st.number_input(
            f"Mes anterior {i}",
            min_value=0,
            value=0,
            step=1,
            key=f"historial_{i}"
        )
        historial.append(valor)

    calcular = st.form_submit_button("Calcular predicción")

if calcular:

    # Construcción de variables con la misma lógica del entrenamiento
    datos = {
        "Año": int(anio),
        "Numero_mes": int(mes),
        "Trimestre": (int(mes) - 1) // 3 + 1,
        "Mes_seno": np.sin(2 * np.pi * int(mes) / 12),
        "Mes_coseno": np.cos(2 * np.pi * int(mes) / 12),
        "Rezago_1": historial[0],
        "Rezago_2": historial[1],
        "Rezago_3": historial[2],
        "Rezago_6": historial[5],
        "Rezago_12": historial[11],
        "Promedio_movil_3": np.mean(historial[:3]),
        "Promedio_movil_6": np.mean(historial[:6]),
        "Oficina de circunscripción consular": oficina
    }

    entrada = pd.DataFrame([datos]).reindex(columns=columnas)

    if entrada.isnull().any().any():
        st.error("Hay variables faltantes. No se puede realizar la predicción.")
    else:
        prediccion = max(0, float(modelo.predict(entrada)[0]))

        st.success("Predicción calculada correctamente.")
        st.metric(
            "Volumen mensual estimado",
            f"{prediccion:,.0f} registros"
        )

        st.caption(
            "La estimación depende de los datos históricos ingresados "
            "y no representa una cifra garantizada."
        )

st.divider()
st.subheader("Información del modelo")

st.write("**Modelo:**", paquete["nombre"])
st.write("**MAE validación cruzada:**", round(paquete["mae_validacion"], 2))
st.write("**MAE prueba:**", round(paquete["mae_prueba"], 2))
st.write("**R² prueba:**", round(paquete["r2_prueba"], 4))
