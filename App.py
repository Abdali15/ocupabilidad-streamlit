# -*- coding: utf-8 -*-
import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import gdown

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ocupación hotelera",
    page_icon="🏨",
    layout="centered",
)

# ---------------------------------------------------------
# COLUMNAS USADAS EN EL ENTRENAMIENTO DEL MODELO
# (mismas que salieron en tu archivo Preparado.pickle)
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "DEPARTAMENTO_APURÍMAC",
    "DEPARTAMENTO_AREQUIPA",
    "DEPARTAMENTO_AYACUCHO",
    "DEPARTAMENTO_CAJAMARCA",
    "DEPARTAMENTO_CALLAO",
    "DEPARTAMENTO_CUSCO",
    "DEPARTAMENTO_HUANCAVELICA",
    "DEPARTAMENTO_HUÁNUCO",
    "DEPARTAMENTO_ICA",
    "DEPARTAMENTO_JUNÍN",
    "DEPARTAMENTO_LA LIBERTAD",
    "DEPARTAMENTO_LAMBAYEQUE",
    "DEPARTAMENTO_LIMA",
    "DEPARTAMENTO_LORETO",
    "DEPARTAMENTO_MADRE DE DIOS",
    "DEPARTAMENTO_MOQUEGUA",
    "DEPARTAMENTO_PASCO",
    "DEPARTAMENTO_PIURA",
    "DEPARTAMENTO_PUNO",
    "DEPARTAMENTO_SAN MARTÍN",
    "DEPARTAMENTO_TACNA",
    "DEPARTAMENTO_TUMBES",
    "DEPARTAMENTO_UCAYALI",
    "DEPARTAMENTO_ÁNCASH",
    "segmento_hotel_ALBERGUE TODAS CONSOLIDADAS",
    "segmento_hotel_APART HOTEL 3 ESTRELLAS",
    "segmento_hotel_APART HOTEL 4 ESTRELLAS",
    "segmento_hotel_APART HOTEL 5 ESTRELLAS",
    "segmento_hotel_APART HOTEL TODAS CONSOLIDADAS",
    "segmento_hotel_ECOLODGE ECOLODGE",
    "segmento_hotel_ECOLODGE TODAS CONSOLIDADAS",
    "segmento_hotel_HOSTAL 1 ESTRELLA",
    "segmento_hotel_HOSTAL 2 ESTRELLAS",
    "segmento_hotel_HOSTAL 3 ESTRELLAS",
    "segmento_hotel_HOSTAL TODAS CONSOLIDADAS",
    "segmento_hotel_HOTEL 1 ESTRELLA",
    "segmento_hotel_HOTEL 2 ESTRELLAS",
    "segmento_hotel_HOTEL 3 ESTRELLAS",
    "segmento_hotel_HOTEL 4 ESTRELLAS",
    "segmento_hotel_HOTEL 5 ESTRELLAS",
    "segmento_hotel_HOTEL TODAS CONSOLIDADAS",
    "segmento_hotel_NO CLASIFICADO NO CATEGORIZADO",
    "segmento_hotel_NO CLASIFICADO TODAS CONSOLIDADAS",
    "segmento_hotel_NO DISPONIBLE NO DISPONIBLE",
    "segmento_hotel_RESORT 3 ESTRELLAS",
    "segmento_hotel_RESORT 4 ESTRELLAS",
    "segmento_hotel_RESORT 5 ESTRELLAS",
    "segmento_hotel_RESORT TODAS CONSOLIDADAS",
    "segmento_hotel_TODAS CONSOLIDADAS 1 ESTRELLA",
    "segmento_hotel_TODAS CONSOLIDADAS 2 ESTRELLAS",
    "segmento_hotel_TODAS CONSOLIDADAS 3 ESTRELLAS",
    "segmento_hotel_TODAS CONSOLIDADAS 4 ESTRELLAS",
    "segmento_hotel_TODAS CONSOLIDADAS 5 ESTRELLAS",
    "segmento_hotel_TODAS CONSOLIDADAS ALBERGUE",
    "segmento_hotel_TODAS CONSOLIDADAS ECOLODGE",
    "segmento_hotel_TODAS CONSOLIDADAS NO CATEGORIZADO",
    "segmento_hotel_TODAS CONSOLIDADAS TODAS CONSOLIDADAS",
    "NÚMERO DE ESTABLECIMIENTO_2019",
    "NÚMERO DE HABITACIONES_2019",
    "TOTAL PERNOCT MES - EXT (DÍAS)_2019",
    "TOTAL DE ARRIBOS EN EL MES_2019",
]

# Valores numéricos "típicos" (aprox.) para las variables continuas
DEFAULT_NUMERIC_VALUES = {
    "NÚMERO DE ESTABLECIMIENTO_2019": 0.043,
    "NÚMERO DE HABITACIONES_2019": 0.038,
    "TOTAL PERNOCT MES - EXT (DÍAS)_2019": 0.002,
    "TOTAL DE ARRIBOS EN EL MES_2019": 0.007,
}

# Listas para los selectbox (las sacamos de FEATURE_COLUMNS)
DEPARTAMENTOS = sorted(
    [c.replace("DEPARTAMENTO_", "") for c in FEATURE_COLUMNS if c.startswith("DEPARTAMENTO_")]
)
SEGMENTOS = sorted(
    [c.replace("segmento_hotel_", "") for c in FEATURE_COLUMNS if c.startswith("segmento_hotel_")]
)

# ---------------------------------------------------------
# DESCARGA Y CARGA DEL MODELO
# ---------------------------------------------------------
MODEL_FILE = "modelo_rf_ocupabilidad.pkl"
MODEL_ID = "11DGwFoqmqwb1Llex90aJQRfaDv7wPD6n"  # id de tu archivo en Drive
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"


def download_model_if_needed() -> None:
    """Descarga el modelo desde Google Drive si no existe localmente."""
    if not os.path.exists(MODEL_FILE):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)


@st.cache_resource(show_spinner="Cargando modelo de ocupabilidad...")
def load_model():
    download_model_if_needed()
    model = joblib.load(MODEL_FILE)
    return model


# ---------------------------------------------------------
# FUNCIONES AUXILIARES PARA FEATURES Y PRONÓSTICOS
# ---------------------------------------------------------
def crear_vector_caracteristicas(departamento: str, segmento: str) -> pd.DataFrame:
    """
    Crea un DataFrame con una sola fila y todas las columnas que espera el modelo,
    activando el departamento y segmento seleccionados, y usando valores numéricos por defecto.
    """
    X = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS, dtype=float)

    # One-hot del departamento
    dept_col = f"DEPARTAMENTO_{departamento}"
    if dept_col in X.columns:
        X.at[0, dept_col] = 1.0

    # One-hot del segmento hotelero
    seg_col = f"segmento_hotel_{segmento}"
    if seg_col in X.columns:
        X.at[0, seg_col] = 1.0

    # Variables numéricas con valores por defecto
    for col, val in DEFAULT_NUMERIC_VALUES.items():
        if col in X.columns:
            X.at[0, col] = float(val)

    return X


MESES_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def proximo_tres_meses():
    hoy = dt.date.today()
    meses = []
    for i in range(3):
        m = (hoy.month - 1 + i) % 12 + 1
        y = hoy.year + (hoy.month - 1 + i) // 12
        meses.append((m, y))
    return meses


def factor_estacional(mes: int) -> float:
    """Factor simple para simular alta / media / baja temporada."""
    # Alta: vacaciones y fiestas (julio, agosto, diciembre)
    if mes in (7, 8, 12):
        return 1.15
    # Media: enero, marzo, junio
    if mes in (1, 3, 6):
        return 1.05
    # Baja: resto
    return 0.95


def razon_mes(mes: int) -> str:
    if mes == 7:
        return "por Fiestas Patrias y las vacaciones de medio año."
    if mes == 8:
        return "por la continuación de las vacaciones de medio año."
    if mes == 12:
        return "por las fiestas de fin de año y vacaciones largas."
    if mes in (1, 3, 6):
        return "por tratarse de una temporada media con flujo turístico moderado."
    return "porque se espera un flujo turístico relativamente menor frente a los otros meses."


# ---------------------------------------------------------
# INTERFAZ DE USUARIO
# ---------------------------------------------------------
st.title("hotelera")

st.write(
    """
Selecciona un **departamento del Perú** y un **segmento hotelero**, 
y la aplicación te mostrará la **ocupación hotelera esperada para los próximos 3 meses** 
(a partir del mes actual).

Además, indicará **en cuál de esos meses habría mayor afluencia de visitantes** 
y explicará brevemente el motivo.
"""
)

modelo = load_model()

with st.form("form_ocupabilidad"):
    departamento = st.selectbox("Departamento del Perú", DEPARTAMENTOS, index=DEPARTAMENTOS.index("LIMA"))
    segmento = st.selectbox("Segmento de hotel", SEGMENTOS)

    submitted = st.form_submit_button("Calcular ocupación esperada")

if submitted:
    # Crear vector de características
    X = crear_vector_caracteristicas(departamento, segmento)

    # Predicción base
    pred_base = float(modelo.predict(X)[0])

    meses = proximo_tres_meses()
    resultados = []

    for mes, anio in meses:
        factor = factor_estacional(mes)
        pred_mes = pred_base * factor
        resultados.append((mes, anio, pred_mes))

    # Mostrar resultados
    st.subheader("Ocupación hotelera esperada (visitantes / pernoctaciones)")

    for mes, anio, valor in resultados:
        st.write(
            f"- **{MESES_ES[mes].capitalize()} {anio}**: {valor:,.0f} visitantes (aprox.)"
        )

    # Mes con mayor ocupación
    mejor_mes, mejor_anio, mejor_valor = max(resultados, key=lambda x: x[2])

    st.success(
        f"El mes con **mayor afluencia esperada** es **{MESES_ES[mejor_mes].capitalize()} {mejor_anio}**, "
        f"con aproximadamente **{mejor_valor:,.0f} visitantes**, {razon_mes(mejor_mes)}"
    )

    st.caption(
        "Nota: Los valores mostrados son estimaciones basadas en el modelo de Random Forest entrenado "
        "con datos históricos de ocupación hotelera en el Perú."
    )



