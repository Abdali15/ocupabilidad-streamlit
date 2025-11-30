import gdown
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas.api.types as ptypes
import datetime as dt

# =======================
# DESCARGA DESDE GOOGLE DRIVE
# =======================

def download_from_drive(url, output):
    """Descarga un archivo desde Google Drive si no existe localmente."""
    if not os.path.exists(output):
        st.write(f"📥 Descargando {output} desde Google Drive...")
        gdown.download(url, output, quiet=False)
    else:
        st.write(f"✔ {output} ya existe, no se descarga.")

# TUS ARCHIVOS (CORREGIDOS)
URL_PICKLE_PREPARADO = "https://drive.google.com/uc?id=1pyXp26KqJQtXTqrBZ-igSKFG-NRmQefm"
URL_MODEL_RF = "https://drive.google.com/uc?id=11DGwFoqmqwb1Llex90aJQRfaDv7wPD6n"

DATA_PATH = "Preparado.pickle"
MODEL_PATH = "modelo_rf_ocupabilidad.pkl"

# Descarga los archivos si no existen
download_from_drive(URL_PICKLE_PREPARADO, DATA_PATH)
download_from_drive(URL_MODEL_RF, MODEL_PATH)

# --------------------------------------------------------
# Configuración de Streamlit
# --------------------------------------------------------
st.set_page_config(
    page_title="Planificador de ocupación hotelera - Perú",
    layout="wide"
)

# Fecha y hora actual
now = dt.datetime.now()
now_str = now.strftime("%d/%m/%Y %H:%M:%S")

col_time, col_title = st.columns([2, 8])
with col_time:
    st.markdown(f"**📅 Fecha y hora actual:** {now_str}")
with col_title:
    st.title("🧳 Planificador de viaje por ocupación hotelera")

st.markdown("""
Selecciona un **departamento del Perú** y la aplicación te mostrará la
**ocupación hotelera esperada para los próximos 3 meses**.

Además, indicará **qué mes será el más transitado** y explicará el motivo.
""")

# --------------------------------------------------------
# Carga de datos y modelo
# --------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_pickle(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"No se pudo cargar un archivo necesario:\n\n{e}")
    st.stop()

if "target_ocupabilidad" not in df.columns:
    st.error("El archivo preparado NO contiene la columna 'target_ocupabilidad'.")
    st.stop()

# --------------------------------------------------------
# Preparar datos y predicciones
# --------------------------------------------------------
X = df.drop(columns=["target_ocupabilidad"])
y = df["target_ocupabilidad"]

df["prediccion"] = model.predict(X)

mae  = mean_absolute_error(y, df["prediccion"])
rmse = np.sqrt(mean_squared_error(y, df["prediccion"]))
r2   = r2_score(y, df["prediccion"])

# --------------------------------------------------------
# Reconstrucción del departamento (one-hot)
# --------------------------------------------------------
dept_cols = [c for c in X.columns if c.startswith("DEPARTAMENTO_")]

if dept_cols:
    dept_matrix = X[dept_cols].values
    idx_max = dept_matrix.argmax(axis=1)
    dept_names = np.array([c.replace("DEPARTAMENTO_", "") for c in dept_cols])
    df["DEPARTAMENTO"] = dept_names[idx_max]
else:
    df["DEPARTAMENTO"] = "No disponible"

# --------------------------------------------------------
# Manejo de meses
# --------------------------------------------------------
possible_month_cols = [c for c in df.columns if c.upper() in ["MES", "MES_NUM", "MES_NOMBRE"]]
MES_COL = possible_month_cols[0] if possible_month_cols else None

MESES_NOMBRES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"
]

MAPA_NUM_A_MES = {i + 1: nombre for i, nombre in enumerate(MESES_NOMBRES)}

MES_NUM_ACTUAL = now.month
MES_NOMBRE_ACTUAL = MAPA_NUM_A_MES[MES_NUM_ACTUAL]

MESES_SIG_NUM = [((MES_NUM_ACTUAL - 1 + i) % 12) + 1 for i in range(1, 4)]
MESES_SIG_NOMBRES = [MAPA_NUM_A_MES[m] for m in MESES_SIG_NUM]

# --------------------------------------------------------
# Clasificación de ocupación
# --------------------------------------------------------
q1, q2 = df["prediccion"].quantile([0.33, 0.66])

def clasificar_ocupacion(valor):
    if valor <= q1:
        return "Baja ocupación (poco transitado)", "🟢"
    elif valor <= q2:
        return "Ocupación media (actividad moderada)", "🟡"
    else:
        return "Alta ocupación (muy transitado)", "🔴"

# --------------------------------------------------------
# Festividades por departamento
# --------------------------------------------------------
FESTIVIDADES = {
    "CUSCO": [
        {"Mes": "Junio", "Fecha": "24 de junio", "Evento": "Inti Raymi"},
        {"Mes": "Junio", "Fecha": "Corpus Christi", "Evento": "Procesión tradicional"},
        {"Mes": "Julio", "Fecha": "15–18 julio", "Evento": "Virgen del Carmen (Paucartambo)"}
    ],
    "LIMA": [
        {"Mes": "Julio", "Fecha": "28–29 julio", "Evento": "Fiestas Patrias"},
        {"Mes": "Octubre", "Fecha": "18–28 octubre", "Evento": "Señor de los Milagros"}
    ]
}

def obtener_festividades_depto_mes(depto, mes):
    fest = FESTIVIDADES.get(depto.upper(), [])
    fest_mes = [f for f in fest if f["Mes"].lower() == mes.lower()]
    return fest, fest_mes

# --------------------------------------------------------
# UI: selección de departamento
# --------------------------------------------------------
st.subheader("✈️ Elige tu destino")

departamentos = sorted(df["DEPARTAMENTO"].unique())
depto_sel = st.selectbox("Departamento", departamentos)

df_depto = df[df["DEPARTAMENTO"] == depto_sel]

# Filtrado por meses futuros
if MES_COL:
    serie_mes = df[MES_COL]
    if ptypes.is_numeric_dtype(serie_mes):
        subset_future = df[(df["DEPARTAMENTO"] == depto_sel) & (serie_mes.isin(MESES_SIG_NUM))]
    else:
        subset_future = df[(df["DEPARTAMENTO"] == depto_sel) &
                           (serie_mes.astype(str).str.title().isin(MESES_SIG_NOMBRES))]
else:
    subset_future = df_depto

if subset_future.empty:
    subset_future = df_depto

ocup_promedio = subset_future["prediccion"].mean()

percentile = (df["prediccion"] <= ocup_promedio).mean() * 100
porcentaje = round(percentile, 1)

nivel_texto_global, icono_global = clasificar_ocupacion(ocup_promedio)

# --------------------------------------------------------
# Mostrar resultado
# --------------------------------------------------------
st.subheader(f"📅 Pronóstico para los próximos 3 meses en {depto_sel}")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### {icono_global} {nivel_texto_global}")
    st.metric("Ocupación esperada (promedio)", f"{ocup_promedio:,.2f}")

with col2:
    st.metric("Percentil histórico", f"{porcentaje:.1f}%")

# --------------------------------------------------------
# Festividades
# --------------------------------------------------------
st.markdown("---")
st.subheader(f"🎉 Festividades en {depto_sel}")

fest_depto = FESTIVIDADES.get(depto_sel.upper(), [])

if fest_depto:
    df_fest = pd.DataFrame(fest_depto)
    st.table(df_fest[["Mes", "Fecha", "Evento"]])
else:
    st.info("No hay festividades registradas para este departamento.")
