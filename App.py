import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas.api.types as ptypes
import datetime as dt

# --------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------------------------------------
st.set_page_config(
    page_title="Planificador de ocupación hotelera - Perú",
    layout="wide"
)

now = dt.datetime.now()
now_str = now.strftime("%d/%m/%Y %H:%M:%S")

col_time, col_title = st.columns([2, 8])
with col_time:
    st.markdown(f"**📅 Fecha y hora actual:** {now_str}")
with col_title:
    st.title("🧳 Planificador de viaje por ocupación hotelera")

st.markdown("""
Selecciona un **departamento del Perú** y la aplicación te mostrará la
**ocupación hotelera esperada para los próximos 3 meses** (a partir del mes actual).

Además, indicará **en cuál de esos meses habrá mayor afluencia de visitantes**
y explicará el motivo (festividades y/o temporada turística).
""")

# --------------------------------------------------------
# URLS ACTUALIZADAS – ARCHIVO PARQUET LIGERO
# --------------------------------------------------------
PARQUET_URL = "https://drive.google.com/uc?id=1EzhL9JqVVgsA0SrVZK2hLqmzsxBp9bzg"
MODEL_URL    = "https://drive.google.com/uc?id=11DGwFoqmqwb1Llex90aJQRfaDv7wPD6n"

# --------------------------------------------------------
# FUNCIONES DE DESCARGA
# --------------------------------------------------------
@st.cache_data
def load_parquet_from_url(url):
    """Descarga un archivo parquet desde Google Drive y lo carga en memoria."""
    resp = requests.get(url)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))

@st.cache_resource
def load_model_from_url(url):
    """Descarga y carga el modelo RandomForest entrenado."""
    resp = requests.get(url)
    resp.raise_for_status()
    return joblib.loads(resp.content)


# --------------------------------------------------------
# CARGA DE DATA Y MODELO
# --------------------------------------------------------
df = load_parquet_from_url(PARQUET_URL)
model = load_model_from_url(MODEL_URL)

if "target_ocupabilidad" not in df.columns:
    st.error("El archivo parquet no contiene la columna target_ocupabilidad.")
    st.stop()

# --------------------------------------------------------
# PREPARAR X, y y PREDICCIONES
# --------------------------------------------------------
X = df.drop(columns=["target_ocupabilidad"])
y = df["target_ocupabilidad"]
df["prediccion"] = model.predict(X)

mae  = mean_absolute_error(y, df["prediccion"])
rmse = np.sqrt(mean_squared_error(y, df["prediccion"]))
r2   = r2_score(y, df["prediccion"])

# --------------------------------------------------------
# RECONSTRUIR DEPARTAMENTO desde ONE-HOT
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
# COLUMNAS DE MES (si existieran)
# --------------------------------------------------------
possible_month_cols = [c for c in df.columns if c.upper() in ["MES", "MES_NUM", "MES_NOMBRE"]]
MES_COL = possible_month_cols[0] if possible_month_cols else None

MESES_NOMBRES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"
]
MAPA_MES_A_NUM = {nombre: i + 1 for i, nombre in enumerate(MESES_NOMBRES)}
MAPA_NUM_A_MES = {i + 1: nombre for i, nombre in enumerate(MESES_NOMBRES)}

MES_NUM_ACTUAL = now.month
MES_NOMBRE_ACTUAL = MAPA_NUM_A_MES[MES_NUM_ACTUAL]
MESES_SIG_NUM = [((MES_NUM_ACTUAL - 1 + i) % 12) + 1 for i in range(1, 4)]
MESES_SIG_NOMBRES = [MAPA_NUM_A_MES[m] for m in MESES_SIG_NUM]

# --------------------------------------------------------
# CLASIFICACIÓN DE OCUPACIÓN
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
# CATÁLOGO DE FESTIVIDADES
# --------------------------------------------------------
FESTIVIDADES = {
    "CUSCO": [
        {"Mes": "Junio", "Fecha": "24 junio", "Evento": "Inti Raymi"},
        {"Mes": "Junio", "Fecha": "Corpus Christi", "Evento": "Procesión tradicional"},
    ],
    "LIMA": [
        {"Mes": "Julio", "Fecha": "28-29 julio", "Evento": "Fiestas Patrias"},
        {"Mes": "Octubre", "Fecha": "Mes Morado", "Evento": "Procesión del Señor de los Milagros"},
    ],
}

def puntuar_mes(depto, mes_num):
    mes_nombre = MAPA_NUM_A_MES[mes_num]
    festiv = FESTIVIDADES.get(depto.upper(), [])

    fest_mes = [f for f in festiv if f["Mes"].lower() == mes_nombre.lower()]
    score = 0
    razones = []

    if fest_mes:
        score += 2.0
        razones.append("hay festividades importantes")

    if mes_nombre in ["Enero", "Febrero", "Marzo"]:
        score += 1.5
        razones.append("temporada de verano y vacaciones")

    return score, mes_nombre, fest_mes, razones

# --------------------------------------------------------
# INTERFAZ DE USUARIO
# --------------------------------------------------------
st.subheader("✈️ Elige tu destino")

departamentos = sorted(df["DEPARTAMENTO"].unique())
depto_sel = st.selectbox("¿A qué departamento quieres ir?", departamentos)

st.markdown("---")

# --------------------------------------------------------
# FILTRAR PRONÓSTICO
# --------------------------------------------------------
df_depto = df[df["DEPARTAMENTO"] == depto_sel]

subset = df_depto.copy()
ocup_promedio = subset["prediccion"].mean()

percentile = (df["prediccion"] <= ocup_promedio).mean() * 100
porcentaje = round(percentile, 1)

nivel_texto_global, icono_global = clasificar_ocupacion(ocup_promedio)

scores = []
for mes in MESES_SIG_NUM:
    scores.append(puntuar_mes(depto_sel, mes))

score_top, mes_top_nombre, fest_top, razones = max(scores, key=lambda x: x[0])[0:4]

if razones:
    razon_top = ", ".join(razones)
else:
    razon_top = "no se identifican factores de temporada"

st.subheader(f"📅 Pronóstico de ocupación para los próximos 3 meses en {depto_sel}")
st.caption(f"A partir de {MES_NOMBRE_ACTUAL}")

colA, colB = st.columns(2)
with colA:
    st.markdown(f"### {icono_global} {nivel_texto_global}")
    st.metric("Ocupación promedio", f"{ocup_promedio:,.2f}")
with colB:
    st.metric("Percentil histórico", f"{porcentaje}%")

st.write(
    f"Entre los meses **{', '.join(MESES_SIG_NOMBRES)}**, "
    f"el mes más visitado sería **{mes_top_nombre}**, porque {razon_top}."
)

# --------------------------------------------------------
# TABLA DE FESTIVIDADES
# --------------------------------------------------------
st.markdown("---")
st.subheader(f"🎉 Festividades en {depto_sel}")

fest = FESTIVIDADES.get(depto_sel.upper(), [])
if fest:
    st.table(pd.DataFrame(fest))
else:
    st.info("Este departamento no tiene festividades registradas en la app.")

