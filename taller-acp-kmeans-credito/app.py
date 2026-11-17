"""
Aplicación del taller "ACP y K-Means": capacidad de cupo de empresas calificadas.

Ejecutar con:
    streamlit run app.py

Todo lo que se muestra se calcula aquí, a partir del CSV, con las funciones de
analisis.py. No hay tablas ni imágenes precalculadas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analisis as A

RUTA = Path(__file__).parent / "data" / "corporateCreditRatingWithFinancialRatios.csv"

st.set_page_config(
    page_title="Cupo y perfiles financieros: ACP y K-Means",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Identidad visual
# ---------------------------------------------------------------------------

TINTA, VERDE, OCRE, VINO, GRIS = "#1D2B36", "#2E5E4E", "#C8963E", "#A63D40", "#8A9A94"
PALETA_GRUPOS = ["#2E5E4E", "#C8963E", "#A63D40", "#4F7F8C", "#7A9A63",
                 "#B5703A", "#6B4E71", "#8C8C3A", "#3E6A8A", "#7E2E3A"]
FUENTE = "Libre Franklin, Helvetica, Arial, sans-serif"
FUENTE_TITULOS = "Source Serif 4, Georgia, serif"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap');
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp input,
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stCaptionContainer"] { font-family: 'Libre Franklin', Helvetica, Arial, sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Source Serif 4', Georgia, serif !important;
  color: #1D2B36; letter-spacing: -0.01em; }
.stApp h1 { font-weight: 700; font-size: 2.25rem; line-height: 1.15; max-width: 52rem; }
.stApp h2 { font-weight: 600; font-size: 1.55rem; margin-top: 1.4rem; }
.stApp h3 { font-weight: 600; font-size: 1.2rem; }
.lead { font-size: 1.08rem; line-height: 1.6; color: #3A4A55; max-width: 60rem; }
.nota { font-size: .92rem; color: #4A5A64; border-left: 3px solid #C8963E; padding: .35rem .8rem;
  background: #FBF6EC; margin: .6rem 0 1rem 0; max-width: 62rem; }
/* Escala de calificación */
.escala-wrap { margin: 1.2rem 0 .4rem 0; max-width: 64rem; }
.pines { position: relative; height: 172px; }
.pin { position: absolute; transform: translateX(-50%); text-align: center; }
.pin .rotulo { display: inline-block; white-space: nowrap; font-family: 'Libre Franklin', sans-serif;
  font-size: .82rem; font-weight: 600; color: #1D2B36; background: #FFFFFF; padding: .25rem .55rem;
  border-radius: 3px; border-top: 3px solid var(--c); box-shadow: 0 1px 0 #D9DFDB; }
.pin .rotulo small { display: block; font-weight: 500; color: #5B6B75; font-size: .74rem; }
.pin .tallo { width: 2px; background: var(--c); margin: 0 auto; }
.escala { display: grid; grid-template-columns: repeat(22, 1fr); gap: 2px; }
.escala div { font-family: 'Libre Franklin', sans-serif; font-size: .72rem; font-weight: 600;
  text-align: center; padding: .45rem 0; color: #FFFFFF; border-radius: 2px; }
.escala .ig { background: #2E5E4E; }
.escala .es { background: #A63D40; }
.escala-ejes { display: grid; grid-template-columns: 10fr 12fr; gap: 2px; margin-top: .3rem;
  font-family: 'Libre Franklin', sans-serif; font-size: .78rem; color: #4A5A64; }
.escala-ejes div:last-child { text-align: right; }
.ficha { background: #FFFFFF; border-radius: 4px; padding: 1rem 1.2rem; border-left: 5px solid var(--c); }
.ficha .t { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem; font-weight: 600; color: #1D2B36; }
.ficha .s { font-family: 'Libre Franklin', sans-serif; color: #4A5A64; font-size: .93rem; margin-top: .25rem; }
</style>
""",
    unsafe_allow_html=True,
)


def fmt(x, d=1):
    """Número con punto de miles y coma decimal."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def estilo(fig, alto=430, titulo=None, leyenda_abajo=True):
    fig.update_layout(
        height=alto, title=titulo, separators=",.",
        font=dict(family=FUENTE, color=TINTA, size=13),
        title_font=dict(family=FUENTE_TITULOS, size=17, color=TINTA),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=70, r=20, t=90 if titulo else 45, b=60),
        hoverlabel=dict(font_family=FUENTE),
    )
    if leyenda_abajo:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""))
    fig.update_xaxes(gridcolor="#DDE3DF", zerolinecolor="#B8C2BD", linecolor="#B8C2BD", automargin=True)
    fig.update_yaxes(gridcolor="#DDE3DF", zerolinecolor="#B8C2BD", linecolor="#B8C2BD", automargin=True)
    return fig


def mostrar(fig):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def tabla(df, decimales=None, defecto=2, indice=True, filas_enteras=()):
    """Muestra un DataFrame con formato numérico en español."""
    decimales = decimales or {}

    def formateador(col):
        d = decimales.get(col, defecto)

        def f(v):
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                return "–"
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                return fmt(v, d)
            return v
        return f

    sty = df.style.format({c: formateador(c) for c in df.columns}, na_rep="–")
    filas = [f for f in filas_enteras if f in df.index]
    if filas:
        sty = sty.format(lambda v: fmt(v, 0), subset=pd.IndexSlice[filas, :])
    st.dataframe(sty, hide_index=not indice)


ESCALA_DIVERGENTE = [[0, "#A63D40"], [0.5, "#F4F2EE"], [1, "#2E5E4E"]]

# ---------------------------------------------------------------------------
# Cálculos (con caché: solo se repiten si cambian los parámetros)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Leyendo y reparando el archivo…")
def datos():
    df, registro = A.cargar_datos(str(RUTA))
    obs = A.construir_observaciones(df)
    return df, registro, obs


@st.cache_data(show_spinner="Preparando los datos y ajustando el ACP…")
def modelo_acp(variables: tuple, q: float, normalizar: bool):
    _, _, obs = datos()
    prep = A.preparar_matriz(obs, list(variables), q, normalizar)
    acp = A.orientar_componentes(A.ajustar_acp(prep.Z, prep.variables))
    return prep, acp


@st.cache_data(show_spinner="Probando k de 2 a 10…")
def evaluar(variables: tuple, q: float, normalizar: bool, n_comp: int, semilla: int):
    _, acp = modelo_acp(variables, q, normalizar)
    return A.evaluar_k(acp["puntajes"][:, :n_comp], 2, 10, semilla)


@st.cache_data(show_spinner="Agrupando con K-Means…")
def segmentar(variables: tuple, q: float, normalizar: bool, n_comp: int, k: int, semilla: int):
    _, _, obs = datos()
    prep, acp = modelo_acp(variables, q, normalizar)
    S = acp["puntajes"][:, :n_comp]
    km = A.ajustar_kmeans(S, k, semilla)
    nombres = A.nombrar_segmentos(obs, prep, km["etiquetas"])
    validacion = A.validar_con_rating(obs, km["etiquetas"], nombres)
    validacion["Postura de cupo"] = validacion["Grado de inversión (%)"].apply(lambda p: A.lectura_de_cupo(p)[0])
    # Colores y orden por rentabilidad del centro (CP1), no por la calificación
    orden_g = list(np.argsort(-km["modelo"].cluster_centers_[:, 0]))
    orden = [nombres[g] for g in orden_g]
    colores = {nombres[g]: PALETA_GRUPOS[i % len(PALETA_GRUPOS)] for i, g in enumerate(orden_g)}
    return km, nombres, validacion, orden, colores


# ---------------------------------------------------------------------------
# Barra lateral: navegación y parámetros
# ---------------------------------------------------------------------------

PAGINAS = ["1. Contexto", "2. Datos", "3. Preparación", "4. ACP", "5. K-Means",
           "6. Simulador de cupo", "7. Conclusiones"]

df, registro, obs = datos()

with st.sidebar:
    st.markdown("### Capacidad de cupo")
    st.caption("Taller de ACP y K-Means. Machine Learning aplicado a Economía y Finanzas, PUJ Cali.")
    pagina = st.radio("Sección", PAGINAS, label_visibility="collapsed")
    st.divider()
    st.markdown("**Parámetros del modelo**")
    st.caption("Todo se recalcula al cambiarlos. Los valores iniciales son los del análisis.")
    variables = st.multiselect(
        "Razones que entran al modelo", A.RATIOS, default=A.VARIABLES_MODELO,
        format_func=A.nombre,
        help="Las seis razones que no vienen marcadas se excluyen por los motivos de la pantalla 3.",
    )
    q_pct = st.select_slider(
        "Winsorización: percentil recortado en cada cola", options=[0.0, 0.5, 1.0, 2.5, 5.0], value=1.0,
        format_func=lambda v: f"{fmt(v, 1)} %",
        help="Los valores por debajo del percentil elegido se llevan a ese percentil, y lo mismo en la cola alta.",
    )
    normalizar = st.toggle(
        "Normalizar cada fila a longitud 1", value=False,
        help="Es el paso adicional del cuaderno de clase. Borra la magnitud y deja solo el estilo del perfil.",
    )

if len(variables) < 4:
    st.error("Elige al menos cuatro razones en la barra lateral: el taller pide un mínimo de cuatro variables cuantitativas.")
    st.stop()

VARS = tuple(v for v in A.RATIOS if v in variables)  # orden estable
Q = q_pct / 100
prep, acp = modelo_acp(VARS, Q, normalizar)

with st.sidebar:
    usar_kaiser = st.toggle("Número de componentes por criterio de Kaiser", value=True,
                            help="Conserva los componentes con valor propio mayor que el promedio.")
    if usar_kaiser:
        n_comp = max(2, acp["kaiser"])
        st.caption(f"Kaiser conserva {acp['kaiser']} componente(s); se usan {n_comp}.")
    else:
        n_comp = st.slider("Componentes que se conservan", 2, len(VARS), min(3, len(VARS)))
    k = st.slider("Número de grupos (k)", 2, 10, 3)
    semilla = int(st.number_input("Semilla de K-Means", min_value=0, max_value=9999, value=42, step=1))
    st.divider()
    st.caption("Datos: *Corporate Credit Rating with Financial Ratios* (Kaggle). "
               "Código: `app.py` y `analisis.py` en el repositorio.")

km, nombres, validacion, orden, colores = segmentar(VARS, Q, normalizar, n_comp, k, semilla)
S = acp["puntajes"][:, :n_comp]
etiquetas = km["etiquetas"]
segmento = pd.Series([nombres[g] for g in etiquetas], index=obs.index, name="Segmento")
var_tab = acp["varianza"]
var_conservada = var_tab["Acumulada (%)"].iloc[n_comp - 1]

ACP_DEFECTO = set(VARS) == set(A.VARIABLES_MODELO) and Q == 0.01 and not normalizar
TODO_DEFECTO = ACP_DEFECTO and n_comp == 3 and k == 3 and semilla == 42
NOMBRES_CP = {0: "rentabilidad", 1: "rotación frente a margen", 2: "solidez de balance"}


def etiqueta_cp(i):
    base = f"CP{i + 1}"
    if ACP_DEFECTO and i in NOMBRES_CP:
        base += f" ({NOMBRES_CP[i]})"
    return f"{base}, {fmt(var_tab['Varianza explicada (%)'].iloc[i])} %"


def aviso_configuracion():
    if not TODO_DEFECTO:
        st.markdown(
            '<div class="nota">Los parámetros de la barra lateral son distintos a los del análisis '
            "(10 razones, winsorización al 1 %, sin normalizar filas, 3 componentes, k = 3, semilla 42). "
            "Las cifras y gráficos de esta pantalla corresponden a tus parámetros; los párrafos de "
            "interpretación describen la configuración original.</div>",
            unsafe_allow_html=True,
        )


def tira_rating(val: pd.DataFrame):
    """La escala de AAA a D, con la calificación mediana de cada grupo marcada encima."""
    celdas = "".join(
        f'<div class="{"ig" if p <= 10 else "es"}">{A.PUNTAJE_A_RATING[p]}</div>' for p in range(1, 23)
    )
    pines = []
    for seg in orden:
        p = int(np.clip(round(val.loc[seg, "Puntaje mediano"]), 1, 22))
        nivel = len(pines) % 3          # alturas escalonadas para que los rótulos no se monten
        izq = (p - 0.5) / 22 * 100
        alto_tallo = 14 + 54 * (2 - nivel)
        abajo = 0
        pines.append(
            f'<div class="pin" style="left:{izq:.2f}%; bottom:{abajo}px; --c:{colores[seg]}">'
            f'<div class="rotulo">{seg}<small>{A.PUNTAJE_A_RATING[p]}, '
            f'{fmt(val.loc[seg, "Grado de inversión (%)"], 0)} % con grado de inversión</small></div>'
            f'<div class="tallo" style="height:{alto_tallo}px"></div></div>'
        )
    html = (
        '<div class="escala-wrap">'
        f'<div class="pines">{"".join(pines)}</div>'
        f'<div class="escala">{celdas}</div>'
        '<div class="escala-ejes"><div>Grado de inversión: AAA a BBB-</div>'
        "<div>Grado especulativo: BB+ a D</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def mapa_grupos(ejes=(0, 1), alto=520, extra=None):
    """Dispersión de las empresas en dos componentes, coloreadas por grupo."""
    i, j = ejes
    base = pd.DataFrame({"x": S[:, i] if i < n_comp else acp["puntajes"][:, i],
                         "y": S[:, j] if j < n_comp else acp["puntajes"][:, j],
                         "Segmento": segmento, "Empresa": obs["Etiqueta"],
                         "Rating": obs["Rating mediano"]})
    fig = px.scatter(base, x="x", y="y", color="Segmento", color_discrete_map=colores,
                     category_orders={"Segmento": orden}, opacity=.55,
                     hover_data={"Empresa": True, "Rating": True, "x": ":.2f", "y": ":.2f", "Segmento": False})
    fig.update_traces(marker=dict(size=6))
    centros = km["modelo"].cluster_centers_
    if i < n_comp and j < n_comp:
        fig.add_trace(go.Scatter(x=centros[:, i], y=centros[:, j], mode="markers", name="Centroides",
                                 marker=dict(symbol="x", size=16, color=TINTA, line=dict(width=2, color="white")),
                                 hovertext=[nombres[g] for g in range(len(centros))], hoverinfo="text"))
    if extra is not None:
        fig.add_trace(extra)
    fig.update_xaxes(title=etiqueta_cp(i))
    fig.update_yaxes(title=etiqueta_cp(j))
    return estilo(fig, alto)


# ---------------------------------------------------------------------------
# 1. Contexto
# ---------------------------------------------------------------------------

def pagina_contexto():
    st.title("Cuánto cupo darle a una empresa, leído desde sus razones financieras")
    st.markdown(
        '<p class="lead">Un comité de crédito decide el cupo de una empresa mirando su liquidez, su deuda, '
        "sus márgenes y su rentabilidad. Este trabajo resume diez de esas razones con análisis de componentes "
        "principales y agrupa a 2.694 empresas-periodo con K-Means. La calificación de las agencias no entra al "
        "modelo; se usa después, para comprobar si los grupos tienen sentido crediticio. Así quedan en la escala:</p>",
        unsafe_allow_html=True,
    )
    tira_rating(validacion)
    st.caption("Calificación mediana de cada grupo con los parámetros actuales. Los grupos se formaron solo con razones financieras.")

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.header("La pregunta")
        st.markdown(
            "¿Las razones financieras de las empresas calificadas se pueden resumir en pocas dimensiones y agrupar "
            "en perfiles que correspondan a capacidades de cupo distintas?\n\n"
            "- **Estructura.** ¿Cuántos ejes bastan para describir la situación financiera de una empresa, y de qué razones están hechos? (ACP)\n"
            "- **Grupos.** ¿Cuántos perfiles financieros distintos sostiene la base y en qué se diferencian? (K-Means)\n"
            "- **Cupo.** ¿Esos perfiles corresponden a calificaciones distintas, y qué postura de cupo sugiere cada uno?\n\n"
            "Un perfil le dice al analista a qué empresas se parece la que tiene enfrente antes de hacer el análisis "
            "completo, y permite fijar políticas por grupo: monto, garantías, covenants y frecuencia de revisión."
        )
        st.header("Por qué ACP y K-Means")
        st.markdown(
            "- **Hay redundancia.** Los márgenes operativo, EBITDA, antes de impuestos y neto tienen correlaciones de 0,63 a 0,93 entre sí. "
            "El ACP muestra cuántas historias distintas hay detrás de las diez columnas.\n"
            "- **No hay etiqueta de cupo.** Nadie dice cuánto cupo merece cada empresa. K-Means agrupa sin etiqueta y resume "
            "cada grupo en un centro, la *empresa típica*, que sirve para ubicar a una empresa nueva.\n"
            "- **El rating es una prueba independiente.** Al dejarlo fuera del modelo, se puede verificar si los grupos "
            "formados con razones corresponden a calidades crediticias distintas."
        )
    with c2:
        st.header("Los datos")
        m1, m2 = st.columns(2)
        m1.metric("Calificaciones en el archivo", fmt(len(df), 0))
        m2.metric("Empresas-periodo analizadas", fmt(len(obs), 0))
        m1.metric("Empresas distintas", fmt(obs["CIK"].nunique(), 0))
        m2.metric("Periodo", f"{df['Rating Date'].dt.year.min()}–{df['Rating Date'].dt.year.max()}")
        st.markdown(
            "**Fuente.** *Corporate Credit Rating with Financial Ratios*, conjunto público de Kaggle. Une calificaciones de "
            "siete agencias (Egan-Jones, S&P, Moody's, Fitch, DBRS, JCR y HR Ratings) a empresas que cotizan en Estados "
            "Unidos con 16 razones de sus estados financieros. Las empresas se identifican con el CIK de la SEC.\n\n"
            "**Unidad de análisis.** Cada fila del archivo es una calificación. Como un mismo estado financiero recibe "
            "varias calificaciones, se deja una fila por empresa y juego de razones, con la mediana de sus calificaciones."
        )
        with st.expander("Limitaciones de la fuente"):
            st.markdown(
                "- Solo empresas grandes, listadas en Estados Unidos y calificadas: no representan a pymes ni a empresas colombianas.\n"
                "- Ventana 2010–2016, un ciclo de recuperación con tasas bajas.\n"
                "- La fuente no documenta la fórmula de cada razón.\n"
                "- Hay fecha de calificación, no de cierre contable: el año es aproximado.\n"
                "- Una empresa aparece en varios años; sus observaciones no son independientes.\n"
                "- 1.707 líneas del CSV vienen mal formadas y se reparan al leer."
            )


# ---------------------------------------------------------------------------
# 2. Datos
# ---------------------------------------------------------------------------

def pagina_datos():
    st.title("Los datos: 7.805 calificaciones, 2.694 estados financieros")
    st.markdown(
        '<p class="lead">Antes de modelar hay que entender qué representa cada fila, reparar el archivo y '
        "decidir la unidad de análisis.</p>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["Lectura del archivo", "Calificaciones", "Duplicados y unidad de análisis",
                                  "Estadísticas descriptivas", "Distribución de una razón"])
    with t1:
        st.markdown(
            "Un `pd.read_csv` directo falla en la línea 1.996. El archivo trae dos defectos, que `analisis.cargar_datos` "
            "repara línea por línea sin descartar filas:")
        rep = pd.DataFrame({"Corrección": list(registro.keys()), "Filas": list(registro.values())})
        tabla(rep, defecto=0, indice=False)
        st.markdown(
            "- **Líneas completas entre comillas:** el lector las ve como una sola columna; se deshace el encierro.\n"
            "- **Coma en `Japan Credit Rating Agency,Ltd.`:** parte el nombre de la agencia en dos columnas; se vuelve a unir.\n\n"
            f"Después de la reparación no hay celdas vacías (faltantes: {int(df.isna().sum().sum())}), así que no hace falta imputar.")
        st.subheader("Primeras filas del archivo reparado")
        st.dataframe(df.head(20), hide_index=True)

    with t2:
        c1, c2 = st.columns(2)
        orden_r = [r for r in A.PUNTAJE_A_RATING.values() if r in df["Rating"].unique()]
        cuenta = df["Rating"].value_counts().reindex(orden_r).reset_index()
        cuenta.columns = ["Rating", "Calificaciones"]
        cuenta["Grado"] = np.where(cuenta["Rating"].map(A.ESCALA_RATING) <= 10, "Grado de inversión", "Grado especulativo")
        fig = px.bar(cuenta, x="Rating", y="Calificaciones", color="Grado",
                     color_discrete_map={"Grado de inversión": VERDE, "Grado especulativo": VINO})
        with c1:
            mostrar(estilo(fig, 380, "Calificaciones por letra"))
        ag = df["Rating Agency"].value_counts().reset_index()
        ag.columns = ["Agencia", "Calificaciones"]
        fig = px.bar(ag, y="Agencia", x="Calificaciones", orientation="h", color_discrete_sequence=[VERDE])
        fig.update_yaxes(autorange="reversed", title="")
        with c2:
            mostrar(estilo(fig, 380, "Calificaciones por agencia"))
        c3, c4 = st.columns(2)
        an = df["Rating Date"].dt.year.value_counts().sort_index().reset_index()
        an.columns = ["Año", "Calificaciones"]
        fig = px.bar(an, x="Año", y="Calificaciones", color_discrete_sequence=[GRIS])
        with c3:
            mostrar(estilo(fig, 340, "Calificaciones por año"))
        se = df["Sector"].map(A.SECTORES).value_counts().reset_index()
        se.columns = ["Sector", "Calificaciones"]
        fig = px.bar(se, y="Sector", x="Calificaciones", orientation="h", color_discrete_sequence=[OCRE])
        fig.update_yaxes(autorange="reversed", title="")
        with c4:
            mostrar(estilo(fig, 340, "Calificaciones por sector"))
        st.markdown(
            f"El {fmt((df['Puntaje rating'] <= 10).mean() * 100)} % de las calificaciones es grado de inversión (BBB- o mejor). "
            "Egan-Jones y S&P emiten siete de cada diez.")

    with t3:
        dup = df.duplicated(subset=["CIK"] + A.RATIOS)
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas idénticas", fmt(int(df.duplicated().sum()), 0))
        c2.metric("Filas que repiten empresa y razones", f"{fmt(int(dup.sum()), 0)} ({fmt(dup.mean() * 100)} %)")
        c3.metric("Estados financieros únicos", fmt(len(obs), 0))
        st.markdown(
            "No hay filas idénticas, pero dos de cada tres repiten la empresa y **todas** sus razones: es el mismo estado "
            "financiero calificado por otra agencia o en otra fecha. Si se dejaran, una empresa calificada trece veces "
            "pesaría trece veces más en el ACP y en los centros de K-Means.\n\n"
            "**Decisión:** una fila por CIK y juego de razones; su calificación es la **mediana** de las recibidas y su año "
            "el de la calificación más reciente.")
        ticker = st.selectbox("Ver las calificaciones de una empresa", sorted(df["Ticker"].unique()),
                              index=sorted(df["Ticker"].unique()).index("AVT"))
        ej = df[df["Ticker"] == ticker].sort_values("Rating Date")[
            ["Rating Agency", "Rating", "Rating Date", "Current Ratio", "Net Profit Margin", "ROA - Return On Assets"]]
        ej["Rating Date"] = ej["Rating Date"].dt.date
        tabla(ej, defecto=2, indice=False)

    with t4:
        st.markdown("Estadísticas de las 16 razones en las 2.694 observaciones, antes de cualquier transformación.")
        desc = obs[A.RATIOS].describe(percentiles=[.01, .25, .5, .75, .99]).T.drop(columns="count")
        desc["Asimetría"] = obs[A.RATIOS].skew()
        desc.index = [A.nombre(v) for v in desc.index]
        desc.columns = ["Media", "Desv. estándar", "Mínimo", "P1", "P25", "Mediana", "P75", "P99", "Máximo", "Asimetría"]
        tabla(desc, defecto=2)
        st.markdown(
            "Las colas son muy largas: el ROE va de −11.258 % a 7.038 % y la deuda LP / capital llega a 171 cuando su rango "
            "natural es 0 a 1. El 98 % central (P1 a P99) es razonable. Con esas colas, unas pocas empresas dominarían la "
            "varianza; por eso se winsoriza (pantalla 3).")

    with t5:
        c1, c2 = st.columns([1, 2.2])
        with c1:
            v = st.selectbox("Razón", A.RATIOS, index=A.RATIOS.index("Net Profit Margin"), format_func=A.nombre)
            recortar = st.toggle("Recortar al percentil 1 y 99 para ver la forma", value=True)
            color_por = st.radio("Separar por", ["Grado de inversión", "Sin separar"])
        serie = obs[v]
        if recortar:
            serie = serie.clip(serie.quantile(.01), serie.quantile(.99))
        base = pd.DataFrame({"valor": serie,
                             "Grado": np.where(obs["Grado de inversión"] == 1, "Grado de inversión", "Grado especulativo")})
        fig = px.histogram(base, x="valor", nbins=60, color="Grado" if color_por != "Sin separar" else None,
                           barmode="overlay", opacity=.65,
                           color_discrete_map={"Grado de inversión": VERDE, "Grado especulativo": VINO},
                           color_discrete_sequence=[VERDE], marginal="box")
        fig.update_xaxes(title=A.nombre(v))
        fig.update_yaxes(title="Empresas-periodo")
        with c2:
            mostrar(estilo(fig, 430, f"Distribución de {A.nombre(v).lower()}"))
        st.caption(f"Familia: {A.FAMILIAS[v]}. Mediana en grado de inversión: "
                   f"{fmt(obs.loc[obs['Grado de inversión'] == 1, v].median(), 2)}; en grado especulativo: "
                   f"{fmt(obs.loc[obs['Grado de inversión'] == 0, v].median(), 2)}.")

    with st.expander("Ver la base consolidada (una fila por empresa-periodo)"):
        sector = st.multiselect("Filtrar por sector", sorted(obs["Sector"].map(A.SECTORES).unique()))
        vista = obs.copy()
        vista["Sector"] = vista["Sector"].map(A.SECTORES)
        if sector:
            vista = vista[vista["Sector"].isin(sector)]
        st.dataframe(vista[["Empresa", "Ticker", "Sector", "Año", "Calificaciones", "Rating mediano"] + A.RATIOS],
                     hide_index=True)


# ---------------------------------------------------------------------------
# 3. Preparación
# ---------------------------------------------------------------------------

def pagina_preparacion():
    st.title("Preparación: qué entra al modelo y en qué escala")
    st.markdown(
        '<p class="lead">Tres decisiones: qué razones usar, cómo tratar los valores extremos y cómo poner '
        "todas las razones en la misma escala.</p>", unsafe_allow_html=True)

    st.header("Selección de variables")
    filas = []
    for v in A.RATIOS:
        filas.append({"Razón": A.nombre(v), "Familia": A.FAMILIAS[v],
                      "En el análisis": "Sí" if v in A.VARIABLES_MODELO else "No",
                      "En tu configuración": "Sí" if v in VARS else "No",
                      "Motivo de la exclusión": A.EXCLUIDAS.get(v, "")})
    st.dataframe(pd.DataFrame(filas), hide_index=True)
    st.markdown(
        "Las diez razones del análisis cubren las cinco familias que mira un comité de crédito: liquidez, apalancamiento, "
        "márgenes, eficiencia y rentabilidad. Las seis excluidas fallan por redundancia exacta, por un denominador que "
        "cambia de signo con patrimonio negativo o porque están expresadas por acción.")

    st.header("Valores extremos: winsorización")
    st.markdown(
        f"Con los parámetros actuales cada razón se recorta en los percentiles {fmt(Q * 100)} y {fmt(100 - Q * 100)}. "
        "La empresa se queda en la base con su posición relativa, pero deja de arrastrar la varianza. Borrar los "
        "extremos sacaría justo a las empresas más riesgosas.")
    lim = prep.limites.copy()
    lim["Valores recortados"] = prep.recortes
    lim["Asimetría antes"] = obs[list(VARS)].skew()
    lim["Asimetría después"] = prep.X_wins.skew()
    lim.index = [A.nombre(v) for v in lim.index]
    tabla(lim, decimales={"Valores recortados": 0})

    v = st.selectbox("Comparar antes y después", VARS, format_func=A.nombre,
                     index=list(VARS).index("Net Profit Margin") if "Net Profit Margin" in VARS else 0)
    comp = pd.concat([pd.DataFrame({"Valor": obs[v], "Serie": "Original"}),
                      pd.DataFrame({"Valor": prep.X_wins[v], "Serie": "Winsorizada"})])
    fig = px.box(comp, x="Valor", y="Serie", color="Serie", points="outliers",
                 color_discrete_map={"Original": GRIS, "Winsorizada": VERDE})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=A.nombre(v))
    fig.update_yaxes(title="")
    mostrar(estilo(fig, 300, f"{A.nombre(v)}: antes y después de recortar"))

    st.header("Correlaciones")
    corr = prep.X_wins.corr()
    corr.index = corr.columns = [A.nombre(c) for c in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=ESCALA_DIVERGENTE, zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(title="r"))
    mostrar(estilo(fig, 560, "Correlación de Pearson entre las razones del modelo (datos winsorizados)", leyenda_abajo=False))
    if ACP_DEFECTO:
        st.markdown(
            "- **Un bloque de rentabilidad.** Márgenes operativo, antes de impuestos y neto: 0,83 a 0,93. ROA y ROI: 0,92.\n"
            "- **Margen frente a rotación.** La rotación de activos se opone al margen bruto (−0,53) y al EBITDA (−0,48): "
            "se gana con margen o con volumen (identidad de DuPont).\n"
            "- **El balance va por su lado.** Razón corriente y deuda LP / capital casi no se correlacionan con los márgenes.\n\n"
            "Diez columnas, pero claramente menos de diez historias: eso justifica el ACP.")

    st.header("Escala")
    st.markdown(
        "Cada razón se estandariza a media 0 y desviación 1, lo que equivale a hacer el ACP sobre la matriz de "
        "correlaciones. Sin ese paso, los márgenes (en decenas de puntos porcentuales) dominarían a la rotación (en veces).")
    if normalizar:
        st.markdown('<div class="nota">Está activada la normalización de filas. Cada empresa queda como una dirección de '
                    "longitud 1: se conserva el estilo del perfil, pero se pierde su magnitud. Una empresa con margen "
                    "neto de −37 % y otra con −2 % pueden quedar juntas.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "**A diferencia del cuaderno de clase, no se normaliza cada fila.** Allí la pregunta era de estilo de uso de "
            "la tarjeta. Aquí es de nivel: un margen neto de −37 % es mucho peor que uno de −2 %, y esa magnitud es lo que "
            "importa para el cupo. Puedes activar la normalización en la barra lateral y ver cómo desaparece el grupo en pérdidas.")


# ---------------------------------------------------------------------------
# 4. ACP
# ---------------------------------------------------------------------------

def pagina_acp():
    st.title("Análisis de componentes principales")
    aviso_configuracion()
    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes conservados", n_comp)
    c2.metric("Varianza conservada", f"{fmt(var_conservada)} %")
    c3.metric("Varianza del plano CP1–CP2", f"{fmt(var_tab['Acumulada (%)'].iloc[1])} %")

    st.header("Varianza explicada")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=var_tab["Componente"], y=var_tab["Varianza explicada (%)"], name="Varianza de cada componente",
                    marker_color=[VERDE if i < n_comp else "#C9D1CC" for i in range(len(var_tab))])
        fig.add_scatter(x=var_tab["Componente"], y=var_tab["Acumulada (%)"], name="Varianza acumulada",
                        mode="lines+markers", line=dict(color=TINTA), yaxis="y2")
        fig.update_layout(yaxis=dict(title="Varianza explicada (%)"),
                          yaxis2=dict(title="Acumulada (%)", overlaying="y", side="right", range=[0, 105],
                                      showgrid=False))
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=80, y1=80,
                      line=dict(color=VINO, dash="dash", width=1))
        fig.add_annotation(xref="paper", x=1, yref="y2", y=80, text="80 %", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(color=VINO))
        mostrar(estilo(fig, 420, "Gráfico de sedimentación"))
    with c2:
        vt = var_tab.set_index("Componente")
        tabla(vt, defecto=2)

    criterios = pd.DataFrame({
        "Criterio": ["Kaiser (valor propio mayor que el promedio)", "Varianza acumulada de al menos 80 %",
                     "Configuración actual"],
        "Componentes": [acp["kaiser"], acp["n_80"], n_comp],
        "Varianza conservada (%)": [var_tab["Acumulada (%)"].iloc[acp["kaiser"] - 1],
                                    var_tab["Acumulada (%)"].iloc[acp["n_80"] - 1], var_conservada],
    })
    tabla(criterios, decimales={"Componentes": 0}, indice=False)
    if ACP_DEFECTO:
        st.markdown(
            "**Decisión del análisis: tres componentes (78,7 %).** Kaiser y el codo coinciden en tres: el cuarto tiene un "
            "valor propio de 0,84, menor que el de una razón original, y su lectura (liquidez y deuda con el mismo signo) es "
            "más sectorial que crediticia. Para pasar el 80 % harían falta cuatro (87,1 %); agrupar con cuatro produce casi "
            "los mismos grupos (índice de Rand ajustado 0,98).")

    st.header("Cargas: de qué está hecho cada eje")
    n_ver = min(max(n_comp, 4), len(VARS))
    car = acp["cargas"].iloc[:, :n_ver].copy()
    car.index = [A.nombre(v) for v in car.index]
    fig = px.imshow(car, text_auto=".2f", color_continuous_scale=ESCALA_DIVERGENTE, zmin=-.8, zmax=.8, aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(title="Carga"))
    c1, c2 = st.columns([1.1, 1])
    with c1:
        mostrar(estilo(fig, 470, "Cargas de las razones", leyenda_abajo=False))
    with c2:
        lecturas_def = {
            0: "**CP1, rentabilidad.** Todas las cargas grandes son de rentabilidad: márgenes antes de impuestos, "
               "operativo y neto (0,42 a 0,44), ROA, EBITDA y ROI. Liquidez y deuda casi no pesan.",
            1: "**CP2, rotación frente a margen.** Sube con la rotación de activos (0,60), ROI y ROA; baja con el margen "
               "bruto (−0,50) y el EBITDA. Arriba, quien gana por volumen (comercio); abajo, quien gana por margen "
               "(servicios públicos, salud). Es la identidad de DuPont.",
            2: "**CP3, solidez de balance.** Sube con la razón corriente (0,73) y baja con la deuda LP / capital (−0,64). "
               "Arriba, empresas líquidas y poco endeudadas.",
        }
        for i in range(n_comp):
            if ACP_DEFECTO and i in lecturas_def:
                st.markdown(lecturas_def[i])
            else:
                st.markdown(f"**CP{i + 1}** ({fmt(var_tab['Varianza explicada (%)'].iloc[i])} %): "
                            f"{A.leer_componente(acp['cargas'].iloc[:, i])}.")

    st.header("Las empresas en el espacio de los componentes")
    c1, c2, c3 = st.columns(3)
    opciones = list(range(len(VARS)))
    ex = c1.selectbox("Eje horizontal", opciones, index=0, format_func=lambda i: f"CP{i + 1}")
    ey = c2.selectbox("Eje vertical", opciones, index=1, format_func=lambda i: f"CP{i + 1}")
    color = c3.selectbox("Color", ["Grado de inversión", "Sector", "Sin color"])
    P = acp["puntajes"]
    base = pd.DataFrame({"x": P[:, ex], "y": P[:, ey], "Empresa": obs["Etiqueta"], "Rating": obs["Rating mediano"],
                         "Grado de inversión": np.where(obs["Grado de inversión"] == 1, "Grado de inversión", "Grado especulativo"),
                         "Sector": obs["Sector"].map(A.SECTORES)})
    kw = dict(color_discrete_map={"Grado de inversión": VERDE, "Grado especulativo": VINO})
    fig = px.scatter(base, x="x", y="y", color=None if color == "Sin color" else color, opacity=.5,
                     hover_data={"Empresa": True, "Rating": True, "x": ":.2f", "y": ":.2f"},
                     color_discrete_sequence=[VERDE] if color == "Sin color" else px.colors.qualitative.Safe, **kw)
    fig.update_traces(marker=dict(size=6))
    # Flechas de las cargas (biplot)
    esc = 0.9 * min(np.abs(P[:, ex]).max(), np.abs(P[:, ey]).max())
    for vname in VARS:
        lx, ly = acp["cargas"].loc[vname].iloc[ex] * esc, acp["cargas"].loc[vname].iloc[ey] * esc
        fig.add_annotation(x=lx, y=ly, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=1.4, arrowcolor=TINTA, text="")
        fig.add_annotation(x=lx * 1.08, y=ly * 1.08, text=A.nombre(vname), showarrow=False,
                           font=dict(size=11, color=TINTA), bgcolor="rgba(255,255,255,.75)")
    fig.update_xaxes(title=etiqueta_cp(ex))
    fig.update_yaxes(title=etiqueta_cp(ey))
    plano = var_tab["Varianza explicada (%)"].iloc[[ex, ey]].sum() if ex != ey else var_tab["Varianza explicada (%)"].iloc[ex]
    mostrar(estilo(fig, 600, f"Biplot: el plano resume el {fmt(plano)} % de la varianza"))
    if ACP_DEFECTO and (ex, ey) == (0, 1):
        st.markdown(
            "El color es la calificación, que no participó en el ACP. Las empresas con grado de inversión se concentran a "
            "la derecha (más rentables) y las de grado especulativo dominan la cola izquierda, donde están las empresas en "
            "pérdidas. En el centro hay mucha superposición: la rentabilidad explica parte de la calificación, no toda.")


# ---------------------------------------------------------------------------
# 5. K-Means
# ---------------------------------------------------------------------------

def pagina_kmeans():
    st.title("Segmentación con K-Means")
    aviso_configuracion()
    st.markdown(
        f'<p class="lead">K-Means agrupa a las empresas en el espacio de los {n_comp} componentes conservados '
        f"({fmt(var_conservada)} % de la varianza), no sobre las razones originales: así cada dimensión cuenta una vez "
        "y los cinco márgenes no pesan cinco veces.</p>", unsafe_allow_html=True)

    st.header("Cuántos grupos: codo y silueta")
    ev = evaluar(VARS, Q, normalizar, n_comp, semilla)
    k_sil = int(ev.loc[ev["Silueta media"].idxmax(), "k"])
    c1, c2 = st.columns(2)
    fig = px.line(ev, x="k", y="Inercia", markers=True, color_discrete_sequence=[VERDE])
    fig.add_scatter(x=[k], y=[ev.set_index("k").loc[k, "Inercia"]], mode="markers", name=f"k elegido = {k}",
                    marker=dict(size=15, color=VINO))
    fig.update_xaxes(title="Número de grupos (k)", dtick=1)
    fig.update_yaxes(title="Suma de cuadrados dentro de los grupos")
    with c1:
        mostrar(estilo(fig, 380, "Método del codo"))
    fig = px.line(ev, x="k", y="Silueta media", markers=True, color_discrete_sequence=[OCRE])
    fig.add_scatter(x=[k], y=[ev.set_index("k").loc[k, "Silueta media"]], mode="markers", name=f"k elegido = {k}",
                    marker=dict(size=15, color=VINO))
    fig.update_xaxes(title="Número de grupos (k)", dtick=1)
    fig.update_yaxes(title="Silueta media")
    with c2:
        mostrar(estilo(fig, 380, f"Silueta: el máximo está en k = {k_sil}"))
    tabla(ev.set_index("k"), decimales={"Inercia": 0, "Silueta media": 3, "Grupo más pequeño": 0, "Caída de inercia (%)": 1})
    if TODO_DEFECTO:
        st.markdown(
            "**Decisión: k = 3.** La silueta alcanza su máximo (0,313) y el cambio de pendiente más fuerte del codo ocurre "
            "al pasar de dos a tres grupos (la inercia cae 28,4 %, luego 16,9 %). Con dos grupos, las empresas en pérdidas "
            "se mezclan con las de margen estrecho y se pierde el grupo más riesgoso. El tercer grupo es pequeño (84), "
            "pero es el que más importa separar para el cupo. Con otras diez semillas la partición casi no cambia (ARI ≥ 0,98).")
    elif k != k_sil:
        st.info(f"Con tus parámetros la silueta máxima está en k = {k_sil}; estás viendo k = {k}.")

    st.header("Los grupos")
    resumen = pd.DataFrame({"Segmento": segmento, "Silueta": km["silueta"]}).groupby("Segmento").agg(
        **{"Empresas-periodo": ("Silueta", "size"), "Silueta media": ("Silueta", "mean")}).loc[orden]
    resumen["Participación (%)"] = resumen["Empresas-periodo"] / len(obs) * 100
    cen = pd.DataFrame(km["modelo"].cluster_centers_, columns=[f"Centro CP{i + 1}" for i in range(n_comp)],
                       index=[nombres[g] for g in range(k)])
    tabla(resumen.join(cen), decimales={"Empresas-periodo": 0, "Silueta media": 3, "Participación (%)": 1})
    st.caption("Cada grupo se nombra por su perfil frente al promedio de la base (márgenes, rotación y, si se aparta mucho, "
               "deuda o liquidez). El número interno de K-Means no tiene orden ni significado y no se muestra.")

    c1, c2, c3 = st.columns([1, 1, 1])
    ex = c1.selectbox("Eje horizontal ", list(range(n_comp)), index=0, format_func=lambda i: f"CP{i + 1}")
    ey = c2.selectbox("Eje vertical ", list(range(n_comp)), index=1, format_func=lambda i: f"CP{i + 1}")
    ver3d = c3.toggle("Ver en 3D", value=False, disabled=n_comp < 3)
    if ver3d and n_comp >= 3:
        base = pd.DataFrame({"CP1": S[:, 0], "CP2": S[:, 1], "CP3": S[:, 2], "Segmento": segmento,
                             "Empresa": obs["Etiqueta"]})
        fig = px.scatter_3d(base, x="CP1", y="CP2", z="CP3", color="Segmento", color_discrete_map=colores,
                            category_orders={"Segmento": orden}, opacity=.6, hover_data={"Empresa": True})
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(scene=dict(xaxis_title=etiqueta_cp(0), yaxis_title=etiqueta_cp(1), zaxis_title=etiqueta_cp(2)))
        mostrar(estilo(fig, 650, f"Los grupos en los tres primeros componentes ({fmt(var_tab['Acumulada (%)'].iloc[2])} % de la varianza)"))
    else:
        plano = var_tab["Varianza explicada (%)"].iloc[[ex, ey]].sum() if ex != ey else var_tab["Varianza explicada (%)"].iloc[ex]
        fig = mapa_grupos((ex, ey))
        fig.update_layout(title=f"Mapa de los grupos: este plano resume el {fmt(plano)} % de la varianza")
        mostrar(fig)
    if TODO_DEFECTO:
        st.markdown(
            "En el plano CP1–CP2 los grupos se ordenan de izquierda a derecha por rentabilidad, y la frontera entre los dos "
            "grandes es diagonal. En el plano CP1–CP3 los tres grupos ocupan todo el eje de solidez de balance: **la deuda y "
            "la liquidez no forman grupos; varían dentro de cada perfil.**")

    st.header("Qué tan nítidos son los grupos")
    sil = pd.DataFrame({"Segmento": segmento, "Silueta": km["silueta"]})
    fig = go.Figure()
    y0, ticks, textos = 0, [], []
    for seg in orden[::-1]:
        v = np.sort(sil.loc[sil["Segmento"] == seg, "Silueta"].values)
        y = np.arange(y0, y0 + len(v))
        fig.add_scatter(x=v, y=y, fill="tozerox", mode="lines", line=dict(color=colores[seg], width=1),
                        fillcolor=colores[seg], name=seg, hoverinfo="skip")
        ticks.append(y0 + len(v) / 2)
        textos.append(f"{seg} (n = {fmt(len(v), 0)})")
        y0 += len(v) + 50
    fig.add_vline(x=km["silueta_media"], line_dash="dash", line_color=TINTA,
                  annotation_text=f"Silueta media = {fmt(km['silueta_media'], 3)}", annotation_position="top right")
    fig.update_yaxes(tickvals=ticks, ticktext=textos, showgrid=False)
    fig.update_xaxes(title="Coeficiente de silueta de cada empresa-periodo")
    fig.update_layout(showlegend=False)
    c1, c2 = st.columns([1.6, 1])
    with c1:
        mostrar(estilo(fig, 430, "Diagrama de silueta"))
    with c2:
        st.metric("Silueta media", fmt(km["silueta_media"], 3))
        st.metric("Empresas con silueta negativa", f"{fmt((km['silueta'] < 0).mean() * 100)} %")
        st.metric("Empresas en la frontera (silueta < 0,1)", f"{fmt((km['silueta'] < .1).mean() * 100)} %")
    if TODO_DEFECTO:
        st.markdown(
            "Una silueta media de 0,31 indica estructura débil a moderada. El grupo en pérdidas es el más nítido (0,43); "
            "los dos grandes comparten una frontera ancha. El 12,2 % de las empresas está por debajo de 0,1: para ellas la "
            "etiqueta del grupo es una referencia débil y pesa más el análisis individual.")

    st.header("Perfil de cada grupo con las razones originales")
    perfil = A.perfilar_segmentos(obs, etiquetas, nombres, list(VARS))[orden]
    tabla(perfil, defecto=2, filas_enteras=["Empresas-periodo"])
    Zdf = pd.DataFrame(prep.escalador.transform(prep.X_wins), columns=VARS).assign(Segmento=segmento.values)
    mz = Zdf.groupby("Segmento").mean().loc[orden]
    mz.columns = [A.nombre(c) for c in mz.columns]
    fig = px.imshow(mz.T, text_auto=".2f", color_continuous_scale=ESCALA_DIVERGENTE, zmin=-2, zmax=2, aspect="auto")
    fig.update_layout(coloraxis_colorbar=dict(title="Desv."))
    mostrar(estilo(fig, 470, "Cuánto se aparta cada grupo del promedio (en desviaciones estándar)", leyenda_abajo=False))
    if TODO_DEFECTO:
        st.markdown(
            "- **Margen alto, rotación baja (39 %).** Margen operativo mediano de 21 % y neto de 13 %, con rotación de 0,43: "
            "venden poco por activo pero cada venta deja mucho. ROA de 6,2 %.\n"
            "- **Margen estrecho, rotación alta (58 %).** Margen operativo de 7,6 % y neto de 3,8 %, compensado con una "
            "rotación de 0,95. Su ROA (4,0 %) no queda tan atrás: ganan por volumen.\n"
            "- **En pérdidas operativas (3 %).** Margen operativo de −33 %, ROA de −18 %. Su margen bruto (43 %) no es malo: "
            "el problema está en los gastos que la venta no cubre.\n\n"
            "**La deuda no separa a los grupos:** la deuda LP / capital mediana es 0,43, 0,42 y 0,46.")

    st.header("Validación externa con la calificación")
    st.markdown("La calificación no entró al modelo. Ahora se cruza con los grupos:")
    tabla(validacion.loc[orden], decimales={"Empresas-periodo": 0, "Puntaje mediano": 1, "Grado de inversión (%)": 1})
    fam = obs["Puntaje"].apply(lambda p: "AAA a AA-" if p <= 4 else "A+ a A-" if p <= 7 else "BBB+ a BBB-" if p <= 10
                               else "BB+ a BB-" if p <= 13 else "B+ a B-" if p <= 16 else "CCC+ o menor")
    FAM = ["AAA a AA-", "A+ a A-", "BBB+ a BBB-", "BB+ a BB-", "B+ a B-", "CCC+ o menor"]
    ct = (pd.crosstab(segmento, fam, normalize="index") * 100).reindex(columns=FAM).fillna(0).loc[orden[::-1]]
    largo = ct.reset_index().melt(id_vars="Segmento", var_name="Calificación", value_name="Porcentaje")
    fig = px.bar(largo, y="Segmento", x="Porcentaje", color="Calificación", orientation="h",
                 category_orders={"Calificación": FAM},
                 color_discrete_sequence=["#16352B", "#3C7A65", "#9BBFAE", "#E3C285", "#C0705F", "#7A2328"],
                 text=largo["Porcentaje"].map(lambda v: f"{v:.0f}%" if v >= 6 else ""))
    fig.update_xaxes(title="Porcentaje de empresas-periodo del grupo", range=[0, 100])
    fig.update_yaxes(title="")
    mostrar(estilo(fig, 330, "Calificación mediana dentro de cada grupo"))
    if TODO_DEFECTO:
        st.markdown(
            "Formados solo con razones financieras, los grupos **ordenan la calificación**: BBB+, BB+ y B+, con 73 %, 48 % y "
            "23 % de grado de inversión. Seis escalones separan al primero del tercero. Cada grupo contiene calificaciones de "
            "casi todo el rango: las agencias miran también tamaño, sector y posición competitiva.")

    with st.expander("Composición sectorial de los grupos"):
        sec = (pd.crosstab(obs["Sector"].map(A.SECTORES), segmento, normalize="index") * 100)[orden]
        fig = px.imshow(sec, text_auto=".0f", color_continuous_scale=["#F4F2EE", VERDE], aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title="%"))
        mostrar(estilo(fig, 470, "Porcentaje de las empresas de cada sector en cada grupo", leyenda_abajo=False))

    with st.expander("Buscar empresas dentro de un grupo"):
        seg = st.selectbox("Grupo", orden)
        lista = obs.assign(Segmento=segmento, Silueta=km["silueta"])
        lista = lista[lista["Segmento"] == seg].sort_values("Silueta")
        lista["Sector"] = lista["Sector"].map(A.SECTORES)
        st.caption("Ordenadas de menor a mayor silueta: las primeras están en la frontera con otro grupo.")
        tabla(lista[["Empresa", "Ticker", "Año", "Sector", "Rating mediano", "Silueta"] + list(VARS)],
              decimales={"Año": 0, "Silueta": 3}, indice=False)


# ---------------------------------------------------------------------------
# 6. Simulador de cupo
# ---------------------------------------------------------------------------

def _cargar_valores(fuente_vals: dict):
    for v, x in fuente_vals.items():
        st.session_state[f"sim_{v}"] = float(round(x, 4))


def pagina_simulador():
    st.title("Simulador de cupo: ¿a qué grupo se parece una empresa?")
    st.markdown(
        '<p class="lead">Ingresa las razones de una empresa o carga una de la base. Se le aplican los mismos recortes, '
        "la misma estandarización y la misma proyección del ACP, y se busca el centro de K-Means más cercano.</p>",
        unsafe_allow_html=True)
    aviso_configuracion()

    medianas = obs[list(VARS)].median()
    for v in VARS:
        if f"sim_{v}" not in st.session_state:
            st.session_state[f"sim_{v}"] = float(round(medianas[v], 4))

    opciones = ["Mediana de toda la base"] + [f"Mediana del grupo: {s}" for s in orden] + list(obs["Etiqueta"].sort_values())
    c1, c2 = st.columns([3, 1])
    elegido = c1.selectbox("Cargar valores de", opciones, index=0)
    if c2.button("Cargar", type="primary"):
        if elegido == "Mediana de toda la base":
            _cargar_valores(medianas.to_dict())
        elif elegido.startswith("Mediana del grupo: "):
            s = elegido.replace("Mediana del grupo: ", "")
            _cargar_valores(obs.loc[segmento == s, list(VARS)].median().to_dict())
        else:
            _cargar_valores(obs.loc[obs["Etiqueta"] == elegido, list(VARS)].iloc[0].to_dict())
        st.rerun()

    st.subheader("Razones de la empresa")
    cols = st.columns(4)
    valores = {}
    for i, v in enumerate(VARS):
        paso = 0.05 if A.FAMILIAS[v] in ("Liquidez", "Apalancamiento", "Eficiencia") else 0.5
        valores[v] = cols[i % 4].number_input(A.nombre(v), step=paso, format="%.2f", key=f"sim_{v}")

    u = A.ubicar_empresa(prep, acp, n_comp, km, valores)
    seg = nombres[u["grupo"]]
    pct_gi = validacion.loc[seg, "Grado de inversión (%)"]
    postura, detalle = A.lectura_de_cupo(pct_gi)

    st.header("Resultado")
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        st.markdown(
            f'<div class="ficha" style="--c:{colores[seg]}"><div class="t">{seg}</div>'
            f'<div class="s">Postura sugerida: <b>{postura}</b>. {detalle}</div>'
            f'<div class="s">En este grupo, {fmt(pct_gi)} % de las empresas tiene grado de inversión; su rating mediano '
            f'es {validacion.loc[seg, "Rating mediano"]}.</div></div>', unsafe_allow_html=True)
        segundo = nombres[u["segundo"]] if u["segundo"] is not None else "–"
        st.markdown(" ")
        if u["margen"] < 0.15:
            st.warning(f"La empresa está en la frontera con **{segundo}** (el centro más cercano está solo "
                       f"{fmt(u['margen'] * 100, 0)} % más cerca que el segundo). Conviene análisis individual.")
        else:
            st.markdown(f"La asignación es clara: el centro de su grupo está {fmt(u['margen'] * 100, 0)} % más cerca que "
                        f"el del segundo más cercano (**{segundo}**).")

        # Ajuste por solidez de balance, si existe un componente de balance
        j = A.componente_balance(acp["cargas"], n_comp)
        if j is not None:
            signo = 1 if acp["cargas"].iloc[:, j]["Current Ratio"] > 0 else -1
            tab_b, cortes = A.tercios_balance(obs, S[:, j], etiquetas, nombres, signo)
            valor = u["puntajes"][j] * signo
            c13, c23 = cortes.loc[seg]
            tercio = A.TERCIOS[0] if valor < c13 else (A.TERCIOS[1] if valor < c23 else A.TERCIOS[2])
            gi_t = tab_b.loc[seg, tercio]
            post_t = A.lectura_de_cupo(gi_t)[0]
            st.markdown(
                f"**Solidez de balance (CP{j + 1}).** Frente a las demás empresas de su grupo, está en el "
                f"**{tercio.lower()}**. En esa franja, {fmt(gi_t)} % tiene grado de inversión, lo que con la misma regla "
                f"corresponde a **{post_t.lower()}**.")
            if A.TERCIOS.index(tercio) == 0:
                st.markdown("Dentro de su grupo, es de las primeras empresas que conviene revisar.")
        st.caption("Regla de cupo: dos tercios o más de grado de inversión en el grupo → amplio; entre un tercio y dos "
                   "tercios → condicionado; menos de un tercio → restringido. Es una traducción ilustrativa, no un modelo "
                   "de riesgo ni un monto en dinero.")
    with c2:
        dist = pd.DataFrame({"Grupo": [nombres[g] for g in range(k)], "Distancia": u["distancias"]}).sort_values("Distancia")
        fig = px.bar(dist, x="Distancia", y="Grupo", orientation="h", color="Grupo", color_discrete_map=colores)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title="", autorange="reversed")
        fig.update_xaxes(title="Distancia al centro en el espacio del ACP")
        mostrar(estilo(fig, 260, "Distancia a cada centro"))
        pos = pd.DataFrame({"Componente": [etiqueta_cp(i) for i in range(n_comp)], "Empresa": u["puntajes"],
                            "Centro de su grupo": km["modelo"].cluster_centers_[u["grupo"]]})
        tabla(pos.set_index("Componente"), defecto=2)

    estrella = go.Scatter(x=[u["puntajes"][0]], y=[u["puntajes"][1]], mode="markers", name="Empresa simulada",
                          marker=dict(symbol="star", size=24, color="#F2C14E", line=dict(width=2, color=TINTA)))
    fig = mapa_grupos((0, 1), 520, extra=estrella)
    fig.update_layout(title="Dónde cae la empresa en el plano CP1–CP2")
    mostrar(fig)


# ---------------------------------------------------------------------------
# 7. Conclusiones
# ---------------------------------------------------------------------------

def pagina_conclusiones():
    st.title("Qué aprendimos de los datos")
    aviso_configuracion()
    c = st.columns(4)
    c[0].metric("Varianza conservada", f"{fmt(var_conservada)} %", help=f"{n_comp} componentes")
    c[1].metric("Grupos", k)
    c[2].metric("Silueta media", fmt(km["silueta_media"], 3))
    c[3].metric("Rating mediano del mejor y el peor grupo",
                f"{validacion['Rating mediano'].iloc[0]} a {validacion['Rating mediano'].iloc[-1]}")
    tira_rating(validacion)

    st.header("Hallazgos")
    st.markdown(
        "1. **Diez razones caben en tres ejes con sentido económico.** Rentabilidad (47,9 %), rotación frente a margen "
        "(19,1 %) y solidez de balance (11,6 %) conservan el 78,7 % de la variación. Son las dimensiones que usa un "
        "comité de crédito, pero esta vez las propusieron los datos.\n"
        "2. **Hay tres perfiles, con fronteras anchas.** Margen alto y rotación baja (39 %), margen estrecho y rotación "
        "alta (58 %) y en pérdidas operativas (3 %). La silueta máxima (0,31) está en k = 3 y la partición es estable "
        "frente a la semilla.\n"
        "3. **Los perfiles ordenan la calificación sin haberla visto.** BBB+, BB+ y B+, con 73 %, 48 % y 23 % de grado "
        "de inversión. Es la evidencia de que la segmentación captura capacidad de pago.\n"
        "4. **La deuda no forma grupos, pero ordena dentro de ellos.** La deuda LP / capital mediana es casi igual en los "
        "tres grupos; aun así, dentro del grupo en pérdidas la proporción con grado de inversión pasa de 7 % en el tercio "
        "de balance más débil a 36 % en el más sólido.\n"
        "5. **Lectura de cupo.** El perfil fija la postura (amplio, condicionado, restringido) y el balance señala a quién "
        "revisar primero dentro de cada grupo.")

    st.header("Limitaciones")
    st.markdown(
        "- **No es un modelo de riesgo.** K-Means describe semejanza; no estima probabilidad de incumplimiento ni un monto de cupo.\n"
        "- **La muestra no es el cliente típico de un banco local:** empresas grandes, listadas en Estados Unidos, 2010–2016.\n"
        "- **Las decisiones de preparación mueven el resultado.** Normalizar filas (como en clase) o no winsorizar cambia los "
        "grupos: pruébalo en la barra lateral.\n"
        "- **Faltan dimensiones del cupo:** tamaño, cobertura de intereses, deuda de corto plazo, flujo de caja comparable.\n"
        "- **Observaciones repetidas:** una empresa aparece en varios años y sus observaciones no son independientes.")

    st.header("Uso responsable")
    st.markdown(
        "El resultado sirve para priorizar: decidir a quién mirar con más cuidado y con qué política de partida. La decisión "
        "final de cupo sigue requiriendo el análisis individual, sobre todo para el 12 % de empresas que están en la frontera "
        "entre grupos.")


# ---------------------------------------------------------------------------

{
    "1. Contexto": pagina_contexto,
    "2. Datos": pagina_datos,
    "3. Preparación": pagina_preparacion,
    "4. ACP": pagina_acp,
    "5. K-Means": pagina_kmeans,
    "6. Simulador de cupo": pagina_simulador,
    "7. Conclusiones": pagina_conclusiones,
}[pagina]()
