"""
analisis.py
-----------
Funciones del taller "ACP y K-Means aplicados a la capacidad de cupo de empresas
calificadas". Tanto el notebook como la aplicación de Streamlit llaman a estas
mismas funciones, así que cada tabla y cada gráfico se recalcula a partir de los
datos crudos.

Flujo:
    cargar_datos -> construir_observaciones -> preparar_matriz
    -> ajustar_acp -> evaluar_k -> ajustar_kmeans -> perfilar_segmentos
    -> lectura_de_cupo / ubicar_empresa
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

# ---------------------------------------------------------------------------
# Diccionarios
# ---------------------------------------------------------------------------

RATIOS = [
    "Current Ratio",
    "Long-term Debt / Capital",
    "Debt/Equity Ratio",
    "Gross Margin",
    "Operating Margin",
    "EBIT Margin",
    "EBITDA Margin",
    "Pre-Tax Profit Margin",
    "Net Profit Margin",
    "Asset Turnover",
    "ROE - Return On Equity",
    "Return On Tangible Equity",
    "ROA - Return On Assets",
    "ROI - Return On Investment",
    "Operating Cash Flow Per Share",
    "Free Cash Flow Per Share",
]

NOMBRES = {
    "Current Ratio": "Razón corriente",
    "Long-term Debt / Capital": "Deuda LP / capital",
    "Debt/Equity Ratio": "Deuda / patrimonio",
    "Gross Margin": "Margen bruto (%)",
    "Operating Margin": "Margen operativo (%)",
    "EBIT Margin": "Margen EBIT (%)",
    "EBITDA Margin": "Margen EBITDA (%)",
    "Pre-Tax Profit Margin": "Margen antes de impuestos (%)",
    "Net Profit Margin": "Margen neto (%)",
    "Asset Turnover": "Rotación de activos",
    "ROE - Return On Equity": "ROE (%)",
    "Return On Tangible Equity": "Rentab. patrimonio tangible (%)",
    "ROA - Return On Assets": "ROA (%)",
    "ROI - Return On Investment": "ROI (%)",
    "Operating Cash Flow Per Share": "Flujo operativo por acción",
    "Free Cash Flow Per Share": "Flujo libre por acción",
}

FAMILIAS = {
    "Current Ratio": "Liquidez",
    "Long-term Debt / Capital": "Apalancamiento",
    "Debt/Equity Ratio": "Apalancamiento",
    "Gross Margin": "Márgenes",
    "Operating Margin": "Márgenes",
    "EBIT Margin": "Márgenes",
    "EBITDA Margin": "Márgenes",
    "Pre-Tax Profit Margin": "Márgenes",
    "Net Profit Margin": "Márgenes",
    "Asset Turnover": "Eficiencia",
    "ROE - Return On Equity": "Rentabilidad",
    "Return On Tangible Equity": "Rentabilidad",
    "ROA - Return On Assets": "Rentabilidad",
    "ROI - Return On Investment": "Rentabilidad",
    "Operating Cash Flow Per Share": "Flujo de caja",
    "Free Cash Flow Per Share": "Flujo de caja",
}

# Variables que entran al modelo por defecto (10 razones).
VARIABLES_MODELO = [
    "Current Ratio",
    "Long-term Debt / Capital",
    "Gross Margin",
    "Operating Margin",
    "EBITDA Margin",
    "Pre-Tax Profit Margin",
    "Net Profit Margin",
    "Asset Turnover",
    "ROA - Return On Assets",
    "ROI - Return On Investment",
]

# Variables que se dejan fuera y por qué.
EXCLUIDAS = {
    "EBIT Margin": "Coincide con el margen operativo en el 97,4 % de las "
    "observaciones (correlación 0,9997). Entrar con las dos duplicaría el mismo dato.",
    "Debt/Equity Ratio": "Tiene el patrimonio en el denominador. Cuando el "
    "patrimonio es negativo (112 observaciones, 4,2 %) el signo se invierte y una "
    "empresa insolvente aparece como poco endeudada.",
    "ROE - Return On Equity": "Mismo problema de signo: 29 de las 112 empresas con "
    "patrimonio negativo muestran ROE positivo con margen neto negativo. ROA y ROI "
    "miden la rentabilidad sin depender del patrimonio.",
    "Return On Tangible Equity": "Mismo problema de signo, más extremo "
    "(valores entre −14.023 % y 9.883 %).",
    "Operating Cash Flow Per Share": "Está expresada por acción: depende del número "
    "de acciones en circulación y no es comparable entre empresas.",
    "Free Cash Flow Per Share": "Está expresada por acción: depende del número de "
    "acciones en circulación y no es comparable entre empresas.",
}

ESCALA_RATING = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10, "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16, "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC+": 20, "CC": 20, "C": 21, "D": 22,
}
# Etiqueta legible para cada puntaje (se usa la notación S&P / Fitch).
PUNTAJE_A_RATING = {
    1: "AAA", 2: "AA+", 3: "AA", 4: "AA-", 5: "A+", 6: "A", 7: "A-",
    8: "BBB+", 9: "BBB", 10: "BBB-", 11: "BB+", 12: "BB", 13: "BB-",
    14: "B+", 15: "B", 16: "B-", 17: "CCC+", 18: "CCC", 19: "CCC-",
    20: "CC", 21: "C", 22: "D",
}
UMBRAL_GRADO_INVERSION = 10  # BBB- o mejor

SECTORES = {
    "BusEq": "Equipos de negocio y tecnología",
    "Chems": "Químicos",
    "Durbl": "Bienes durables",
    "Enrgy": "Energía",
    "Hlth": "Salud",
    "Manuf": "Manufactura",
    "Money": "Finanzas",
    "NoDur": "Bienes de consumo no durables",
    "Other": "Otros",
    "Shops": "Comercio",
    "Telcm": "Telecomunicaciones",
    "Utils": "Servicios públicos",
}


def nombre(var: str) -> str:
    """Nombre en español de una razón financiera."""
    return NOMBRES.get(var, var)


def rating_desde_puntaje(p: float) -> str:
    """Convierte un puntaje (1 = AAA, 22 = D) en la letra más cercana."""
    if pd.isna(p):
        return "s. d."
    return PUNTAJE_A_RATING[int(np.clip(round(p), 1, 22))]


# ---------------------------------------------------------------------------
# 1. Carga robusta del CSV
# ---------------------------------------------------------------------------

def cargar_datos(ruta: str) -> tuple[pd.DataFrame, dict]:
    """
    Lee el CSV corrigiendo los dos defectos de formato que trae el archivo:

    * 1.685 líneas vienen enteras entre comillas, con las comillas internas
      duplicadas. pandas las lee como una sola columna.
    * 22 líneas traen la agencia "Japan Credit Rating Agency,Ltd." con una coma
      sin comillas, lo que produce 26 campos en vez de 25.

    Devuelve el DataFrame con tipos numéricos y un registro de las correcciones.
    """
    with open(ruta, newline="", encoding="utf-8") as f:
        lineas = f.read().splitlines()

    encabezado = next(csv.reader([lineas[0]]))
    filas, registro = [], Counter()
    for linea in lineas[1:]:
        if not linea.strip():
            registro["líneas vacías"] += 1
            continue
        campos = next(csv.reader([linea]))
        if len(campos) == 1:  # línea completa entre comillas
            campos = next(csv.reader([campos[0]]))
            registro["líneas entre comillas reparadas"] += 1
        if len(campos) == len(encabezado) + 1 and campos[1].strip() == "Ltd.":
            campos = [campos[0] + "," + campos[1]] + campos[2:]
            registro["coma en el nombre de la agencia reparada"] += 1
        if len(campos) != len(encabezado):
            registro["líneas descartadas"] += 1
            continue
        filas.append(campos)

    df = pd.DataFrame(filas, columns=encabezado)
    for c in RATIOS + ["CIK", "Binary Rating", "SIC Code"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Rating Date"] = pd.to_datetime(df["Rating Date"], errors="coerce")
    df["Puntaje rating"] = df["Rating"].map(ESCALA_RATING)
    registro["filas leídas"] = len(df)
    return df, dict(registro)


# ---------------------------------------------------------------------------
# 2. Unidad de análisis: un estado financiero por empresa
# ---------------------------------------------------------------------------

def construir_observaciones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cada fila del archivo es una calificación. Una misma empresa y un mismo
    juego de razones aparece varias veces cuando la califican varias agencias o
    en varias fechas. Para que ninguna empresa pese más por tener más
    calificaciones, se deja una fila por (CIK, razones) y se resume su
    calificación con la mediana de las recibidas.
    """
    claves = ["CIK"] + RATIOS
    obs = (
        df.groupby(claves, as_index=False, sort=False)
        .agg(
            Empresa=("Corporation", "last"),
            Ticker=("Ticker", "first"),
            Sector=("Sector", "first"),
            Fecha=("Rating Date", "max"),
            Calificaciones=("Rating", "size"),
            Puntaje=("Puntaje rating", "median"),
        )
    )
    obs["Rating mediano"] = obs["Puntaje"].apply(rating_desde_puntaje)
    obs["Grado de inversión"] = (obs["Puntaje"] <= UMBRAL_GRADO_INVERSION).astype(int)
    obs["Año"] = obs["Fecha"].dt.year
    obs["Etiqueta"] = (
        obs["Empresa"].str.strip() + " (" + obs["Ticker"].astype(str) + ", "
        + obs["Año"].astype(str) + ")"
    )
    return obs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Preparación: winsorización y estandarización
# ---------------------------------------------------------------------------

@dataclass
class Preparacion:
    variables: list
    q: float
    normalizar_filas: bool
    limites: pd.DataFrame          # límites inferior y superior por variable
    X_wins: pd.DataFrame           # razones recortadas
    escalador: StandardScaler
    Z: np.ndarray                  # matriz que entra al ACP
    recortes: pd.Series = field(default=None)  # filas recortadas por variable


def preparar_matriz(obs: pd.DataFrame, variables: list, q: float = 0.01,
                    normalizar_filas: bool = False) -> Preparacion:
    """
    1. Recorta (winsoriza) cada razón en los percentiles q y 1-q: los valores
       extremos se llevan al límite en vez de borrarse.
    2. Estandariza (media 0, desviación 1).
    3. Opcional: normaliza cada fila a longitud 1, como en el cuaderno de clase.
    """
    X = obs[variables].astype(float)
    inf, sup = X.quantile(q), X.quantile(1 - q)
    X_w = X.clip(inf, sup, axis=1)
    recortes = ((X < inf) | (X > sup)).sum()
    esc = StandardScaler().fit(X_w)
    Z = esc.transform(X_w)
    if normalizar_filas:
        Z = normalize(Z)
    limites = pd.DataFrame({"Límite inferior": inf, "Límite superior": sup})
    return Preparacion(variables, q, normalizar_filas, limites, X_w, esc, Z, recortes)


def transformar_nueva(prep: Preparacion, valores: dict) -> np.ndarray:
    """Aplica a una empresa nueva exactamente la misma preparación."""
    x = pd.DataFrame([valores])[prep.variables].astype(float)
    x = x.clip(prep.limites["Límite inferior"], prep.limites["Límite superior"], axis=1)
    z = prep.escalador.transform(x)
    if prep.normalizar_filas:
        z = normalize(z)
    return z


# ---------------------------------------------------------------------------
# 4. Análisis de componentes principales
# ---------------------------------------------------------------------------

def ajustar_acp(Z: np.ndarray, variables: list) -> dict:
    """Ajusta el ACP con todos los componentes y arma las tablas de lectura."""
    acp = PCA().fit(Z)
    k = len(acp.explained_variance_)
    nombres_pc = [f"CP{i + 1}" for i in range(k)]
    varianza = pd.DataFrame({
        "Componente": nombres_pc,
        "Valor propio": acp.explained_variance_,
        "Varianza explicada (%)": acp.explained_variance_ratio_ * 100,
        "Acumulada (%)": np.cumsum(acp.explained_variance_ratio_) * 100,
    })
    cargas = pd.DataFrame(acp.components_.T, index=variables, columns=nombres_pc)
    return {
        "modelo": acp,
        "varianza": varianza,
        "cargas": cargas,
        "puntajes": acp.transform(Z),
        # Criterio de Kaiser: valor propio mayor que el promedio. Con datos
        # estandarizados el promedio es 1; con filas normalizadas es menor.
        "kaiser": max(1, int((acp.explained_variance_ > acp.explained_variance_.mean()).sum())),
        "n_80": int(np.searchsorted(np.cumsum(acp.explained_variance_ratio_), 0.80) + 1),
    }


def orientar_componentes(res_acp: dict) -> dict:
    """
    El signo de un componente es arbitrario. Para que la lectura no cambie entre
    corridas, se orienta cada componente de modo que su carga de mayor valor
    absoluto sea positiva. Cambiar el signo no altera ni la varianza ni las
    distancias entre empresas.
    """
    cargas = res_acp["cargas"].copy()
    signos = np.sign(cargas.apply(lambda c: c[c.abs().idxmax()]))
    signos[signos == 0] = 1
    res = dict(res_acp)
    res["cargas"] = cargas * signos.values
    res["puntajes"] = res_acp["puntajes"] * signos.values
    res["signos"] = signos.values
    return res


def leer_componente(cargas: pd.Series, n: int = 4) -> str:
    """Resume en una frase las variables que más pesan en un componente."""
    orden = cargas.abs().sort_values(ascending=False).index[:n]
    pos = [nombre(v) for v in orden if cargas[v] > 0]
    neg = [nombre(v) for v in orden if cargas[v] < 0]
    texto = ""
    if pos:
        texto += "sube con " + ", ".join(pos)
    if neg:
        texto += ("; " if texto else "") + "baja con " + ", ".join(neg)
    return texto


# ---------------------------------------------------------------------------
# 5. K-Means
# ---------------------------------------------------------------------------

def evaluar_k(S: np.ndarray, k_min: int = 2, k_max: int = 10,
              semilla: int = 42) -> pd.DataFrame:
    """Inercia (codo) y silueta media para cada k."""
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(S)
        filas.append({
            "k": k,
            "Inercia": km.inertia_,
            "Silueta media": silhouette_score(S, km.labels_),
            "Grupo más pequeño": int(np.bincount(km.labels_).min()),
        })
    tabla = pd.DataFrame(filas)
    tabla["Caída de inercia (%)"] = -tabla["Inercia"].pct_change() * 100
    return tabla


def ajustar_kmeans(S: np.ndarray, k: int, semilla: int = 42) -> dict:
    km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(S)
    sil = silhouette_samples(S, km.labels_)
    return {"modelo": km, "etiquetas": km.labels_, "silueta": sil,
            "silueta_media": float(sil.mean())}


# ---------------------------------------------------------------------------
# 6. Perfiles, nombres y lectura de cupo
# ---------------------------------------------------------------------------

MARGENES = ["Gross Margin", "Operating Margin", "EBITDA Margin",
            "Pre-Tax Profit Margin", "Net Profit Margin"]


def _nivel(z: float, alto: str, medio: str, bajo: str, corte: float = 0.25) -> str:
    if z >= corte:
        return alto
    if z <= -corte:
        return bajo
    return medio


def nombrar_segmentos(obs: pd.DataFrame, prep: Preparacion,
                      etiquetas: np.ndarray) -> dict:
    """
    Pone un nombre descriptivo a cada grupo a partir de su perfil financiero,
    nunca a partir del número que le asigna K-Means. Se compara la media
    estandarizada del grupo con la de toda la base (0).
    """
    Zdf = pd.DataFrame(prep.escalador.transform(prep.X_wins), columns=prep.variables)
    Zdf["g"] = etiquetas
    medias = Zdf.groupby("g").mean()
    margen_op = obs.assign(g=etiquetas).groupby("g")
    nombres = {}
    for g in medias.index:
        partes = []
        mvars = [v for v in MARGENES + ["ROA - Return On Assets"] if v in prep.variables]
        if "Operating Margin" in prep.variables and \
                margen_op.get_group(g)["Operating Margin"].median() < 0:
            base = "En pérdidas operativas"
        elif mvars:
            zm = medias.loc[g, mvars].mean()
            base = _nivel(zm, "Margen alto", "Margen medio", "Margen estrecho")
        else:
            base = "Perfil mixto"
        partes.append(base)
        if "Asset Turnover" in prep.variables and base != "En pérdidas operativas":
            partes.append(_nivel(medias.loc[g, "Asset Turnover"],
                                 "rotación alta", "rotación media", "rotación baja"))
        if "Long-term Debt / Capital" in prep.variables:
            z = medias.loc[g, "Long-term Debt / Capital"]
            if abs(z) >= 0.5:
                partes.append("más apalancadas" if z > 0 else "menos apalancadas")
        if "Current Ratio" in prep.variables:
            z = medias.loc[g, "Current Ratio"]
            if abs(z) >= 0.5:
                partes.append("más líquidas" if z > 0 else "menos líquidas")
        texto = partes[0] + (", " + ", ".join(partes[1:]) if len(partes) > 1 else "")
        nombres[g] = texto
    # Si dos grupos quedan con el mismo nombre, se distinguen por tamaño.
    vistos = Counter(nombres.values())
    if any(c > 1 for c in vistos.values()):
        tam = pd.Series(etiquetas).value_counts()
        for nom, c in vistos.items():
            if c > 1:
                gs = sorted([g for g in nombres if nombres[g] == nom], key=lambda g: -tam[g])
                for i, g in enumerate(gs):
                    nombres[g] = f"{nom} ({chr(65 + i)})"
    return nombres


def perfilar_segmentos(obs: pd.DataFrame, etiquetas: np.ndarray, nombres: dict,
                       variables: list) -> pd.DataFrame:
    """Mediana de cada razón por grupo, más el tamaño del grupo."""
    tabla = obs.assign(Segmento=[nombres[g] for g in etiquetas])
    perfil = tabla.groupby("Segmento")[variables].median().T
    perfil.index = [nombre(v) for v in perfil.index]
    tam = tabla["Segmento"].value_counts()
    perfil.loc["Empresas-periodo"] = tam
    perfil.loc["Participación (%)"] = tam / tam.sum() * 100
    return perfil


def validar_con_rating(obs: pd.DataFrame, etiquetas: np.ndarray,
                       nombres: dict) -> pd.DataFrame:
    """
    Validación externa: la calificación NO entra al modelo. Después de formar
    los grupos se mira qué calificación tienen sus empresas.
    """
    t = obs.assign(Segmento=[nombres[g] for g in etiquetas])
    res = t.groupby("Segmento").agg(
        n=("Puntaje", "size"),
        grado_inv=("Grado de inversión", "mean"),
        puntaje_med=("Puntaje", "median"),
    )
    res["Grado de inversión (%)"] = res.pop("grado_inv") * 100
    res["Rating mediano"] = res["puntaje_med"].apply(rating_desde_puntaje)
    res = res.rename(columns={"n": "Empresas-periodo", "puntaje_med": "Puntaje mediano"})
    return res.sort_values("Puntaje mediano")


def lectura_de_cupo(pct_grado_inversion: float) -> tuple[str, str]:
    """
    Regla ilustrativa: traduce la proporción de empresas con grado de inversión
    de un segmento en una postura de cupo. Los cortes en tercios son una
    decisión explícita del análisis, no un estándar regulatorio.
    """
    if pct_grado_inversion >= 200 / 3:
        return ("Cupo amplio",
                "La mayoría del segmento tiene grado de inversión. Cupo estándar o "
                "ampliable, con revisión anual.")
    if pct_grado_inversion >= 100 / 3:
        return ("Cupo condicionado",
                "El segmento mezcla grado de inversión y especulativo. Cupo sujeto "
                "a covenants de apalancamiento y a revisión semestral.")
    return ("Cupo restringido",
            "Predomina el grado especulativo. Cupo bajo, con garantías y "
            "seguimiento trimestral.")


def ubicar_empresa(prep: Preparacion, acp: dict, n_comp: int, km: dict,
                   valores: dict) -> dict:
    """Ubica una empresa en el espacio del ACP y le asigna el centro más cercano."""
    z = transformar_nueva(prep, valores)
    s = acp["modelo"].transform(z)[:, :n_comp] * acp["signos"][:n_comp]
    centros = km["modelo"].cluster_centers_
    dist = np.linalg.norm(centros - s, axis=1)
    orden = np.argsort(dist)
    d1, d2 = dist[orden[0]], dist[orden[1]] if len(dist) > 1 else np.inf
    margen = (d2 - d1) / d2 if np.isfinite(d2) and d2 > 0 else 1.0
    return {"puntajes": s.ravel(), "grupo": int(orden[0]),
            "segundo": int(orden[1]) if len(dist) > 1 else None,
            "distancias": dist, "margen": float(margen)}


# ---------------------------------------------------------------------------
# 7. Solidez de balance dentro de cada grupo
# ---------------------------------------------------------------------------

TERCIOS = ["Tercio más débil", "Tercio medio", "Tercio más sólido"]


def componente_balance(cargas: pd.DataFrame, n_comp: int):
    """
    Busca, entre los componentes conservados, el que enfrenta liquidez y deuda:
    razón corriente y deuda LP / capital como sus dos cargas más grandes y con
    signos opuestos. Devuelve su posición (0, 1, 2…) o None si no existe.
    """
    cr, ld = "Current Ratio", "Long-term Debt / Capital"
    if cr not in cargas.index or ld not in cargas.index:
        return None
    for j in range(min(n_comp, cargas.shape[1])):
        c = cargas.iloc[:, j]
        top2 = set(c.abs().sort_values(ascending=False).index[:2])
        if top2 == {cr, ld} and np.sign(c[cr]) != np.sign(c[ld]):
            return j
    return None


def tercios_balance(obs: pd.DataFrame, puntaje_balance: np.ndarray,
                    etiquetas: np.ndarray, nombres: dict, signo: float = 1.0):
    """
    Divide cada grupo en tercios según su solidez de balance y calcula la
    proporción de grado de inversión en cada tercio. `signo` se usa para que
    "más sólido" signifique más liquidez y menos deuda.
    """
    t = obs.assign(Segmento=[nombres[g] for g in etiquetas],
                   cp=np.asarray(puntaje_balance) * signo)
    t["Tercio"] = t.groupby("Segmento")["cp"].transform(
        lambda s: pd.qcut(s, 3, labels=TERCIOS))
    tabla = t.pivot_table(index="Segmento", columns="Tercio",
                          values="Grado de inversión", aggfunc="mean",
                          observed=True) * 100
    cortes = t.groupby("Segmento")["cp"].quantile([1 / 3, 2 / 3]).unstack()
    cortes.columns = ["Corte 1/3", "Corte 2/3"]
    return tabla.reindex(columns=TERCIOS), cortes
