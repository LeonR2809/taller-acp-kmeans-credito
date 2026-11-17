# Capacidad de cupo y perfiles financieros de empresas calificadas

Taller individual de ACP y K-Means. Machine Learning aplicado a Economía y Finanzas, Facultad de Ciencias Económicas y Administrativas, Pontificia Universidad Javeriana Cali.

| | |
|---|---|
| **Estudiante** | *[Nombre completo]* |
| **Aplicación** | *[enlace de Streamlit]* |
| **Video de sustentación** | *[enlace del video]* |

## Contexto

El cupo de crédito de una empresa se decide leyendo sus razones financieras: liquidez, deuda, márgenes, eficiencia y rentabilidad. Muchas de esas razones cuentan la misma historia y no existe una etiqueta que diga cuánto cupo merece cada empresa.

**Pregunta.** ¿Las razones financieras de las empresas calificadas se pueden resumir en pocas dimensiones y agrupar en perfiles que correspondan a capacidades de cupo distintas?

* **Estructura (ACP):** ¿cuántos ejes describen la situación financiera de una empresa y de qué razones están hechos?
* **Grupos (K-Means):** ¿cuántos perfiles financieros distintos hay y en qué se diferencian?
* **Cupo:** ¿esos perfiles corresponden a calificaciones distintas y qué postura de cupo sugiere cada uno?

La calificación de las agencias **no entra al modelo**. Se usa después como validación externa: si los grupos formados solo con razones financieras ordenan la calificación, la segmentación captura capacidad de pago.

## Fuente de los datos

* **Conjunto:** *Corporate Credit Rating with Financial Ratios*, publicado en Kaggle: *[pegar aquí el enlace]*.
* **Archivo:** `data/corporateCreditRatingWithFinancialRatios.csv`, incluido en el repositorio sin modificaciones.
* **Cobertura:** 7.805 calificaciones emitidas entre abril de 2010 y diciembre de 2016 por siete agencias (Egan-Jones, Standard & Poor's, Moody's, Fitch, DBRS, Japan Credit Rating Agency y HR Ratings) a 686 empresas que cotizan en Estados Unidos, identificadas por su CIK de la SEC. Incluye 16 razones financieras, el sector y la calificación en letras.
* **Datos sensibles:** la información es pública (estados financieros y calificaciones publicadas) y no contiene datos personales.

**Limitaciones de la fuente**

* Solo hay empresas grandes, listadas en Estados Unidos y calificadas, así que no representan a pymes ni a empresas colombianas.
* La ventana 2010–2016 corresponde a un ciclo de tasas bajas.
* La fuente no documenta la fórmula de cada razón.
* El archivo trae la fecha de la calificación, no la del cierre contable, así que el año es aproximado.
* Una empresa aparece en varios años y sus observaciones no son independientes.
* 1.707 líneas del CSV vienen mal formadas y se reparan al leer.

## Metodología

1. **Lectura robusta.** Un `pd.read_csv` directo falla. Hay 1.685 líneas enteras entre comillas y 22 con una coma sin comillas en el nombre de una agencia. `analisis.cargar_datos` las repara sin descartar filas. Después de la reparación no hay valores faltantes.
2. **Unidad de análisis.** El 65 % de las filas repite el mismo estado financiero calificado por otra agencia o en otra fecha. Se deja una fila por empresa y juego de razones (2.694 empresas-periodo) con la mediana de sus calificaciones.
3. **Selección de variables.** Entran 10 razones: razón corriente, deuda LP / capital, márgenes bruto, operativo, EBITDA, antes de impuestos y neto, rotación de activos, ROA y ROI. Se excluyen seis:
   * el margen EBIT, que duplica al operativo;
   * deuda / patrimonio, ROE y rentabilidad del patrimonio tangible, que cambian de signo con patrimonio negativo;
   * los flujos por acción, que no son comparables entre empresas.
4. **Valores extremos.** Cada razón se winsoriza en los percentiles 1 y 99.
5. **Escala.** Las razones se estandarizan a media 0 y desviación 1. A diferencia del cuaderno de clase, **no** se normaliza cada fila, porque la pregunta es de nivel y no de estilo.
6. **ACP.** Se ajusta sobre la matriz estandarizada. El número de componentes se decide con el criterio de Kaiser, el 80 % de varianza acumulada y el codo del gráfico de sedimentación.
7. **K-Means.** Se agrupa sobre los componentes conservados con k de 2 a 10, usando el método del codo y el coeficiente de silueta (`n_init=10`, `random_state=42`). La estabilidad se comprueba con el índice de Rand ajustado frente a otras semillas.
8. **Interpretación.** Los grupos se nombran por su perfil, no por su número. Se describen con las medianas de las razones originales, se validan con la calificación y se traducen en una postura de cupo ilustrativa.
9. **Sensibilidad.** Se repite el análisis normalizando filas, sin winsorizar, winsorizando al 5 % y con k = 4.

## Resultados principales

**Componentes.** Tres conservan el **78,7 %** de la varianza:

* **CP1, rentabilidad (47,9 %):** márgenes operativo, antes de impuestos y neto, ROA, ROI y EBITDA.
* **CP2, rotación frente a margen (19,1 %):** la rotación de activos enfrentada al margen bruto, es decir, la identidad de DuPont.
* **CP3, solidez de balance (11,6 %):** la razón corriente enfrentada a la deuda LP / capital.

**Grupos.** Con k = 3 la silueta media llega a su máximo, 0,313.

| Segmento | Empresas-periodo | Margen operativo mediano | Rotación mediana | Grado de inversión | Rating mediano | Postura de cupo |
|---|---|---|---|---|---|---|
| Margen alto, rotación baja | 1.053 (39 %) | 21,1 % | 0,43 | 73,4 % | BBB+ | Amplio |
| Margen estrecho, rotación alta | 1.557 (58 %) | 7,6 % | 0,95 | 48,2 % | BB+ | Condicionado |
| En pérdidas operativas | 84 (3 %) | −32,7 % | 0,39 | 22,6 % | B+ | Restringido |

* **Los grupos ordenan la calificación sin haberla visto.** Entre el primero y el tercero hay seis escalones.
* **La deuda no separa a los grupos.** La deuda LP / capital mediana es 0,43, 0,42 y 0,46. Dentro de cada grupo, en cambio, el balance indica a quién revisar primero: en el grupo en pérdidas, el grado de inversión pasa de 7 % en el tercio de balance más débil a 36 % en el más sólido.
* **Las fronteras son anchas.** El 12,2 % de las empresas tiene silueta por debajo de 0,1.
* **La lectura de cupo no es un modelo de riesgo.** Es una traducción ilustrativa: dos tercios o más de grado de inversión en el grupo es cupo amplio, entre uno y dos tercios es condicionado, y menos de un tercio es restringido.

## Estructura del repositorio

```
├── app.py                                  # Aplicación de Streamlit (siete pantallas)
├── analisis.py                             # Carga, preparación, ACP, K-Means, perfiles y lectura de cupo
├── Taller_ACP_KMeans_Cupo_Credito.ipynb    # Desarrollo completo del taller, ya ejecutado
├── data/
│   └── corporateCreditRatingWithFinancialRatios.csv
├── requirements.txt
├── .streamlit/config.toml                  # Tema visual de la aplicación
└── README.md
```

La aplicación no usa imágenes ni tablas guardadas. Cada tabla, métrica y gráfico se calcula al abrirla, a partir del CSV, con las funciones de `analisis.py`. La sección 6 del notebook comprueba que esas funciones reproducen exactamente los resultados del notebook.

## Cómo ejecutar el proyecto

Se recomienda Python 3.11 o 3.12.

```bash
git clone https://github.com/TU_USUARIO/taller-acp-kmeans-credito.git
cd taller-acp-kmeans-credito
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Aplicación
streamlit run app.py

# Notebook
pip install notebook
jupyter notebook Taller_ACP_KMeans_Cupo_Credito.ipynb
```

**Despliegue.** La aplicación está publicada en Streamlit Community Cloud, conectada a este repositorio, con `app.py` como archivo principal.

## La aplicación

| Pantalla | Contenido |
|---|---|
| 1. Contexto | Pregunta, fuente y escala de calificación con la posición de cada grupo |
| 2. Datos | Reparación del archivo, calificaciones, duplicados, descriptivas y distribución de cada razón |
| 3. Preparación | Variables excluidas y motivos, winsorización, correlaciones y escala |
| 4. ACP | Varianza explicada, sedimentación, criterios, cargas y biplot |
| 5. K-Means | Codo y silueta, mapa 2D y 3D, diagrama de silueta, perfiles, validación con el rating y sectores |
| 6. Simulador de cupo | Ubica una empresa nueva o de la base en un grupo y sugiere una postura de cupo |
| 7. Conclusiones | Hallazgos, limitaciones y uso responsable |

En la barra lateral se pueden cambiar las razones del modelo, la winsorización, la normalización de filas, el número de componentes, el número de grupos y la semilla.
