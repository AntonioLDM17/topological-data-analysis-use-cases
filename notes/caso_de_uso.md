# Topological Data Analysis for ECG: Normal vs Arrhythmic

Este repositorio implementa un **caso de uso real** de Topological Data Analysis (TDA) aplicado a señales de ECG, comparando:

- Un **ECG normal** (ritmo sinusal normal, base NSRDB de PhysioNet)
- Un **ECG arrítmico** (MIT-BIH Arrhythmia Database)

El objetivo es mostrar, de forma visual y cuantitativa, cómo la **estructura topológica** de la dinámica del corazón cambia entre un paciente sano y uno con arritmias, utilizando:

- **Embeddings de ventana deslizante (sliding window embedding)**
- **Complejos de Vietoris–Rips**
- **Homología persistente en H₀ y H₁**
- **Diagramas de persistencia**

Este caso de uso sirve como **complemento práctico** al README teórico de TDA del proyecto, donde se introducen de forma abstracta conceptos como complejos simpliciales, módulos persistentes, barcodes y diagramas de persistencia. Aquí esos conceptos se materializan sobre un problema biomédico concreto.

---

## 1. Resumen del pipeline

El script `tda_timeseries.py` hace lo siguiente:

1. **Descarga/carga dos registros de ECG reales**:
   - `16265` del dataset **NSRDB** (normal)
   - `100` del dataset **MITDB** (arrítmico)

2. **Remuestrea** las señales a una frecuencia objetivo (por defecto 128 Hz o 30 Hz, según configuración) para que ambas sean comparables.

3. Para cada registro:
   - Corta un **segmento de 30 segundos**.
   - Representa la serie temporal y la guarda en `outputs/<tag>_timeseries.png`.
   - Construye un **sliding window embedding** de dimensión `dim` y retardo `tau`.
   - Estandariza el embedding y lo proyecta a 2D usando **PCA**, guardando la nube de puntos en `outputs/<tag>_embedding.png`.
   - Calcula la **homología persistente** (Vietoris–Rips, dimensiones 0 y 1).
   - Guarda los **diagramas de persistencia H₀ y H₁** en `outputs/<tag>_persistence.png`.
   - Calcula la **persistencia máxima en H₁** como medida del “ciclo dominante”.

4. Finalmente imprime una **comparación de persistencias H₁** entre el ECG normal y el arrítmico, junto con una interpretación básica.

---

## 2. Instalación y dependencias

### 2.1. Requisitos

- Python 3.9+ (recomendado)
- Acceso a internet (para descargar los registros desde PhysioNet)
- Una cuenta aceptando las condiciones de uso de PhysioNet (para usar `wfdb` con `pn_dir`)

### 2.2. Instalación de librerías

```bash
pip install numpy matplotlib scikit-learn wfdb ripser persim
````

---

## 3. Cómo ejecutar el código

Desde la carpeta donde tengas el script:

```bash
python src/tda_timeseries.py
```

Se ejecutará el pipeline completo y verás por consola algo parecido a:

* Proceso del ECG normal
* Proceso del ECG arrítmico
* Persistencias máximas H₁ para cada uno
* Interpretación rápida

### 3.1. Archivos de salida

Se crean automáticamente en la carpeta `outputs/`:

Para el **ECG normal** (`tag="normal"`):

* `normal_timeseries.png` → serie temporal (ECG vs tiempo)
* `normal_embedding.png` → embedding PCA 2D del sliding window
* `normal_persistence.png` → diagramas de persistencia H₀ y H₁

Para el **ECG arrítmico** (`tag="arritmico"`):

* `arritmico_timeseries.png`
* `arritmico_embedding.png`
* `arritmico_persistence.png`

---

## 4. Estructura del código

El script se organiza en bloques lógicos:

### 4.1. Carga y remuestreo de ECG: `load_ecg_segment`

```python
def load_ecg_segment(record_name="16265", pn_dir="nsrdb",
                     fs_target=128, duration_seconds=30, channel=0):
    ...
```

* Usa `wfdb.rdrecord(record_name, pn_dir=pn_dir)` para leer el registro desde PhysioNet.

* Extrae la señal del canal indicado (`channel`, por defecto 0).

* Si la frecuencia original `fs_orig` es distinta de `fs_target`, remuestrea con:

  ```python
  x_resampled, _ = wfproc.resample_sig(x, fs_orig, fs_target)
  ```

* Recorta los **primeros `duration_seconds` segundos**.

* Devuelve:

  * `t`: vector de tiempos
  * `ecg`: señal 1D remuestreada
  * `fs`: frecuencia efectiva (igual a `fs_target` si remuestrea)

**Motivación**
Mediante el remuestreo garantizamos que el número de puntos por segundo sea comparable entre el ECG normal y el arrítmico. Esto es crucial para:

* Comparar topología de forma justa.
* Evitar que el cálculo de homología explote computacionalmente en el caso arrítmico (registros de 360 Hz).

---

### 4.2. Sliding window embedding: `sliding_window`

```python
def sliding_window(x, dim, tau):
    N = len(x)
    M = N - (dim - 1) * tau
    ...
    windows = np.stack(
        [x[i:i + M] for i in range(0, dim * tau, tau)],
        axis=1
    )
```

Dada una serie `x`:

* `dim` = dimensión del embedding (número de “lags” o retrasos).
* `tau` = retardo (en número de muestras).
* Cada punto del embedding es:

  [
  X_t = (x_t,\ x_{t+\tau},\ x_{t+2\tau},\ \dots,\ x_{t+(dim-1)\tau})
  ]

El resultado es una matriz `X` de tamaño `(M, dim)`, donde:

* `M = N - (dim - 1)\cdot\tau`
* Cada fila de `X` es una “ventana” del ECG.

**Interpretación geométrica (Takens / teoría de sistemas dinámicos)**

* La serie temporal `x(t)` se interpreta como una **proyección** de un sistema dinámico subyacente (el corazón).
* El sliding window embedding reconstruye (bajo ciertas condiciones) una **variedad inmersa** en (\mathbb{R}^{\text{dim}}) que refleja la dinámica del sistema.
* Si el sistema es aproximadamente **periódico**, la órbita en el espacio de estados tiene forma de **círculo** (un bucle).
* Si la dinámica es **irregular** (arritmia), la trayectoria es más caótica, y la estructura de círculo se rompe.

Este embedding es el puente entre:

* la señal 1D en el tiempo,
* y un conjunto de puntos en (\mathbb{R}^{\text{dim}}) sobre el que se puede hacer TDA.

---

### 4.3. Visualización de la serie: `plot_time_series`

```python
def plot_time_series(t, ecg, title, save_path):
    ...
    plt.plot(t, ecg)
    ...
    plt.savefig(save_path, dpi=300)
```

Simplemente representa `ecg(t)` vs tiempo y lo guarda en disco. Sirve para ver:

* Ritmo sinusal regular (normal)
* Picos irregulares, intervalos variables, etc. (arrítmico)

---

### 4.4. Embedding en 2D con PCA: `plot_embedding_2d`

```python
def plot_embedding_2d(X, title, save_path):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
```

* `X` es el embedding de ventana deslizante en dimensión `dim` (p.ej. 30).
* PCA proyecta los puntos a 2D para poder visualizar la **forma global**:

  * En el ECG normal, se espera algo aproximadamente **circular**.
  * En el arrítmico, la forma será más deformada, menos “bucle perfecto”.

**Esta figura es la primera evidencia visual de la topología subyacente.**

---

### 4.5. Homología persistente: `compute_persistence`

```python
def compute_persistence(X, maxdim=1):
    result = ripser(X, maxdim=maxdim)
    return result["dgms"]
```

* Usa `ripser` para construir el **complejo de Vietoris–Rips** sobre los puntos de `X`.
* Calcula la **homología persistente** hasta dimensión 1:

  * `H₀`: componentes conexas (clusters).
  * `H₁`: ciclos (bucles).

Devuelve una lista `dgms`:

* `dgms[0]`: pares (birth, death) de H₀.
* `dgms[1]`: pares (birth, death) de H₁.

---

### 4.6. Diagramas de persistencia: `plot_persistence_diagrams`

```python
def plot_persistence_diagrams(dgms, title_prefix, save_path):
    plt.subplot(1, 2, 1)
    plot_diagrams(dgms[0], show=False)  # H0
    ...
    plt.subplot(1, 2, 2)
    plot_diagrams(dgms[1], show=False)  # H1
```

Muestra y guarda:

* **H₀**: cómo se forman y fusionan componentes a medida que aumenta el radio del complejo VR.
* **H₁**: cómo aparecen y desaparecen ciclos (bucles) en el espacio de puntos.

Cada punto ((b, d)) es un “feature topológico”:

* **Birth** = escala a la que aparece el ciclo.
* **Death** = escala a la que desaparece (por llenarse o colapsar).
* **Persistencia** = (d - b) → mide cuán relevante es el ciclo:

  * Persistencia grande = estructura geométrica robusta.
  * Persistencia pequeña = ruido.

---

### 4.7. Pipeline para un registro: `process_ecg_record`

```python
def process_ecg_record(record_name, pn_dir, tag, duration_seconds=30, dim=30, tau=2):
    ...
    t, ecg, fs = load_ecg_segment(...)
    plot_time_series(...)
    X = sliding_window(ecg, dim, tau)
    X_scaled = StandardScaler().fit_transform(X)
    plot_embedding_2d(...)
    dgms = compute_persistence(X_scaled, maxdim=1)
    plot_persistence_diagrams(...)
    ...
    H1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    ...
    max_pers = (H1[:, 1] - H1[:, 0]).max()
```

Hace todo el flujo para un registro **normal** o **arrítmico**:

1. Carga la señal (remuestreada).
2. Guarda su serie temporal.
3. Construye embedding + estandariza.
4. Guarda el embedding PCA.
5. Calcula la homología persistente.
6. Guarda los diagramas H₀ y H₁.
7. Calcula la **persistencia máxima** en H₁:

[
\text{max_pers} = \max_i (d_i - b_i), \quad (b_i, d_i) \in \text{H1}.
]

**Interpretación**
Max persistencia H₁ ≈ fuerza del bucle dominante → medida de **periodicidad topológica** de la señal.

---

### 4.8. Comparación normal vs arrítmico (bloque `__main__`)

```python
if __name__ == "__main__":
    pers_normal = process_ecg_record(... normal ...)
    pers_arr    = process_ecg_record(... arritmico ...)
    ...
    print(f"Persistencia H1 (normal)    = {pers_normal:.4f}")
    print(f"Persistencia H1 (arrítmico) = {pers_arr:.4f}")
```

* Ejecuta el pipeline para:

  * `record_name="16265", pn_dir="nsrdb"` → **normal**
  * `record_name="100", pn_dir="mitdb"` → **arrítmico**
* Compara `pers_normal` y `pers_arr`.

En un escenario ideal:

* `pers_normal` > `pers_arr`

→ La señal normal exhibe un **ciclo topológico más fuerte** en H₁, consistente con una **dinámica periódica estable** (latido regular).
→ La señal arrítmica, al romper la regularidad, **“rompe” también la estructura de círculo** en el embedding; el ciclo en H₁ es menos persistente o está fragmentado.

---

## 5. Teoría topológica detrás del caso de uso

### 5.1. De una serie temporal a una nube de puntos

La señal ECG es una función:

[
x : {0, 1, \dots, N-1} \to \mathbb{R}
]

La vemos como una observación escalar de un sistema dinámico subyacente (el corazón + sistema eléctrico + medida).

La **teoría de Takens** dice, de forma muy informal, que podemos reconstruir la dinámica original en un espacio de dimensión superior usando vectores de retardos:

[
X_t = (x_t, x_{t+\tau}, x_{t+2\tau}, \dots, x_{t+(m-1)\tau}) \in \mathbb{R}^m
]

El conjunto de todos esos vectores (X_t) (para muchos `t`) forma una nube de puntos que “vive” cerca de una **variedad** (o atractor) que refleja la dinámica del sistema.

* Si el sistema es **periódico**, la órbita es topológicamente un **círculo**.
* Si hay **cuasiperiodicidad**, pueden surgir toros.
* Si la dinámica es más caótica, la topología se vuelve más compleja.

En este proyecto:

* Para el ECG normal, esperamos una dinámica muy próxima a un **círculo** → un ciclo 1D dominante.
* Para el ECG arrítmico, la dinámica está perturbada, se introducen irregularidades → el círculo se deforma y la topología no muestra un ciclo tan limpio.

---

### 5.2. Complejo de Vietoris–Rips

Dada una nube de puntos (X = {x_1, ..., x_M}) en (\mathbb{R}^m), el **complejo de Vietoris–Rips** a escala (\epsilon) se construye así:

* Cada punto es un **vértice**.
* Un **simplejo** (arista, triángulo, tetraedro, etc.) se añade si **todos sus vértices están a distancia ≤ ε** entre sí.

Al aumentar ε desde 0:

* Aparecen aristas, triángulos, etc.
* Los puntos se empiezan a conectar, aparecen **componentes** y **ciclos**.
* Estos ciclos pueden “llenarse” con triángulos y desaparecer.

---

### 5.3. Homología y grupos de Betti

La **homología** computa, en términos muy intuitivos:

* H₀: número de componentes conexas.
* H₁: número de “agujeros” tipo ciclo (como un círculo).
* H₂: cavidades (como el interior de una esfera).

En nuestro caso:

* **H₀** nos habla de cómo se conectan los puntos (clusters).
* **H₁** detecta **bucles** → lo importante aquí.

---

### 5.4. Homología persistente

En lugar de fijar un solo ε, consideramos una **filtración**:

[
\epsilon_0 < \epsilon_1 < \dots < \epsilon_K
]

Y miramos cómo cambian los grupos de homología a medida que aumentamos ε.

* Un ciclo puede **nacer** a cierta escala (cuando se forma un “agujero”).
* Luego puede **morir** a otra escala (cuando ese agujero se rellena).

Cada ciclo se representa como un par:

[
(b, d)
]

* `b` = “birth” (nacimiento)
* `d` = “death” (muerte)
* **Persistencia** = `d - b` = cuánto tiempo sobrevive el ciclo.

Cuanto mayor la persistencia, más robusta es la estructura frente a ruido.

---

### 5.5. Interpretación específica para ECG

En el embedding:

* ECG normal → puntos dispuestos alrededor de una trayectoria casi cerrada → topológicamente un círculo → **un ciclo 1D muy persistente**.
* ECG arrítmico → picos y ritmos irregulares → la trayectoria se deforma, se “engorda”, se rompe → el ciclo 1D es menos claro y su persistencia tiende a bajar.

Por eso, la **persistencia máxima en H₁** se usa como:

> **Indicador topológico de periodicidad cardiaca**.

---

## 6. Relación con el README teórico de TDA del proyecto

El README general de TDA del proyecto introduce, de forma abstracta, los conceptos de:

* **Homología persistente** y módulos de persistencia
* **Complejos de Čech y Vietoris–Rips**
* **Features topológicas** (componentes, ciclos, vacíos) en H₀, H₁, H₂
* **Barcodes y diagramas de persistencia**
* Nociones de **estabilidad** y distancias (Wasserstein, bottleneck)

Este repositorio implementa un ejemplo concreto donde todos esos conceptos se materializan:

* El ECG, tras el **sliding window embedding**, se convierte en un **point cloud** sobre el que construimos un **Vietoris–Rips complex**, tal y como se describe en la parte teórica.
* La **filtración** de ese complejo genera los grupos de homología en distintas escalas; sus generadores se codifican en **diagramas de persistencia**, exactamente como en la definición general de persistent homology.
* Los **0-ciclos (H₀)** se interpretan como componentes del embedding, mientras que los **1-ciclos (H₁)** se interpretan aquí como la **estructura de bucle asociada a la periodicidad cardiaca**.
* El principio “**features persistentes = estructura real, features cortas = ruido**”, explicado en el README teórico, se verifica empíricamente: el ECG normal presenta un ciclo H₁ más persistente que el arrítmico.

En otras palabras, **este caso de uso funciona como una demostración aplicada de la teoría desarrollada en el README de TDA**: todo el formalismo abstracto se traduce en una herramienta concreta para comparar latidos sanos y patológicos.

---

## 7. Relación con el paper original y su repositorio

Este caso de uso está inspirado en el trabajo:

> **M. Dindin, Y. Umeda, F. Chazal – “Topological Data Analysis for Arrhythmia Detection through Modular Neural Networks” (2019)**

* Paper: [https://arxiv.org/abs/1906.05795](https://arxiv.org/abs/1906.05795)
* Código oficial: [https://github.com/anticdimi/tda-arrhythmia-detection](https://github.com/anticdimi/tda-arrhythmia-detection)

### 7.1. ¿Qué hace el paper?

Muy resumido:

* Propone una **arquitectura de deep learning multicanal** para **detectar y clasificar arritmias** a partir de ECG de PhysioNet.
* Utiliza **Topological Data Analysis** como uno de los canales principales:

  * Calcula **barcodes 1D** de sublevel/upper-level sets del ECG 1D.
  * Convierte los barcodes en **Betti curves**.
  * Alimenta esas Betti curves a una **CNN** como canal TDA.
* Combina:

  * Canal TDA (Betti curves),
  * Canal autoencoder,
  * Canal de características clásicas,
  * Otros canales convolucionales.
* Entrena y evalúa un modelo de **clasificación supervisada (binary + multi-clase)** con validación por paciente.

### 7.2. Similitudes con este repositorio

* Ambos usan **ECGs reales de PhysioNet** (NSRDB y MITDB).
* Ambos emplean **Topological Data Analysis** para capturar la “forma” del latido.
* En ambos casos, la TDA se usa para:

  * Ser robusta frente a deformaciones temporales (bradicardia, taquicardia).
  * Describir geométricamente la señal de forma más estable que features puramente locales.
* El objetivo conceptual es parecido: **mejorar la comprensión/diagnóstico de arritmias usando TDA**.

### 7.3. Diferencias importantes

**1. Tipo de TDA utilizada**

* **Paper**:

  * Usa **sublevel/upper-level set persistence** sobre la señal 1D.
  * Obtiene barcodes 1D → los transforma en **Betti curves**.
  * Trabaja principalmente en **H₀** (componentes conectas).
* **Este repositorio**:

  * Usa **sliding window embedding** para reconstruir la dinámica en (\mathbb{R}^m).
  * Construye un **complejo de Vietoris–Rips** sobre el embedding.
  * Analiza homología en **H₀ y H₁**, con foco en **ciclos 1D** (forma de círculo).

**2. Objetivo del experimento**

* **Paper**:

  * Entrena un modelo de **clasificación supervisada** (detección de arritmias + clasificación en hasta 13 clases).
  * Arquitectura modular compleja (CNNs, autoencoder, canal TDA…).
* **Este repositorio**:

  * Se centra en un **caso de uso exploratorio y explicativo**:

    * Comparar topológicamente un ECG normal vs uno arrítmico.
    * Mostrar visualmente embeddings y diagramas de persistencia.
    * Usar la **persistencia máxima en H₁** como indicador de periodicidad.

**3. Complejidad del modelo**

* **Paper**:

  * Red neuronal profunda multicanal (varios canales de entrada + fusión).
  * Entrenamiento completo de clasificación.
* **Este repositorio**:

  * No hay modelo de deep learning.
  * Todo el análisis es **no supervisado**, interpretativo y centrado en la geometría/topología.

**4. Representación topológica**

* **Paper**:

  * Representación numérica final = **Betti curves** (1D, aptas como entrada de una CNN).
* **Este repositorio**:

  * Representación final = **diagramas de persistencia H₁** y métricas simples (persistencia máxima).
  * Más orientado a **visualización y comprensión geométrica** que a clasificación.

---

## 8. Qué se podría extender (trabajos futuros)

Algunas extensiones naturales:

* Probar distintos parámetros de embedding (`dim`, `tau`).
* Comparar más de un paciente normal vs varios arrítmicos.
* Calcular otras estadísticas topológicas:

  * Número de ciclos persistentes por encima de un umbral.
  * Distancias entre diagramas (Wasserstein, bottleneck).
* Usar las features topológicas (diagramas → imágenes / vectores) como entrada a un modelo de clasificación.
* **Implementar un modelo de clasificación de arritmia inspirado en el paper original**, por ejemplo:

  * Construir un canal TDA basado en este mismo pipeline (embeddings + H₁).
  * Combinarlo con otros canales (señal cruda, Fourier, autoencoder).
  * Entrenar una red neuronal que integre la información topológica con otras características, y evaluar su capacidad de generalizar a pacientes no vistos.

---

## 9. Resumen conceptual para la memoria/presentación

1. Partimos de **ECG reales** (normal y arrítmico).
2. Reconstruimos la dinámica cardiaca vía **sliding window embedding**.
3. Aplicamos **Topological Data Analysis**:

   * Complejo de Vietoris–Rips
   * Homología persistente
4. Observamos que:

   * En el ECG normal, el embedding parece un círculo y hay un ciclo en H₁ **muy persistente**.
   * En el ECG arrítmico, la estructura de círculo se rompe y la persistencia en H₁ es menor.
5. Concluimos que la TDA proporciona una forma **geométrica y robusta** de medir la periodicidad del corazón más allá de técnicas puramente estadísticas o frecuenciales, y que este caso de uso se alinea con el README teórico del proyecto y con ideas del paper original, pero con un enfoque más simple, visual y didáctico, ideal para proyectos docentes de Geometría de la Información.
