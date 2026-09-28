# Diagnóstico de cáncer de mama: red neuronal, K-means, XGBoost y GMM

Comparación de modelos de aprendizaje supervisado y no supervisado sobre el dataset *Breast Cancer Wisconsin (Diagnostic)*. El objetivo es clasificar un tumor como benigno o maligno a partir de 30 medidas de núcleos celulares, y explorar qué estructura encuentran los métodos no supervisados sin ver el diagnóstico.

Proyecto en equipo del curso de Aprendizaje Máquina del Tecnológico de Monterrey.

## Resumen de hallazgos

- **La red neuronal clasificó bien (96.5 % de accuracy) y con muy pocos falsos negativos.** En 114 tumores de prueba solo dejó pasar 1 maligno de 43 (recall de 0.977), con 3 falsos positivos. En un diagnóstico, el falso negativo es el error más costoso, y ese es justo el punto fuerte del modelo.
- **XGBoost quedó cerca, pero con más falsos negativos.** Con hiperparámetros por defecto obtuvo 95.6 % de accuracy y un recall de 0.930 (3 malignos no detectados), con una precision algo mayor (0.952). Con un conjunto de prueba de 114 casos, la diferencia entre ambos modelos es de pocos tumores y no es concluyente.
- **K-means (k = 3) separa los casos claros, pero no todo el dataset.** Un grupo contiene solo tumores malignos (118 de 118), otro es 91 % benigno (321 benignos y 33 malignos) y un tercero es mixto (36 benignos y 61 malignos). El *Adjusted Rand Index* contra el diagnóstico real es de **0.54**.
- **GMM (8 componentes) encuentra subgrupos poco definidos.** Su silhouette es de 0.128 y su ARI contra el diagnóstico es de 0.32, lo que indica que los 8 subgrupos no están bien separados y coinciden poco con la etiqueta benigno/maligno.
- **Conclusión general:** con estas 30 variables, los métodos supervisados superan con claridad a los no supervisados para predecir el diagnóstico. Los no supervisados sirven para explorar la estructura de los datos, no para reemplazar un clasificador.

### Resultados en el conjunto de prueba (114 tumores)

| Modelo | Accuracy | Precision | Recall | F1 | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|---|
| Red neuronal | 0.9649 | 0.9333 | 0.9767 | 0.9545 | 1 | 3 |
| XGBoost (parámetros por defecto) | 0.9561 | 0.9524 | 0.9302 | 0.9412 | 3 | 2 |

*Clase positiva: tumor maligno. Los conteos de falsos negativos y positivos se derivan de precision, recall y el tamaño del conjunto de prueba.*

### Resultados no supervisados (dataset completo, 569 tumores)

| Método | Configuración | Métrica |
|---|---|---|
| K-means | k = 3, elegido con el método del codo (`KneeLocator`) | ARI = 0.539 |
| GMM | 8 componentes, elegido por AIC (rango de 1 a 9) | Silhouette = 0.128, ARI = 0.321 |

Tabla de contingencia de K-means contra el diagnóstico real:

| Cluster | Benigno | Maligno |
|---|---|---|
| 0 | 36 | 61 |
| 1 | 0 | 118 |
| 2 | 321 | 33 |

## Datos

*Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository (id 17): **569 tumores**, **30 variables numéricas** (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal, cada una con media, error estándar y valor extremo) y el diagnóstico (M = maligno, B = benigno). No hay valores faltantes. Se descarga directamente con `ucimlrepo`.

Cita: Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## Método

**Preprocesamiento.** Etiquetas codificadas con `LabelEncoder` (benigno = 0, maligno = 1). División 80 % entrenamiento y 20 % prueba con semilla 42. Estandarización con `StandardScaler` ajustado solo con los datos de entrenamiento, para no filtrar información al conjunto de prueba.

**Aprendizaje supervisado: red neuronal (Keras).**

```
Dense 6 (ReLU) → Dropout 0.3 → Dense 4 (ReLU) → Dropout 0.3 → Dense 2 (ReLU) → Dropout 0.3 → Dense 1 (sigmoide)
```

227 parámetros entrenables. Adam con tasa de aprendizaje de 10⁻³, entropía cruzada binaria, 150 épocas, lotes de 15. El dropout de 0.3 en cada capa oculta actúa como regularización.

**Aprendizaje no supervisado: K-means.** Sobre los datos estandarizados, con el número de clusters elegido por el método del codo (k = 3), y visualización en 2 dimensiones con PCA. Se compara el resultado contra el diagnóstico real con una tabla de contingencia y el ARI.

**Extra: XGBoost.** Clasificador de árboles con gradiente (`XGBClassifier`). Se definió una versión regularizada (`max_depth=4`, `learning_rate=0.1`, `reg_alpha=0.5`, `reg_lambda=1.0`, `subsample=0.8`, `colsample_bytree=0.8`), pero las métricas reportadas arriba corresponden a la versión con **parámetros por defecto** que se entrena en la celda de evaluación. Incluye gráfica de importancia de variables.

**Extra: GMM.** Modelo de mezclas gaussianas. Se compararon BIC y AIC para 1 a 9 componentes y se eligió AIC (8 componentes) para favorecer la exploración de subgrupos, con visualización en PCA.

## Estructura

```
.
├── Cancer_de_mama_red_neuronal.ipynb   # análisis completo
└── README.md
```

## Cómo ejecutarlo

Requiere Python 3.9 o superior. El notebook se ejecutó en Google Colab.

```bash
pip install pandas numpy scikit-learn tensorflow xgboost ucimlrepo kneed matplotlib seaborn
jupyter notebook Cancer_de_mama_red_neuronal.ipynb
```

## Limitaciones

- El conjunto de prueba tiene solo 114 tumores, y también se usó como datos de validación durante el entrenamiento de la red (sin selección de modelo ni *early stopping* sobre él). Una validación cruzada daría estimaciones más estables.
- La comparación entre la red neuronal y XGBoost es sobre una sola división de datos. No se puede afirmar que uno supere al otro.
- XGBoost se evaluó sin la configuración regularizada que se había definido. Queda pendiente reportar esa versión.
- Es un ejercicio académico sobre un dataset clásico. No es una herramienta de diagnóstico clínico.

## Autores

- David Tinoco Romero, A01801491
- Emilio Páez de la Mora, A01801224

Tecnológico de Monterrey.
