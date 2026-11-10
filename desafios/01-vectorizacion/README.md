# Desafío 1 — Vectorización de texto

- **Se presentó:** clase 1 (jueves 22/10/2026).
- **Nota 10 hasta:** jueves 05/11 (clase 3), a última hora de Argentina. Desde el 06/11 baja un punto; ver la [tabla de notas máximas](../README.md#nota-máxima-según-la-fecha-de-entrega).
- **Entrega:** por el [formulario](https://forms.gle/wsWkQJWAFeh6g6qv5), con el link a este repo o a un Colab. Individual o en pareja.
- **Resolución:** [`desafio-1.ipynb`](desafio-1.ipynb). Es la plantilla de la materia (`Desafíos/Desafio_1.ipynb`: el ejemplo de la clase y la consigna) con los 4 puntos resueltos al final, cada uno con su análisis.
- **Teoría:** [resumen de la clase 1](../../clases/01-introduccion-vectorizacion/resumen.md): TF-IDF (§18), similitud coseno (§20), Naïve Bayes (§22) y el ejemplo de la plantilla (§24).
- **Estado:** resuelto, falta entregar.

## Lo que ya hace la plantilla

- Carga **20 Newsgroups** con `fetch_20newsgroups(subset='train'/'test', remove=('headers', 'footers', 'quotes'))`: unas 18.000 publicaciones en 20 temas, ya divididas en train y test.
- `TfidfVectorizer()` sin parámetros → `X_train` de **11.314 documentos × 101.631 términos** (sparse). `vocabulary_['car']` = 25.775.
- Similitud coseno del documento **4811** (un discurso radial del presidente, `talk.politics.misc`) contra todo el train: sus 5 más similares son de la misma clase.
- `MultinomialNB()` sin parámetros → **F1 macro en test = 0.585**, el baseline.

## Consigna

**Cada experimento realizado debe estar acompañado de una explicación o interpretación de lo observado.**

1. **Vectorizar documentos.** Tomar 5 documentos al azar y medir su similaridad con el resto de los documentos. Estudiar los 5 documentos más similares de cada uno y analizar si la similaridad tiene sentido según el contenido del texto y la etiqueta de clasificación.
2. **Construir un modelo de clasificación por prototipos (tipo zero-shot).** Clasificar los documentos de test comparando cada uno con todos los de entrenamiento y asignarle la clase del documento de entrenamiento con mayor similaridad.
3. **Entrenar modelos de clasificación Naïve Bayes para maximizar el desempeño de clasificación** (F1-score macro en test). Considerar cambiar parámetros de instanciación del vectorizador y de los modelos, y probar Naïve Bayes Multinomial y ComplementNB. **No cambiar el hiperparámetro `ngram_range` de los vectorizadores.**
4. **Transponer la matriz documento-término.** Se obtiene una matriz término-documento, que puede interpretarse como una colección de vectorizaciones de palabras. Estudiar la similaridad entre palabras: tomar 5 palabras y estudiar sus 5 más similares. **Elegir las palabras manualmente** para evitar términos poco interpretables.

## Aclaraciones de la clase (02:44:42 a 02:55:36)

- **Punto 1:** igual que el ejemplo, pero con 5 documentos. Analizar por qué salieron esos documentos (por ejemplo, si todos tienen "president" o "Washington"). Si alguno está mal relacionado, entrar a leerlo y explicar las **deficiencias del método** que lo hacen fallar.
- **Punto 2:** no necesita entrenamiento. Es "todo contra todo": cada documento de test se compara con todos los de train y se le asigna la clase del más similar (pregunta de Carlos). Se usa el test que ya está definido en la plantilla (pregunta de Elian). Calcular el **F1 macro** para compararlo con los otros métodos y concluir si sirve, por qué falla, y si es mejor que elegir al azar.
- **Punto 3:** el F1 de la plantilla es el baseline. Mejorarlo cambiando parámetros del vectorizador (por ejemplo, el de stopwords) y del modelo (por ejemplo, `alpha`), y probar `ComplementNB`. Comparar también con el clasificador del punto 2 y explicar por qué una combinación anda mejor que otra.
- **Punto 4:** con la matriz transpuesta cada palabra queda representada por los documentos en que aparece. Elegir 5 palabras interesantes, buscar sus 5 más similares con similitud coseno y analizar en qué casos tiene sentido y en cuáles no.
- **Sobre la evaluación:** se puede usar IA para el código, pero lo más importante es el **análisis**: interpretar los resultados y escribir las conclusiones de cada experimento.

## Resultados

| Modelo | F1 macro (test) |
|---|---|
| Azar | 0.049 |
| Prototipos 1-NN (TF-IDF de la plantilla) | 0.505 |
| Prototipos 1-NN (sin stopwords, `sublinear_tf`) | 0.545 |
| `MultinomialNB(alpha=1)`, baseline de la plantilla | 0.585 |
| `MultinomialNB(alpha=0.01)` con `stop_words='english'` | 0.684 |
| `ComplementNB(alpha=0.3)` con `stop_words='english'` y `sublinear_tf` | **0.698** |

- **Punto 1:** 13 de los 25 vecinos son de la misma clase. La similitud falla con documentos cortos o vacíos, cuando el sentido depende del orden (*program manager*) y cuando el parecido lo explican las stopwords.
- **Punto 2:** el 1-NN es diez veces mejor que el azar, pero queda por debajo de Naïve Bayes. Decide un solo vecino, los 224 documentos de test vacíos caen todos en `rec.autos` y confunde grupos de temas cercanos (religión, política, hardware).
- **Punto 3:** lo que más importa es `alpha`. Con `alpha=1` el suavizado suma 26 veces el peso TF-IDF de cada clase, y el modelo casi no predice las clases chicas. Las variantes de TF-IDF cambian el F1 en ±0.01; sin IDF ni normalización cae unos 0.07. Los parámetros se eligieron con validación cruzada en train (≈ 0.76), que da optimista frente al test (≈ 0.70) porque el test es de fechas posteriores.
- **Punto 4:** los vecinos de cada palabra son del mismo tema, no sinónimos, y cada palabra tiene un solo sentido (*key* → criptografía, *drive* → discos). Las palabras muy raras (ley de Zipf) arruinan los vecinos de *encryption* y *apple*; sin stopwords y con `min_df=10` se arreglan.

**Para entregar:** el link a este notebook en GitHub (el repo es público) o una copia en Colab compartida con los dos docentes.
