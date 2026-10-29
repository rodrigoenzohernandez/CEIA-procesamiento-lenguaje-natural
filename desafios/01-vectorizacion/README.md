# Desafío 1 — Vectorización de texto

- **Se presentó:** clase 1 (jueves 22/10/2026).
- **Nota 10 hasta:** jueves 05/11 (clase 3), a última hora de Argentina. Desde el 06/11 baja un punto; ver la [tabla de notas máximas](../README.md#nota-máxima-según-la-fecha-de-entrega).
- **Entrega:** por el [formulario](https://forms.gle/wsWkQJWAFeh6g6qv5), con el link a este repo o a un Colab. Individual o en pareja.
- **Plantilla:** [`desafio-1.ipynb`](desafio-1.ipynb), copia de `Desafíos/Desafio_1.ipynb` del repo de la materia. Arriba tiene el ejemplo de la clase y al final la consigna, con una celda `# TÚ CODIGO AQUÍ`.
- **Teoría:** [resumen de la clase 1](../../clases/01-introduccion-vectorizacion/resumen.md): TF-IDF (§18), similitud coseno (§20), Naïve Bayes (§22) y el ejemplo de la plantilla (§24).
- **Estado:** pendiente.

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

## ✎ Ideas para resolverlo

- **Punto 1:** fijar la semilla (`np.random.default_rng(seed)`) para que los 5 documentos sean reproducibles. Mostrar el texto (recortado), la clase y la similitud de cada vecino. Las confusiones esperables son entre clases cercanas: los `comp.*` entre sí, las de religión (`alt.atheism`, `soc.religion.christian`, `talk.religion.misc`) y las de política. Revisar también documentos muy cortos: con `remove=(...)` algunos quedan casi vacíos y su vector es pobre.
- **Punto 2:** es un **1-NN** con similitud coseno. Como `TfidfVectorizer` normaliza las filas con L2, alcanza con `X_test @ X_train.T`. Hacerlo por lotes (la matriz densa completa es de 7.532 × 11.314) y tomar el `argmax` de cada fila. Comparar con el azar (F1 macro ≈ 0.05 con 20 clases) y con el baseline de Naïve Bayes.
- **Punto 3:** elegir los parámetros con validación cruzada sobre el train (`Pipeline` + `GridSearchCV` con `scoring='f1_macro'`) y medir el test una sola vez al final, para no sobreajustar al test. Candidatos:
  - En el vectorizador: `stop_words='english'`, `min_df`, `max_df`, `sublinear_tf`, `max_features`, `use_idf`, `norm`.
  - En el modelo: `alpha` en `MultinomialNB` y en `ComplementNB`, que además tiene `norm`.
- **Punto 4:** `X_train.T` da 101.631 términos × 11.314 documentos. Antes de calcular, normalizar las filas (`sklearn.preprocessing.normalize`) o usar `cosine_similarity`. Por la ley de Zipf, conviene elegir palabras de frecuencia media y de temas distintos (por ejemplo, *car*, *god*, *space*, *encryption*, *hockey*). Las palabras raras dan vecinos que solo comparten uno o dos documentos.
- **Conclusiones:** cerrar con una tabla que compare el F1 macro de cada enfoque (azar, prototipos, baseline, mejor Naïve Bayes) y con las limitaciones de BoW que se vieron en la clase: sin orden, sin sinónimos, vectores esparsos.
