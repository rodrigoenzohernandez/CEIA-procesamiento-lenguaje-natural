# Desafío 2 — Word embeddings con Gensim

- **Se presentó:** clase 2 (jueves 29/10/2026).
- **Nota 10 hasta:** jueves 12/11 (clase 4), a última hora de Argentina. Desde el 13/11 baja un punto; ver la [tabla de notas máximas](../README.md#nota-máxima-según-la-fecha-de-entrega).
- **Entrega:** por el [formulario](https://forms.gle/wsWkQJWAFeh6g6qv5), con el link a este repo o a un Colab. Individual o en pareja.
- **Plantilla:** [`desafio-2.ipynb`](desafio-2.ipynb). Es la copia de `Desafíos/Desafio_2.ipynb` del repo de la materia, con el ejemplo de la clase arriba y la consigna al final. La resolución va en este mismo notebook.
- **Teoría:** [resumen de la clase 2](../../clases/02-word-embeddings/resumen.md): CBOW y skip-gram (§16 a §18), negative sampling (§19), t-SNE (§20), Gensim y el ejemplo de la plantilla (§21 a §24) y la consigna (§25).
- **Estado:** pendiente.

## Lo que ya hace la plantilla

- Instala Gensim si falta (la salida guardada muestra la 4.4.0) y baja el *songs dataset* del repo de la materia: letras de 49 artistas, un `.txt` por artista.
- Lee `beatles.txt` con un documento por línea: **1846 documentos**.
- Tokeniza cada línea con `text_to_word_sequence` de Keras, que pasa a minúsculas, saca la puntuación y separa por espacios.
- `Word2Vec(min_count=5, window=2, vector_size=300, negative=20, workers=1, sg=1)` es un **skip-gram**. El vocabulario queda en **445 palabras**.
- Entrena 20 épocas con un callback que imprime la loss de cada una, que baja de 113.045 a 48.960.
- Prueba `most_similar` con `positive` y con `negative` (*darling*, *love*, *four*, *money*), una palabra fuera del vocabulario (da `KeyError`) y `get_vector`.
- Reduce los vectores con `TSNE(random_state=0)` y grafica con Plotly, en 2D y en 3D, las primeras `MAX_WORDS = 200` palabras. Exporta `vectors.tsv` y `labels.tsv` para el proyector de TensorFlow.

## Consigna

**Cada experimento realizado debe estar acompañado de una explicación o interpretación de lo observado, incluido el sustento de las decisiones que tomaron (como modificación de hiperparámetros).**

El notebook de entrega tiene que correr de inicio a fin sin errores.

1. **Crear vectores propios con Gensim**, como en clase, con un corpus propio. Las opciones de dataset están en los enlaces sugeridos en la clase 2.
2. **Elegir términos de interés** y buscar sus términos más similares y menos similares.
3. **Reducir la dimensionalidad** de los embeddings a 2 dimensiones y graficar los embeddings proyectados. Elegir una cantidad de términos (variable `MAX_WORDS`) que deje una visualización adecuada.
4. **Inspeccionar el gráfico** y buscar pequeños grupos de palabras. Interpretarlos e intentar sacar conclusiones. En lo posible, acompañar cada grupo con capturas pegadas en celdas de texto.

La slide 36 de la clase agrega un punto: "Intentar plantear y probar tests de analogías".

## Aclaraciones de la clase (01:58:56 a 02:08:42 y 02:34:41 a 02:40:15)

- **Corpus:** tiene que ser propio y distinto del songs dataset; es lo que da el puntaje completo. Sugerencias:
  - libros de [Project Gutenberg](https://www.gutenberg.org/), de dominio público y sobre todo en inglés;
  - libros de [textos.info](https://www.textos.info/), en español;
  - un dataset propio, como papers de un tema o textos del trabajo.
- **Limpieza:** sacar los caracteres raros y los encabezados y pies de página que se repiten.
- **Segmentación:** cada documento tiene que ser una oración completa. La plantilla separa por línea porque en las canciones cada línea es una oración; en un libro, eso corta las oraciones por la mitad.
- **Hiperparámetros:** para cada uno, elegir un valor, comparar y explicar por qué se eligió y qué se observó. La docente dijo que los considera bastante.
  - `min_count`: con un corpus chico, un valor alto puede volar la mitad de las palabras.
  - `window`: con oraciones largas, probar ventanas mayores que 2.
  - `vector_size`: subirlo solo tiene sentido con un corpus grande.
  - `negative`: más negativos con un corpus chico; si hay overfitting, bajarlo puede ayudar.
  - `sg`: comparar CBOW y skip-gram es opcional.
- **Términos de interés:** idealmente al menos cinco. Buscar los más similares de cada uno y analizar cuáles tienen sentido, cuáles no y cuáles son overfitting del texto.
- **Gráfico:** con la función `reduce_dimensions` de la plantilla, en 2D o 3D, ajustando `MAX_WORDS`.
- **Clusters:** encontrar al menos tres y explicar por qué se agruparon: si tiene sentido o si es una limitación del modelo.
- **Capturas:** t-SNE no da siempre lo mismo, así que la docente no va a ver los mismos grupos. Sacar una captura de cada grupo y pegarla en una celda de texto con el comentario.
- **Conclusiones:** comentar y explicar cada punto de la consigna.

## ✎ Ideas para resolverlo

- **Corpus:** varios libros de un mismo autor, para tener volumen y que los clusters digan algo de ese autor (personajes, lugares, temas). En español sirven los de textos.info; en inglés, por ejemplo, las novelas de Jane Austen o de Dickens en Gutenberg.
- **Pipeline:** limpiar (cortar el encabezado y el pie de Gutenberg) → unir las líneas → `nltk.sent_tokenize` → tokenizar cada oración en minúsculas → `Word2Vec`.
- **Experimentos:** entrenar un modelo base y después cambiar un hiperparámetro por vez (`sg`, `window`, `vector_size`, `min_count`, `negative`, `epochs`). Comparar los vecinos de las mismas palabras y mostrarlo en una tabla.
- **Términos:** elegir palabras de frecuencia media y de temas distintos: un personaje, un lugar, un sentimiento, un objeto, un verbo. Sumar alguna palabra polisémica para mostrar la limitación de tener un solo vector por palabra.
- **Analogías:** probar algunas con `most_similar(positive=[...], negative=[...])`, por ejemplo con pares de personajes (hombre y mujer) o con países y capitales si el corpus los tiene. Si fallan, explicar por qué.
- **Gráfico:** en vez de las primeras `MAX_WORDS` palabras, que son las más frecuentes y en su mayoría stopwords, graficar las palabras sin stopwords o las de interés con sus vecinos.
- Más detalles prácticos en el [resumen de la clase 2](../../clases/02-word-embeddings/resumen.md#-para-los-desafíos).
