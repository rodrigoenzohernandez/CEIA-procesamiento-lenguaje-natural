# Red conceptual — Clase 5: CNN para PLN

Complemento del [resumen](resumen.md). Primero hay una vista general de toda la clase y después un mapa por bloque de temas, en el orden en que se dieron. Los números de sección (§) remiten al resumen.

**Cómo leerla**
- Cada flecha forma una frase: *Functional → permite → cortar la red*. Una flecha sin texto quiere decir "incluye".
- Las flechas punteadas unen conceptos de temas distintos.
- Colores: azul = tema · verde = librería, función o herramienta · amarillo = dato para memorizar · gris punteado = concepto que se desarrolla en otro mapa o en otra clase.

> Todo el contenido sale de la clase. La organización y las frases de las flechas son propias.

## Vista general

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    PLAN["Plan de las clases 5 a 8<br>CNN → seq2seq → atención → LLMs"]
    FUN["① API Functional<br>ramas · varias entradas · cortar la red"]
    REP["② Representaciones de texto<br>clásicas · deep learning · Transformers"]
    CNN["③ CNN para secuencias<br>Conv1D · padding · max pooling"]
    ARQ["④ CNN bifurcada<br>GloVe congelado + filtros de 3, 4 y 5"]
    LIM["⑤ Límites de las CNN<br>solo relaciones locales"]
    ATT["Atención · clases 7 y 8"]

    PLAN -->|arranca con la| CNN
    REP -->|cada token es una fila de la| CNN
    REP -->|BoW + MLP es el baseline de la| ARQ
    FUN -->|permite armar la| ARQ
    CNN -->|se combina en la| ARQ
    ARQ -->|choca con sus| LIM
    LIM -->|se resuelve con la| ATT

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class PLAN,FUN,REP,CNN,ARQ,LIM tema
    class ATT puente
```

## 1. Plan de las clases 5 a 8 (§0)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    RNN["RNN · clase 3<br>procesa de a un token, en orden"] -->|es lenta para| MUCHO["Entrenar con mucho texto"]
    MUCHO -->|alternativa| CNN["CNN para PLN · clase 5<br>filtros que se calculan en paralelo"]
    CNN -->|su limitación es una| VEN["Ventana corta"]
    CNN -->|le sigue| S2S["Seq2seq · clase 6<br>encoder-decoder"]
    S2S -->|es la excusa para entender el| TRF["Transformer · clase 7<br>encoder-decoder + atención"]
    TRF -->|es la arquitectura de los| LLM["LLMs · clase 8"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class CNN,MUCHO tema
    class VEN dato
    class RNN,S2S,TRF,LLM puente
```

## 2. Keras: Sequential y Functional (§1 y §2)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    SEQ["Sequential<br>pila lineal de capas"] -->|no puede armar| BIF["Ramas en paralelo<br>Inception: filtros 1×1 · 3×3 · 7×7"]
    FUN["Functional<br>cada capa es una función sobre un tensor"] -->|sí puede armar| BIF
    FUN --> MULT["Varias entradas y salidas"]
    FUN -->|permite| CORTE["Cortar la red<br>salida de una capa intermedia"]
    CORTE -->|sirve para sacar| EMB["Embeddings<br>vector de 64 de oculta_2"]
    FUN -->|se lee de derecha a izquierda, en zigzag| ZZ["x = Dense(64)(x)"]
    FUN -->|se cierra como un sándwich con| MOD["Model(inputs, outputs)"]
    SEQ -.->|en el notebook| NB["MLP sobre MNIST<br>784 → 128 → 64 → 10"]
    FUN -.->|en el notebook| NB
    NB -->|con la misma semilla| ACC["0,9749 en test con las dos APIs<br>109.386 parámetros"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class SEQ,FUN,BIF,MULT,CORTE,EMB tema
    class ZZ,MOD,NB herr
    class ACC dato
```

## 3. Representaciones de texto (§3 a §10)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    VEC["Vectorizar bien<br>junto con la loss, lo más importante"] --> CLA["Clásicas<br>baseline · pocos recursos"]
    VEC --> DL["Deep learning<br>preentrenados"]
    VEC --> TRF["Transformers<br>embeddings que cambian con el contexto"]
    CLA --> OH["One-hot y multi-hot<br>tamaño del vocabulario · sparse"]
    CLA --> BOW["BoW<br>cuenta · no ve el orden"]
    CLA --> COO["Co-ocurrencia<br>ventana de 4 a cada lado · borochi · Firth"]
    BOW -->|en el notebook| NBB["BoW + MLP en IMDB<br>5.000 palabras · 0,88 en test"]
    COO -->|sus vectores largos y sparse se comprimen con| SVD["SVD truncado, como un PCA<br>vectores cortos y densos"]
    COO -->|en vez de contar, predecir| DL
    DL --> W2V["Word2Vec<br>CBOW: contexto → palabra<br>Skip-gram: palabra → contexto"]
    DL --> GLV["GloVe · FastText"]
    W2V -->|enmascarar es| AUTO["Aprendizaje autosupervisado"]
    W2V -->|sumados dan la| ORA["Representación de la oración"]
    DL -->|vector fijo para banco, lo mejoran los| TRF

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class VEC,CLA,DL,OH,BOW,COO,SVD,W2V,GLV,AUTO,ORA tema
    class NBB herr
    class TRF puente
```

## 4. Conv1D paso a paso (§11 a §14)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    MAT["Oración = matriz<br>una fila por token, con su embedding<br>7 × 4 en el ejemplo"] -->|la recorre un| KER["Filtro de k × d<br>3 × 4: cubre todo el embedding"]
    KER -->|baja de a una fila, stride 1| OUT["Feature map<br>suma de productos en cada ventana"]
    OUT -->|sin padding| V5["5 salidas · L − k + 1<br>t,d,r = −1.0"]
    PAD["Padding = 1 · same<br>ceros solo arriba y abajo"] -->|da| V7["7 salidas<br>∅,t,d = −0.6 · g,o,∅ = −0.5"]
    PAD -->|a los costados alteraría el| EMBD["Embedding de cada palabra"]
    KER -->|con varios filtros sale| COLS["Una columna por filtro"]
    COLS -->|max pooling global| MAXP["0.3 · 1.6 · 1.4<br>un valor por filtro"]
    NBC["Notebook Conv1D<br>Conv1D + set_weights + GlobalMaxPooling1D"] -->|reproduce las| V7
    NBC -->|y el max pooling da| M03["0.3"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class MAT,KER,OUT,PAD,EMBD,COLS tema
    class NBC herr
    class V5,V7,MAXP,M03 dato
```

## 5. La CNN bifurcada del notebook (§15 y §16)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    KIM["Kim 2014 · Zhang y Wallace 2015<br>filtros de 2, 3 y 4 en paralelo + 1-max pooling"] -.->|misma idea| IN
    IN["Input · 200 índices<br>pad_sequences post"] --> EMB["Embedding GloVe 100d<br>trainable=False: feature extraction"]
    MATR["Matriz de embeddings 20.000 × 100<br>GloVe si la palabra está · aleatorio si no"] -->|carga los pesos del| EMB
    EMB --> SD["SpatialDropout1D 0.2"]
    SD --> C3["Conv1D 64 · k = 3<br>BatchNorm · GlobalMaxPooling"]
    SD --> C4["Conv1D 64 · k = 4<br>BatchNorm · GlobalMaxPooling"]
    SD --> C5["Conv1D 64 · k = 5<br>BatchNorm · GlobalMaxPooling"]
    C3 --> CAT["Concatenate · 192"]
    C4 --> CAT
    C5 --> CAT
    CAT --> DR["Dropout 0.6"]
    DR --> OUT["Dense 1 · sigmoide"]
    OUT -->|en test| ACC["0,83 y seguía aprendiendo<br>early stopping con patience=1"]
    FUN["API Functional"] -.->|hace falta para las ramas| CAT

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class IN,EMB,C3,C4,C5,CAT,DR,OUT tema
    class MATR,SD herr
    class ACC dato
    class KIM,FUN puente
```

## 6. Límites de las CNN y atención (§17 y §18)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    CNN["CNN para texto"] -->|solo ve| LOC["Relaciones locales<br>ventanas chicas"]
    LOC -->|pierde| LEJ["El libro … es interesante<br>a 18 posiciones"]
    LEJ -->|agrandar el filtro| PAT["Patea el problema<br>la distancia no es fija"]
    CNN -->|no tiene| GLO["Atención global<br>no aprende a qué palabra mirar"]
    LEJ -->|hace falta otro mecanismo| ATT["Atención<br>transformación ponderada y paralela<br>de secuencias de embeddings"]
    ATT --> ED["Atención encoder-decoder<br>clases 6 y 7"]
    ATT --> SA["Self-attention · Q, K, V<br>base de los Transformers · clase 7"]
    SA -->|permitió los| LLM["LLMs · clase 8"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class CNN,LOC,GLO,ATT tema
    class LEJ,PAT dato
    class ED,SA,LLM puente
```
