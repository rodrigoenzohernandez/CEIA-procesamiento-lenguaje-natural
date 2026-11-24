# Red conceptual — Clase 4: Modelo de lenguaje con LSTM

Complemento del [resumen](resumen.md). Primero hay una vista general de toda la clase y después un mapa por bloque de temas, en el orden en que se dieron. Los números de sección (§) remiten al resumen.

**Cómo leerla**
- Cada flecha forma una frase: *greedy search → elige → la palabra más probable*. Una flecha sin texto quiere decir "incluye".
- Las flechas punteadas unen conceptos de temas distintos.
- Colores: azul = tema · verde = librería, función o herramienta · amarillo = dato para memorizar · gris punteado = concepto que se desarrolla en otro mapa o en otra clase.

> Todo el contenido sale de la clase. La organización y las frases de las flechas son propias.

## Vista general

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    REP["Repaso: RNN · LSTM · GRU<br>clase 3"]
    LM["① Modelo de lenguaje<br>predecir la siguiente palabra"]
    DAT["② Dataset<br>segmentar · tokenizar · truncar ·<br>aumentar · padding"]
    PPL["③ Perplejidad<br>exp de la entropía cruzada"]
    ARQ["④ Arquitectura<br>embedding + 2 LSTM + densas"]
    GEN["⑤ Generación<br>greedy · beam search · muestreo ·<br>temperatura · top-k · top-p"]
    DES["⑥ Desafío 3<br>modelo por caracteres"]

    REP -->|se usan para armar un| LM
    LM -->|necesita un| DAT
    LM -->|se mide con la| PPL
    DAT -->|entrena una| ARQ
    PPL -->|guía el entrenamiento de la| ARQ
    ARQ -->|genera texto con| GEN
    GEN -->|se practica en el| DES
    DAT -->|el desafío cambia la tokenización| DES

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class LM,DAT,PPL,ARQ,GEN,DES tema
    class REP puente
```

## 1. Modelos de lenguaje y LLMs (§1, §2, §4 y §12)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    LM["Modelo de lenguaje<br>P de la siguiente palabra<br>dadas las anteriores"]
    LM -->|es la misma tarea que el| PRE["Preentrenamiento de los LLMs"]
    PRE -->|después suman| RL["Aprendizaje por refuerzo<br>mejora matemática y código,<br>que son verificables"]
    LM -->|genera de a un| TOK["Token<br>el texto generado condiciona<br>lo que sigue"]
    TOK -->|por eso a veces| DIS["Responde, justifica<br>y se disculpa"]
    LLM["LLM comercial"] -->|tiene una| VEN["Ventana de contexto<br>más de 1 M de tokens"]
    VEN -->|se paga por| COS["Tokens de entrada y de salida"]
    VEN -->|si se llena| CMP["Error o lectura parcial ·<br>hoy se compacta o se resume"]
    LLM -->|escala| ESC["Open source: miles de millones ·<br>comerciales: billones"]
    ESC -->|variante| MOE["Mixture of experts<br>32B en total · 3B activos"]
    LM -.->|el de la clase tiene| CHI["1,3 M de parámetros · mapa 4"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class LM,PRE,RL,TOK,LLM,VEN,MOE tema
    class DIS,COS,CMP,ESC dato
    class CHI puente
```

## 2. Preparar el dataset (§3, §5 a §7 y §10)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    COR["Corpus<br>62 cuentos de Borges"]
    COR -->|1| SEG["Segmentación<br>regex después de . , ! ?<br>10.812 frases"]
    SEG -->|2| TOK["Tokenización por palabra<br>text_to_word_sequence +<br>Tokenizer · 15.002 palabras"]
    TOK -->|se elige una| VEN["Ventana de contexto<br>histograma de largos → 10"]
    VEN -->|si la frase es más larga| TRU["3 · Truncado<br>1 + M − N ventanas<br>21.707 secuencias"]
    TRU -->|4| AUM["Aumento de datos<br>prefijos: Todas · Todas las · …"]
    AUM -->|5| PAD["Padding con ceros a la izquierda<br>lo que se predice va a la derecha"]
    PAD -->|da| SEQ["178.448 secuencias de 11<br>X = 10 primeras · y = la 11.ª"]
    TOK -.->|en el desafío 3| CAR["Tokenización por caracteres · mapa 6"]
    SEQ -.->|sirve para cualquier modelo| LLM["También lo usan los LLMs,<br>con subpalabras"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class SEG,VEN,TRU,AUM,PAD tema
    class TOK herr
    class COR,SEQ dato
    class CAR,LLM puente
```

## 3. Probabilidad de una secuencia y perplejidad (§8, §9 y §13)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    BAB["Biblioteca de Babel<br>todas las combinaciones<br>de caracteres"]
    BAB -->|el modelo aprende cuáles| SEN["Tienen sentido"]
    SEN -->|de a una palabra, con la| CAD["Regla de la cadena<br>P de la secuencia = producto<br>de condicionales"]
    CAD -->|su inversa normalizada es la| PPL["Perplejidad<br>cantidad media de<br>palabras posibles"]
    PPL -->|equivale a| EXP["exp de la entropía cruzada"]
    PPL -->|modelo al azar| VV["Perplejidad = V"]
    PPL -->|ejemplo| FOX["fox: 44 · the: 2 ·<br>media geométrica 16,49"]
    PPL -->|mejor que la| ACC["Accuracy<br>hay varias respuestas válidas"]
    EXP -->|se calcula en un| CB["Callback de Keras<br>desde cero o exp de la loss"]
    CB -->|en clase bajó| BAJ["De ~1.000 a ~8<br>ojo: es la de entrenamiento"]
    PPL -.->|no confundir con el| NS["Negative sampling · clase 2"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class BAB,CAD,PPL,EXP tema
    class CB herr
    class VV,FOX,BAJ,ACC dato
    class NS puente
```

## 4. La arquitectura del notebook (§11)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    IN["Entrada: 10 IDs de palabras"]
    IN --> EMB["Embedding · 50 dim<br>entrenable desde cero"]
    EMB --> L1["LSTM 64<br>return_sequences=True<br>many-to-many hacia la capa siguiente"]
    L1 --> DR["Dropout 0.2<br>regulariza"]
    DR --> L2["LSTM 64<br>sin return_sequences<br>many-to-one"]
    L2 --> D32["Dense 32 · relu"]
    D32 --> OUT["Dense V+1 · softmax<br>el +1 es el token de padding"]
    OUT -->|se entrena con| CMP["Sparse categorical crossentropy ·<br>Adam · 50 épocas · batch 300"]
    EMB -.->|alternativa| PRE["Embeddings preentrenados<br>weights + trainable=False<br>Word2Vec · GloVe · fastText"]
    OUT -->|en total| PAR["1.309.793 parámetros<br>la mayoría en el embedding"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class EMB,L1,DR,L2,D32,OUT tema
    class CMP herr
    class IN,PAR dato
    class PRE puente
```

## 5. Generación y estrategias de decodificación (§14 a §18)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    GEN["Generación autorregresiva<br>la salida vuelve a entrar"]
    GEN -->|en el notebook| GS["generate_seq<br>La casa de la espuma que un remo…"]
    GEN -->|estrategia 1| GR["Greedy search<br>la más probable en cada paso<br>Andábamos sin rumbo fijo = 0.045"]
    GEN -->|estrategia 2| BS["Beam search<br>varias ramas · probabilidad conjunta<br>Andábamos sin buscarnos, pero = 0.108"]
    GR -->|es| DET["Determinista<br>siempre la misma salida"]
    BS -->|también es| DET
    GEN -->|estrategia 3| MU["Muestreo según las probabilidades<br>es lo que usan los LLMs"]
    MU -->|se controla con la| TEM["Temperatura<br>divide los logits"]
    TEM -->|T alta| ALT["Aplana: más variedad ·<br>riesgo de incoherencia"]
    TEM -->|T baja| BAJ["Afila: casi siempre<br>la misma palabra"]
    MU -->|se acota con| TK["Top-k: las k más probables ·<br>top-p: probabilidad acumulada"]
    DET -->|con modelos chicos cae en| LOOP["Loops<br>la casa de la casa de…"]
    MU -->|ayuda a salir de los| LOOP

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class GEN,GR,BS,MU,TEM,TK tema
    class GS herr
    class DET,ALT,BAJ,LOOP dato
```

## 6. Desafío 3 (§19)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    DES["Desafío 3<br>nota 10 hasta el 26/11 ·<br>lo corrige Fernando"]
    DES -->|tokeniza por| CAR["Caracteres<br>vocabulario de ~65 ·<br>contexto de 100"]
    CAR -->|por eso no lleva| EMB["Embedding: entran como one-hot"]
    DES -->|corpus| COR["Propio · puede ser el<br>del desafío 2"]
    DES -->|arquitecturas| ARQ["SimpleRNN · LSTM · GRU<br>capas y dropout"]
    ARQ -->|guiarse por la| PPL["Perplejidad de validación<br>callback con early stopping"]
    DES -->|generar con| GEN["Greedy · beam search determinista ·<br>beam search estocástico<br>con varias temperaturas"]
    DES -->|siempre| CON["Conclusiones en cada punto ·<br>contar también lo que no anduvo"]
    PPL -.-> M3["Mapa 3"]
    GEN -.-> M5["Mapa 5"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class DES,CAR,ARQ,PPL,GEN tema
    class COR,EMB,CON dato
    class M3,M5 puente
```
