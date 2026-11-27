# Red conceptual — Clase 6: seq2seq y encoder-decoder

Complemento del [resumen](resumen.md). Primero hay una vista general de toda la clase y después un mapa por bloque de temas, en el orden en que se dieron. Los números de sección (§) remiten al resumen.

**Cómo leerla**
- Cada flecha forma una frase: *Encoder → resume en s → Decoder*. Una flecha sin texto quiere decir "incluye".
- Las flechas punteadas unen conceptos de temas distintos.
- Colores: azul = tema · verde = librería, función, dataset o herramienta · amarillo = dato para memorizar · gris punteado = concepto que se desarrolla en otro mapa o en otra clase.

> Todo el contenido sale de la clase. La organización y las frases de las flechas son propias.

## Vista general

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    CLA["Clasificar · clase 5<br>CNN o RNN many to one: salida fija"]
    S2S["① Seq2seq<br>encoder → vector → decoder"]
    LMC["② LM condicional<br>argmax p(y∣x, θ) · datos emparejados"]
    DIN["③ El decoder por dentro<br>softmax · SOS y EOS · teacher forcing"]
    NB1["④ Notebook de la suma<br>one-hot · bloques A a F"]
    NB2["⑤ Notebook del traductor<br>embeddings · EOS · greedy"]
    CB["⑥ Desafío 4 y cuello de botella"]
    TRF["Transformer · clase 7"]

    CLA -->|para generar hace falta el| S2S
    S2S -->|se formaliza como un| LMC
    S2S -->|genera paso a paso con| DIN
    DIN -->|se practica en el| NB1
    NB1 -->|con palabras y largo variable pasa a ser el| NB2
    NB2 -->|es la base del desafío y tiene un| CB
    CB -->|se resuelve con atención y lleva al| TRF

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class S2S,LMC,DIN,NB1,NB2,CB tema
    class CLA,TRF puente
```

## 1. De clasificar a generar (§0 a §3)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    CNN["CNN · clase 5<br>I love this movie → positivo"] -->|no alcanza para| GEN["Generar texto<br>I love AI → Me encanta la IA"]
    GEN -->|paradigma| S2S["Sequence to sequence"]
    S2S -->|su arquitectura es el| ED["Encoder-decoder"]
    M1["Many to one<br>me siento feliz → positivo"] -->|sin la capa final es el| ENC["Encoder<br>resume la entrada en un vector"]
    MM["Many to many<br>POS tagging: una salida por palabra"] -->|da una salida por paso, como el| DEC["Decoder<br>generador autorregresivo"]
    ED --> ENC
    ED --> DEC
    ENC -->|le pasa el vector al| DEC
    DEC -->|arranca con| SOS["SOS o BOS"]
    DEC -->|termina al generar| EOS["EOS"]
    SOS -.->|se agregan al| VOC["Vocabulario"]
    EOS -.-> VOC
    S2S -->|sirve para| APL["Chatbots · traducción<br>completar imágenes · text2img"]
    APL -->|hoy se resuelven con| LLM["LLMs · clase 8"]
    S2S -->|con atención se vuelve un| TRF["Transformer · clase 7"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class GEN,S2S,ED,M1,MM,ENC,DEC,SOS,EOS,VOC,APL tema
    class CNN,LLM,TRF puente
```

## 2. La traducción como modelo de lenguaje condicional (§4 a §7)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    TRA["Traducir<br>Hola mi nombre es Juan Pérez"] -->|es buscar la secuencia más probable| ARG["y′ = argmax p(y∣x, θ)<br>θ: los pesos · p: la red"]
    TRA -->|el largo cambia entre idiomas: hace falta el| EOS["EOS"]
    LM["Language model · clase 4<br>p(w) = ∏ p(wₜ ∣ w₁ … wₜ₋₁)"] -->|condicionado por la entrada es un| LMC["LM condicional<br>p(w∣x) = ∏ p(wₜ ∣ x, w₁ … wₜ₋₁)"]
    ARG --> LMC
    LMC -->|se entrena con| PAR["Pares de entrada y salida<br>aprendizaje supervisado"]
    PAR --> EJ["La terminal · Rosetta<br>menú: piedra = stone"]
    PAR --> ANKI["Anki · Tatoeba<br>un par por línea"]
    PAR --> FLK["Flickr30k<br>imagen + descripción"]
    PAR -->|escasean para las| ORI["Lenguas originarias<br>aimara: 20.000 pares mal · 150.000 algo mejor"]
    ANKI -->|entrena un| MT["Traductor<br>Kunst kann nicht … → Artistry can't …"]
    FLK -->|entrena un| CAP["Image captioning<br>encoder CNN + decoder"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class TRA,ARG,EOS,LMC,PAR,EJ,MT,CAP tema
    class ANKI,FLK herr
    class ORI dato
    class LM puente
```

## 3. El encoder-decoder por dentro (§8 a §10)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    PAL["Palabras de entrada"] -->|entran como| EMB["Embeddings"]
    EMB --> ENC["Encoder LSTM"]
    ENC -->|resume en s: los estados h y c| DEC["Decoder LSTM"]
    UNI["units: 32, 64, 256<br>el ancho de la flecha horizontal"] -.->|es el tamaño de h y c| ENC
    SOS["SOS"] -->|es la primera entrada del| DEC
    DEC -->|cada paso termina en una| SM["Softmax sobre el vocabulario<br>cada palabra es una clase"]
    SM -->|se elige la más probable| SIG["Palabra siguiente<br>tom → likes → beer → EOS"]
    SIG -->|en inferencia vuelve como entrada del| DEC
    TF["Teacher forcing · al entrenar<br>entra la palabra correcta"] -->|reemplaza la predicha en el| DEC
    SIG -->|un error se arrastra| INF["Inferencia sin referencia<br>exposure bias"]
    INF -->|se evalúa con| BLEU["BLEU · métricas de n-gramas"]
    PPL["Perplejidad<br>4 o 5 está bien en el TP3"] -.->|es para modelos de lenguaje, no para| INF

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class PAL,EMB,ENC,DEC,SOS,SM,SIG,TF,INF tema
    class BLEU herr
    class UNI,PPL dato
```

## 4. Notebook de la suma (§11 a §15)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    DAT["Datos sintéticos<br>5 … 300 → +15 · 20 secuencias de 3"] --> VOC["Vocabulario de 64<br>63 valores + el 0 como SOS"]
    VOC -->|LabelEncoder + to_categorical| OH["One-hot"]
    OH --> X1["X1 · entrada del encoder<br>5, 10, 15"]
    OH --> X2["X2 · teacher forcing<br>SOS, 20, 25"]
    OH --> TG["target<br>20, 25, 30"]
    X1 --> A["Bloque A · encoder<br>LSTM con return_state → h, c"]
    X2 --> B["Bloque B · decoder<br>LSTM con return_sequences + Dense softmax"]
    A -->|initial_state| B
    B --> C["Bloque C · modelo de entrenamiento"]
    TG -->|categorical_crossentropy| C
    C -->|150 épocas con LSTM de 64| OVF["Sobreajuste<br>loss 4,15 → 0,06 · val_loss 4,13 → 8,0"]
    A -.->|mismas capas| D["Bloque D · encoder de inferencia"]
    B -.->|misma LSTM y misma Dense| E["Bloque E · decoder de inferencia<br>token + h, c → predicción + h, c"]
    D --> F["Bloque F · bucle de 3 pasos<br>la predicción vuelve como entrada"]
    E --> F
    F -->|para 20, 25, 30 da| PRED["35, 40, 45<br>pero es un par de train"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class VOC,OH,X1,X2,TG,A,B,C,D,E,F tema
    class DAT herr
    class OVF,PRED dato
```

## 5. Notebook del traductor (§16 a §19)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    DS["spa-eng · 10.000 pares<br>inglés → español"] --> TRI["Tres secuencias<br>entrada · target + EOS · SOS + target"]
    TRI -->|Tokenizer de 8.000 palabras| IDX["Índices<br>EN 4.977 · ES 7.757 palabras"]
    IDX -->|pad_sequences| PAD["Padding<br>pre en el encoder · post en el decoder"]
    GLV["GloVe de 50 dimensiones<br>211 palabras en cero"] -->|matriz congelada del| ENC["Encoder<br>Embedding + Dropout 0,3 + LSTM 256"]
    PAD --> ENC
    PAD --> DEC["Decoder<br>Embedding propio + LSTM 256 + Dense softmax"]
    ENC -->|h y c como initial_state| DEC
    DEC -->|sparse_categorical_crossentropy y Adam| FIT["Entrenamiento<br>ReduceLROnPlateau · EarlyStopping"]
    FIT --> ACC["0,51 en train · 0,34 en validación<br>cortó en la época 38 · pesos de la 32"]
    FIT --> INF["Inferencia greedy<br>argmax en cada paso"]
    INF -->|se detiene con el| EOS["EOS"]
    INF --> EJ["mi hermano me dio un poco<br>aciertos parciales"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class TRI,IDX,PAD,ENC,DEC,FIT,INF,EOS tema
    class DS,GLV herr
    class ACC,EJ dato
```

## 6. Desafío 4 y cierre (§20 y §21)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    TRAD["Traductor de la clase"] -->|extenderlo y analizarlo es el| D4["Desafío 4<br>nota 10 hasta el 10/12"]
    D4 --> P1["Más datos y<br>secuencias más largas"]
    D4 --> P2["Cantidad de neuronas<br>de las capas recurrentes"]
    D4 --> P3["5 traducciones<br>e interpretación"]
    D4 -.->|opcional, con puntos extra| OP["Embeddings en los dos idiomas<br>muestreo aleatorio · PyTorch"]
    TRAD -->|tiene un| CB["Cuello de botella<br>toda la entrada en un vector fijo"]
    CB -->|se resuelve con un| ATT["Mecanismo de atención"]
    ATT -->|si el encoder, el puente y el decoder son atención| TRF["Transformer · clase 7"]
    TF["Teacher forcing"] -.->|es el outputs shifted right del| TRF

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class TRAD,D4,P1,P2,P3,CB,ATT,TF tema
    class OP dato
    class TRF puente
```
