# Red conceptual — Clase 3: Redes recurrentes (Elman, LSTM y GRU)

Complemento del [resumen](resumen.md). Primero hay una vista general de toda la clase y después un mapa por bloque de temas, en el orden en que se dieron. Los números de sección (§) remiten al resumen.

**Cómo leerla**
- Cada flecha forma una frase: *LSTM → agrega una → memoria de largo plazo C*. Una flecha sin texto quiere decir "incluye".
- Las flechas punteadas unen conceptos de temas distintos.
- Colores: azul = tema · verde = librería, función o herramienta · amarillo = dato para memorizar · gris punteado = concepto que se desarrolla en otro mapa o en otra clase.

> Todo el contenido sale de la clase. La organización y las frases de las flechas son propias.

## Vista general

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    EMB["Embeddings · clase 2<br>Word2Vec · GloVe · FastText"]
    ORD["① El orden importa<br>el perro mordió al hombre"]
    RNN["② RNN<br>estado interno que se<br>actualiza paso a paso"]
    ELM["② Celda de Elman<br>SimpleRNN"]
    CFG["③ Configuraciones<br>many-to-one · many-to-many ·<br>seq2seq · multicapa"]
    BPT["④ BPTT<br>el gradiente se desvanece"]
    K3A["⑤ Notebook 3a<br>Keras · return_sequences"]
    LST["⑥ LSTM<br>memoria de largo plazo C"]
    GRU["⑦ GRU<br>LSTM reducida"]
    BID["⑦ Bidireccional<br>contexto futuro"]
    ADD["⑧ Adding problem<br>notebook 3b"]
    GEN["⑨ Generación de texto<br>siguiente token"]

    EMB -->|promediados pierden el| ORD
    ORD -->|lo tienen en cuenta las| RNN
    RNN -->|celda básica| ELM
    ELM -->|se combina en| CFG
    ELM -->|se entrena con| BPT
    ELM -->|en código| K3A
    BPT -->|olvida el inicio:<br>lo resuelve la| LST
    LST -->|versión más barata| GRU
    ELM -->|si hace falta el futuro| BID
    LST -->|se comparan en el| ADD
    GRU --> ADD
    CFG -->|many-to-one para la| GEN

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class ORD,RNN,ELM,CFG,BPT,K3A,LST,GRU,BID,ADD,GEN tema
    class EMB puente
```

## 1. Del embedding a la secuencia (§1 a §3)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    EMB["Embeddings · clase 2<br>vectores densos con significado"]
    EMB --> W2V["Word2Vec<br>CBOW · skip-gram"]
    EMB --> GLO["GloVe<br>matriz de co-ocurrencias"]
    EMB --> FT["FastText<br>n-gramas de caracteres"]
    W2V -->|CBOW promedia el contexto<br>y se pierde el| ORD["Orden"]
    ORD -->|ejemplo| PER["El perro mordió al hombre =<br>El hombre mordió al perro<br>similitud coseno 1"]
    ORD -->|lo tienen en cuenta las| RNN["Redes recurrentes"]
    RNN -->|historia| HIS["Elman 1990 · LSTM 1997 ·<br>adopción en 2014 · estado del<br>arte en traducción en 2016"]
    HIS -->|tardaron por el| HW["Hardware"]
    HIS -.->|las destronaron los| TRA["Transformers · clase 7"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class ORD,RNN tema
    class PER,HIS,HW dato
    class EMB,W2V,GLO,FT,TRA puente
```

## 2. RNN y celda de Elman (§4 a §8)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    RNN["RNN<br>neurona con estado interno<br>que recorre la secuencia"]
    RNN -->|sirve para| APP["Series de tiempo · audio y<br>speech to text · video · bots ·<br>NER · traducción · imágenes"]
    RNN -->|en teoría cualquier largo, pero| LAR["Rinde peor cuanto<br>más larga la secuencia"]
    RNN -->|intuición| CAJ["La cajera<br>x = precio · h = total ·<br>misma regla en cada paso"]
    RNN -->|celda básica| ELM["Celda de Elman<br>h = tanh(W_hh·h + W_hx·x + b)<br>z = h"]
    ELM -->|en Keras| SRNN["SimpleRNN"]
    ELM -->|usa en todos los pasos| SHA["Las mismas matrices<br>pesos compartidos:<br>secuencias de cualquier largo"]
    ELM -->|ejemplo animado| FRA["Vivo en Francia…<br>no sabía nada de ___"]
    FRA -->|dimensiones| DIM["x de 3 · h de 4 ·<br>W_x 4×3 · W_h 4×4 ·<br>softmax al final"]
    ELM -->|parámetros| PAR["d_h·(d_h + d_x + 1)<br>de 3 a 4 da 32"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class RNN,ELM,CAJ,FRA tema
    class SRNN herr
    class DIM,PAR,LAR,SHA dato
```

## 3. Configuraciones y multicapa (§9 y §10)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    CFG["Configuraciones de la RNN"] --> M1["Many-to-one<br>solo el estado final"]
    CFG --> MM["Many-to-many<br>una salida por paso"]
    CFG --> S2S["Seq2seq<br>salida de otro largo"]
    M1 -->|ejemplos| M1E["Siguiente palabra · sentimiento ·<br>clasificar noticias ·<br>precio de una acción"]
    MM -->|ejemplos| MME["NER · etiquetado de palabras"]
    S2S -->|ejemplos| S2E["Traducción · chatbot"]
    CFG -->|se apilan en una| MUL["RNN multicapa<br>más profundidad, conceptos<br>más abstractos"]
    MUL -->|cada capa le pasa<br>a la siguiente| HPA["Su h de cada paso"]
    MUL -->|la última capa define la| CFG
    HPA -->|en Keras| RS["return_sequences=True<br>en todas menos la última"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    class CFG,M1,MM,S2S,MUL tema
    class RS herr
```

## 4. Entrenamiento y memoria corta (§11, §12 y §14)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    BPT["BPTT<br>backpropagation through time"]
    BPT -->|los pesos se actualizan| FIN["Recién al terminar la secuencia<br>L = L1 + … + LT"]
    BPT -->|recorre hacia atrás<br>todos los pasos| PRO["Producto de muchas derivadas"]
    PRO -->|factores menores que 1| VAN["Vanishing gradient"]
    PRO -->|factores mayores que 1| EXG["Exploding gradient"]
    VAN -->|lo agravan| DOS["La profundidad (capas)<br>y el largo de la secuencia"]
    VAN -->|se pierden las| INI["Primeras palabras<br>teléfono malogrado"]
    INI -->|ejemplo| FRA["Francia se diluye:<br>gana un idioma genérico"]
    INI -->|forward| FWD["La información se diluye<br>en el estado"]
    INI -->|backward| BWD["Los pesos casi no<br>aprenden del inicio"]
    INI -.->|lo resuelve la| LST["LSTM · mapa 6"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class BPT,VAN,EXG,INI tema
    class FIN,DOS,FRA dato
    class LST puente
```

## 5. Notebook 3a en Keras (§13)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    DAT["Datos sintéticos<br>1, 2, 3 → 6 … 43, 44, 45 → 132<br>15 ejemplos · forma (3, 1)"]
    DAT --> SEQ["Sequential"]
    SEQ --> SR["SimpleRNN(64)<br>4.224 parámetros"]
    SR --> DEN["Dense(1)<br>65 parámetros"]
    DEN -->|predice| PRE["50 + 51 + 52 → 153.5"]
    SR -->|para apilar| RS["return_sequences=True"]
    RS -->|si falta| ERR["Error: la capa siguiente<br>espera una secuencia"]
    RS -->|la última no lo lleva en un| M1["Many-to-one"]
    SR -->|cambiando la celda| LSM["LSTM(64)<br>16.896 = 4 × 4.224"]
    SR -->|envuelta en| BID["Bidirectional<br>8.448 = 2 × 4.224"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class DAT,M1 tema
    class SEQ,SR,DEN,RS,LSM,BID herr
    class PRE,ERR dato
```

## 6. LSTM (§14)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    LST["LSTM · 1997<br>long short-term memory"]
    LST -->|agrega una| CM["Memoria de largo plazo C<br>el cuaderno"]
    LST -->|mantiene el| HM["Estado oculto h<br>lo que uno tiene en mente"]
    LST -->|4 subceldas| GAT["Olvido f · entrada i ·<br>candidata C̃ · salida o"]
    GAT -->|actualizan la| CM
    CM -->|se actualiza con| FC["cₜ = fₜ∘cₜ₋₁ + iₜ∘C̃ₜ<br>solo productos y sumas"]
    CM -->|la salida filtra| HO["hₜ = oₜ ∘ tanh(cₜ)"]
    HO --> HM
    FC -->|camino por donde viajan<br>la información y el gradiente| PATH["Menos vanishing gradient"]
    PATH -.->|como las| RES["Skip connections · ResNet<br>Visión por Computadora II"]
    LST -->|costo| X4["4 veces los parámetros<br>de Elman · 8 matrices"]
    LST -->|en Keras| KL["LSTM(…)"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class LST,CM,HM,GAT,PATH tema
    class KL herr
    class FC,HO,X4 dato
    class RES puente
```

## 7. GRU y RNN bidireccional (§15 y §16)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    LST["LSTM · mapa 6"] -->|su costo lleva a la| GRU["GRU<br>gated recurrent unit"]
    GRU -->|no tiene C:<br>usa solo| HH["El estado h"]
    GRU -->|compuertas| RZ["Reset r · update z<br>y la candidata h̃"]
    RZ -->|combina| HF["hₜ = (1 − zₜ)∘h̃ₜ + zₜ∘hₜ₋₁"]
    GRU -->|costo| X3["~3 veces los parámetros"]
    GRU -->|a veces le gana a la LSTM| PRB["Probar las dos"]

    RNN["RNN tradicional<br>solo ve el pasado"] -->|no resuelve| VIC["¿Victoria es persona o ciudad?<br>la pista está a futuro"]
    VIC -->|lo resuelve la| BID["RNN bidireccional<br>dos direcciones en paralelo"]
    BID -->|combina los estados| CMB["Concatenar: 4 + 4 = 8 ·<br>sumar o promediar: 4"]
    BID -->|costo| X2["2 veces los parámetros"]
    BID -->|no sirve para| NOS["Generar texto ·<br>procesar en vivo"]
    BID -->|en Keras| KB["Bidirectional(celda)"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class GRU,BID,RZ tema
    class KB herr
    class X3,X2,CMB,HF,NOS dato
    class LST,RNN puente
```

## 8. Adding problem (§17)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    ADD["Adding problem<br>paper de la LSTM · 1997"]
    ADD -->|cada paso tiene| FEA["Número aleatorio entre 0 y 1<br>+ marca 0 o 1"]
    ADD -->|target| SUM["Suma de los 2 marcados<br>uno en cada mitad"]
    ADD -->|datos| DAT["T = 50 · 6.000 muestras ·<br>32 unidades"]
    ADD -->|baseline: predecir la media| BAS["MSE ≈ 0.16"]
    ADD -->|con T = 50| RES["SimpleRNN ≈ baseline ·<br>LSTM y GRU aprenden ·<br>ganó la GRU"]
    ADD -->|según T| TT["SimpleRNN: 0.01 con T = 10<br>y 0.157 con T = 50"]
    RES -->|conclusión| CON["Con dependencias largas<br>hacen falta compuertas"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class ADD,SUM,CON tema
    class DAT,BAS,RES,TT,FEA dato
```

## 9. Generación de texto y cierre (§18 y §19)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    LM["Modelo de lenguaje<br>P(siguiente palabra ∣ anteriores)"]
    LM -->|genera| TOK["Un token a la vez<br>streaming en Gemini, Claude, GPT"]
    TOK -->|usado tal cual| RJ["Primero responde<br>y después justifica"]
    RJ -->|causa| ALU["Alucinaciones"]
    ALU -->|las mitigan los| RAZ["Modelos que razonan<br>antes de responder"]
    LM -->|pasos · clase 4| PAS["Segmentación → tokenización →<br>truncado → aumento de datos →<br>padding"]
    LM -->|arquitectura del curso| REC["Redes recurrentes"]
    LM -.->|hoy| TRA["Transformers y LLMs<br>clases 7 y 8"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class LM,TOK,RJ,RAZ,REC tema
    class PAS,ALU dato
    class TRA puente
```
