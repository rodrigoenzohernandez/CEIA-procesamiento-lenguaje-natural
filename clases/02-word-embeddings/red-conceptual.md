# Red conceptual — Clase 2: Word embeddings

Complemento del [resumen](resumen.md). Primero hay una vista general de toda la clase y después un mapa por bloque de temas, en el orden en que se dieron. Los números de sección (§) remiten al resumen.

**Cómo leerla**
- Cada flecha forma una frase: *CBOW → conviene con → corpus grandes*. Una flecha sin texto quiere decir "incluye".
- Las flechas punteadas unen conceptos de temas distintos.
- Colores: azul = tema · verde = librería, función o herramienta · amarillo = dato para memorizar · gris punteado = concepto que se desarrolla en otro mapa o en otra clase.

> Todo el contenido sale de la clase. La organización y las frases de las flechas son propias.

## Vista general

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    BOW["① Problemas de BoW<br>sin sinónimos · vectores esparsos"]
    EMB["① Embeddings<br>densos · semánticos ·<br>espacio continuo"]
    MET["② Métodos estáticos<br>Word2Vec · GloVe · fastText"]
    EVA["③ Explorar y evaluar<br>analogías · polisemia · sesgos"]
    ENT["④ Entrenamiento<br>CBOW · skip-gram ·<br>negative sampling"]
    GEN["⑤ Gensim y t-SNE"]
    DES["⑤ Desafío 2<br>embeddings de un corpus propio"]
    DIN["⑥ Más allá de los estáticos<br>dinámicos · oraciones · multimodales"]

    BOW -->|se resuelven con| EMB
    EMB -->|se crean con| MET
    MET -->|se exploran con| EVA
    MET -->|se aprenden con| ENT
    ENT -->|ya implementado en| GEN
    GEN -->|se usa en el| DES
    EVA -->|el desafío también pide| DES
    EVA -->|la polisemia lleva a los| DIN

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    class BOW,EMB,MET,EVA,ENT,GEN,DES,DIN tema
```

## 1. De Bag of Words a embeddings (§0 a §4)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    BOW["Conteo · one-hot · TF-IDF<br>Bag of Words · clase 1"]
    BOW -->|problema 1| ORT["Textos parecidos son ortogonales<br>colectivo vs bus → coseno 0"]
    BOW -->|problema 2| DIM["Una dimensión por palabra<br>+180.000 en inglés · esparsos"]
    ORT -->|lo resuelve el| EMB["Embedding<br>representación densa en un<br>espacio vectorial continuo"]
    DIM -->|lo resuelve el| EMB
    EMB -->|no solo de palabras| OTR["Imágenes · voz · oraciones"]
    EMB -->|propiedades buscadas| PRO["Compacto · similitud semántica ·<br>comprensión del contexto"]
    PRO -.->|la tercera solo la tienen los| CTX["Embeddings contextuales<br>Transformers · mapa 6"]
    EMB -->|intuición| FRU["Frutas como vectores<br>dulzor · tamaño · rojo ·<br>tropical · circular"]
    FRU -->|con la tabla| FM["frutilla ≈ manzana<br>coseno 0.95"]
    FRU -->|en la práctica| AUT["Dimensiones y valores<br>aprendidos automáticamente"]
    AUT -->|cuántas| NDIM["100 a 300 en clase ·<br>OpenAI 1536 y 3072 ·<br>según cómputo y corpus"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class BOW,EMB,PRO,FRU,AUT tema
    class ORT,DIM,FM,NDIM dato
    class CTX puente
```

## 2. Métodos estáticos: Word2Vec, GloVe y fastText (§5 a §10)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    HIP["Hipótesis distributiva<br>conocerás a una palabra<br>por las compañías que tiene"]
    HIP -->|la aplica con una red neuronal| W2V["Word2Vec · Google · 2013<br>aprende leyendo texto,<br>sin diccionarios"]
    HIP -->|la aplica contando| GLO["GloVe · Stanford · 2014<br>estadísticas globales"]
    W2V -->|arquitecturas| ARQ["CBOW · skip-gram<br>mapa 4"]
    W2V -.->|se explora en el| PRO["projector.tensorflow.org<br>Word2Vec 10K · 200 dim · PCA"]
    GLO -->|arma una| COO["Matriz de co-ocurrencia<br>pares dentro de una ventana"]
    COO -->|ejemplo con canciones| AMV["amor–vida = 15<br>ventana de 5 a cada lado"]
    GLO -->|tokeniza por| PAL["Palabras completas"]
    GLO -->|entrenado con| GCO["Wikipedia · Common Crawl ·<br>Gigaword 5"]
    W2V -->|misma arquitectura,<br>otra tokenización| FT["fastText · Facebook · 2016"]
    FT -->|tokeniza en| NG["N-gramas de caracteres, de 3 a 6<br>prefijos · sufijos · raíces"]
    NG -->|ventana que se desliza de a uno| TOD["Todas → to · od · da · as"]
    NG -->|permite vectorizar| OOV["Palabras nunca vistas<br>holaaaa → hola<br>texto informal de redes"]
    W2V -->|después llegan| DIN["ELMo · BERT · 2018 en adelante<br>embeddings dinámicos · mapa 6"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class HIP,W2V,GLO,FT,COO,NG tema
    class PRO herr
    class AMV,GCO,TOD dato
    class ARQ,DIN puente
```

## 3. Explorar y evaluar embeddings (§11 a §15)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    PRE["Embeddings preentrenados<br>Colab 2.a"]
    PRE --> FTX["fastText cc.en.300<br>2 M palabras · 300 dim"]
    PRE --> GLV["GloVe Twitter<br>1,2 M palabras · 50 dim"]
    PRE -->|se comparan con una| MAT["Matriz de similitud coseno<br>cosine_similarity + heatmap"]
    MAT -->|en fastText| HH["hello–hi 0.70 ·<br>laptop–computer 0.66"]
    MAT -->|en GloVe| MOR["morning parece un saludo<br>0.79 con hello"]
    PRE -->|se evalúan con| ANA["Tests de analogías<br>París − Francia ≈<br>Madrid − España"]
    ANA -->|en el notebook| KQ["king − man + woman<br>king 0.73 · queen 0.65"]
    PRE -->|limitación| POL["Polisemia<br>un vector por palabra"]
    POL -->|según el corpus| AM["apple salió fruta ·<br>mouse salió animal"]
    PRE -->|limitación| SES["Sesgos del corpus<br>man más cerca de<br>money y power"]
    SES -->|riesgo en| RIE["Créditos · filtrado de CVs ·<br>minorías"]
    SES -->|se mitigan con| MIT["Revisar los vectores ·<br>promediar · corpus curados"]
    POL -.->|lo resuelven los| DIN["Embeddings dinámicos · mapa 6"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class PRE,ANA,POL,SES tema
    class MAT herr
    class HH,MOR,KQ,AM dato
    class DIN puente
```

## 4. Entrenamiento: CBOW, skip-gram y negative sampling (§16 a §19)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    W2V["Word2Vec<br>la entrada es un one-hot"]
    W2V --> CB["CBOW<br>contexto → palabra central<br>muchos a uno"]
    W2V --> SG["Skip-gram<br>palabra central → contexto<br>uno a muchos"]
    CB -->|ejemplo| GAT["El ___ maúlla en el tejado<br>→ gato"]
    CB -->|capas| CBL["Embedding V × N →<br>Lambda: promedio →<br>Dense con softmax sobre V"]
    CB -->|conviene con| CBU["Corpus grandes · rápido ·<br>palabras comunes"]
    CBL -->|con 12.425 palabras y 100 dim| P25["2,5 M parámetros<br>mitad embedding · mitad densa"]
    SG -->|arma| PAR["Pares target–contexto<br>have–I · have–will · have–orange"]
    SG -->|capas| SGL["2 Embedding → producto punto →<br>Dense con sigmoide: 1 o 0"]
    SG -->|conviene con| SGU["Corpus chicos · vocabulario<br>específico · palabras raras"]
    CBL -->|al final se guarda la| MEM["Matriz de embeddings<br>una fila por palabra"]
    SGL -->|de sus dos matrices se usa la del| TGT["Target, por convención"]
    SG -->|agrega negativos al azar| NS["Negative sampling<br>par positivo + K palabras<br>fuera del contexto"]
    CB -->|también puede usar| NS
    NS -->|evita calcular la| SMX["Softmax sobre todo el vocabulario<br>vocab × dim = millones"]
    NS -->|regla| KN["Corpus chico → más negativos ·<br>además regulariza"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class W2V,CB,SG,NS,CBL,SGL,MEM tema
    class GAT,P25,PAR,KN,TGT dato
```

## 5. Gensim, t-SNE y el desafío 2 (§20 a §25)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    GEN["Gensim · Word2Vec<br>Colab 2.b"]
    GEN -->|hiperparámetros| HP["min_count · window ·<br>vector_size · negative ·<br>workers · sg"]
    GEN -->|ejemplo de la plantilla| BEA["Beatles: 1846 documentos ·<br>445 palabras · skip-gram de<br>300 dim · 20 épocas"]
    BEA -->|most_similar| MS["four → five, three, six ·<br>money → can't, buy"]
    BEA -->|palabra nueva| KE["KeyError"]
    BEA -->|se grafica con| TSNE["t-SNE<br>no lineal · estocástico ·<br>solo para visualizar"]
    TSNE -.->|a diferencia de| PCA["PCA<br>lineal"]
    TSNE -->|clusters| CL["Números · saludos ·<br>yellow submarine = overfitting"]
    GEN -->|se usa en el| DES["Desafío 2<br>nota 10 hasta el 12/11"]
    DES -->|corpus propio| COR["Gutenberg · textos.info ·<br>un documento = una oración"]
    DES -->|pide| PID["5 términos o más · gráfico 2D ·<br>3 clusters o más, con capturas ·<br>hiperparámetros justificados"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class TSNE,DES tema
    class GEN,HP herr
    class BEA,MS,KE,CL,PID dato
    class PCA puente
```

## 6. Cierre: más allá de los embeddings estáticos (§26 a §28)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    EST["Embeddings estáticos<br>Word2Vec · GloVe · fastText"]
    EST -->|superan a BoW en| SUP["Semántica · vectores densos"]
    EST -->|no resuelven la| POL["Polisemia<br>banco · apple · mouse"]
    POL -->|la resuelven los| DIN["Embeddings dinámicos<br>BERT · Transformers con atención<br>el vector depende de la frase"]
    DIN -.-> T78["Clases 7 y 8"]
    EMB["Embeddings de algo<br>más que palabras"]
    EMB --> SEN["Sentence embeddings<br>un vector por oración<br>o párrafo · Doc2Vec"]
    SEN -->|son la base de| RAG["RAG<br>párrafos más cercanos a la<br>pregunta → al prompt del LLM"]
    RAG -.->|se ve en| PLN2["PLN II, optativa"]
    EMB --> MUL["Multimodales<br>imágenes y texto en<br>un mismo espacio"]
    MUL -->|ejemplo| CLIP["CLIP · OpenAI<br>imágenes con sus leyendas"]
    EMB -->|aplicaciones| APP["Buscadores · recomendaciones ·<br>base de los LLMs"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class EST,DIN,EMB,SEN,MUL,RAG tema
    class CLIP herr
    class SUP,POL dato
    class T78,PLN2 puente
```
