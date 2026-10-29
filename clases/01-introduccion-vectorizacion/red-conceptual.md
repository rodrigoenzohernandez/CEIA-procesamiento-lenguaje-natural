# Red conceptual — Clase 1: Introducción a NLP y vectorización de documentos

Complemento del [resumen](resumen.md). Primero hay una vista general de toda la clase y después un mapa por bloque de temas, en el orden en que se dieron. Los números de sección (§) remiten al resumen.

**Cómo leerla**
- Cada flecha forma una frase: *TF-IDF → multiplica → TF × IDF*. Una flecha sin texto quiere decir "incluye".
- Las flechas punteadas unen conceptos de temas distintos.
- Colores: azul = tema · verde = librería, función o herramienta · amarillo = dato para memorizar · gris punteado = concepto que se desarrolla en otro mapa o en otra clase.

> Todo el contenido sale de la clase. La organización y las frases de las flechas son propias.

## Vista general

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    LEN["① Lenguaje<br>reglas (Chomsky) ·<br>uso y contexto (Wittgenstein)"]
    NLP["① NLP<br>representar · modelar ·<br>inferir · aprender"]
    HIS["② Historia<br>reglas → estadística →<br>deep learning → Transformers"]
    TAR["③ Aplicaciones y tareas<br>clasificación · tagging ·<br>generación · traducción"]
    PRE["④ Preprocesamiento<br>tokenización · normalización ·<br>stemming / lematización · stopwords"]
    VEC["⑤ Vectorización BoW<br>conteo · binario · TF-IDF"]
    SIM["⑥ Similitud coseno"]
    NB["⑦ Naïve Bayes"]
    DES["⑧ Desafío 1<br>20 Newsgroups"]
    FAM["⑨ Familias de vectorización<br>frecuencia · predicción · contexto"]

    LEN -->|lo estudia el| NLP
    NLP -->|evolucionó en| HIS
    NLP -->|resuelve| TAR
    NLP -->|arranca con el| PRE
    PRE -->|prepara el texto para la| VEC
    VEC -->|se comparan con| SIM
    VEC -->|alimentan a| NB
    SIM -->|se usa en el| DES
    NB -->|se usa en el| DES
    VEC -->|es la primera de las| FAM

    LEN -.->|el significado como uso<br>lleva a los embeddings| FAM

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    class LEN,NLP,HIS,TAR,PRE,VEC,SIM,NB,DES,FAM tema
```

## 1. Lenguaje y NLP (§1 a §4)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    LEN["Lenguaje"]
    LEN -->|depende de| FAC["Semántica · codificación ·<br>orden · contexto · cultura"]
    LEN -->|para la lingüística es| LIN["Símbolos + reglas<br>semántica · sintaxis · pragmática"]
    LIN -->|Chomsky propone la| GG["Gramática generativa<br>reglas que generan todas<br>las oraciones correctas"]
    GG -->|es la base del| NLPC["NLP clásico con reglas"]
    LIN -.->|¿un LLM son reglas aprendidas<br>o un loro estocástico?| LLM["LLMs · clases 7 y 8"]
    LEN -->|para la filosofía| WIT["Wittgenstein<br>el significado de una palabra<br>es su uso en el lenguaje"]
    WIT -->|ejemplo| BAN["banco: entidad financiera ·<br>mueble · banco de peces"]
    WIT -->|lleva a los| EMB["Embeddings · clase 2<br>estáticos o dinámicos"]
    LEN -->|se manifiesta como| MAN["Hablado (speech processing) ·<br>escrito · gestual (señas) ·<br>digital (foco del curso)"]
    LEN -->|lo procesa el| NLP["NLP<br>área de la IA que representa,<br>procesa, comprende y genera<br>lenguaje humano"]
    NLP -->|implica| CUA["1 Representación<br>2 Modelado<br>3 Inferencia<br>4 Aprendizaje"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class LEN,LIN,WIT,NLP tema
    class GG,CUA dato
    class LLM,EMB puente
```

## 2. Historia del NLP (§5)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    TUR["Test de Turing · 1950<br>¿una máquina puede hablar<br>como un humano?"]
    ELI["ELIZA · 1964<br>palabras clave +<br>preguntas predefinidas"]
    SMT["Statistical Machine<br>Translation · IBM · 1990"]
    DL["Deep learning · 2012"]
    TRA["Transformer · 2017<br>Attention Is All You Need<br>(Google)"]
    GEN["IA generativa y LLMs"]

    TUR -->|primeros intentos:<br>bots con reglas| ELI
    ELI -->|de reglas a<br>probabilidades| SMT
    SMT -->|4 pasos| PAS["fertility · NULL insertion ·<br>traducción léxica · distorsión"]
    SMT -->|sin pasos diseñados<br>a mano| DL
    DL -->|tardó por el| HW["Hardware<br>GPUs + texto digitalizado"]
    DL -->|3 ingredientes| ING["Hardware · datos ·<br>investigación"]
    DL -->|las redes recurrentes sufren| VAN["Desvanecimiento<br>del gradiente"]
    VAN -->|lo resuelve el| TRA
    TRA -->|procesa la secuencia| PAR["En paralelo<br>+ posición con embeddings"]
    TRA -->|da origen a la| GEN
    GEN -->|tres ramas| RAM["Encoder-only · encoder-decoder ·<br>decoder-only"]
    GEN -->|hoy| HOY["Modelos multimodales<br>· agentes"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class TUR,ELI,SMT,DL,TRA,GEN tema
    class PAS,HW,ING dato
```

## 3. Aplicaciones y tareas (§6 y §7)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    NLP["NLP"]
    NLP -->|aplicaciones| APP["Traducción · asistentes de voz ·<br>chatbots"]
    APP -->|ejemplos| HER["Google Translate · Alexa · Siri ·<br>ChatGPT · SimSimi"]
    APP -->|chatbot + acciones| AGE["Agente<br>busca en internet, manda mails"]
    NLP -->|tarea| CLA["Clasificación de texto"]
    CLA --> CEJ["Spam · tópicos de noticias ·<br>análisis de sentimiento"]
    NLP -->|tarea| TAG["Tagging<br>clasificación por palabra"]
    TAG --> POS["POS tagging<br>sustantivo · verbo · adjetivo"]
    TAG --> NER["NER<br>persona · lugar · organización · fecha"]
    NLP -->|tarea| GEN["Generación"]
    GEN --> GEJ["Autocompletado · chat · resumen"]
    NLP -->|tarea| MUL["Multimodalidad<br>texto, imagen y video en un<br>espacio vectorial común"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    class NLP,APP,CLA,TAG,GEN,MUL tema
    class HER herr
```

## 4. Preprocesamiento (§8 a §13)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    COR["Corpus"] -->|1| PRE["Preprocesamiento<br>limpieza"]
    PRE -->|2| FE["Feature engineering<br>vectorización · mapa 5"]
    FE -->|3| MOD["Modelo"]
    PRE --> TOK["Tokenización<br>segmentar + asignar un ID<br>la usa cualquier modelo"]
    TOK -->|no es solo split| ESC["D.C. · New York · doesn't ·<br>SARS-CoV-2 · chino · turco"]
    PRE --> NOR["Normalización<br>minúsculas · sin puntuación ·<br>sin caracteres raros"]
    NOR -->|depende de la tarea| DEP["Para clasificar sirve ·<br>en un chatbot la<br>puntuación importa"]
    PRE --> STE["Stemming<br>corta sufijos con reglas<br>rápido · raíces inválidas"]
    PRE --> LEM["Lematización<br>busca el lema en un diccionario<br>palabras válidas · más lenta"]
    STE -->|improving →| IM1["improv"]
    LEM -->|improving →| IM2["improve"]
    PRE --> STO["Stopwords<br>muy frecuentes,<br>poco significado"]
    STO -->|ojo| NO["La lista de NLTK saca el 'no':<br>'No estoy tan solo…' →<br>tan solo mismo"]
    NLTK["NLTK<br>word_tokenize · sent_tokenize ·<br>stopwords · SnowballStemmer ·<br>pos_tag · FreqDist"]
    SPA["spaCy<br>es_core_news_sm"]
    TOK -.->|en el notebook| NLTK
    STE -.->|en el notebook| NLTK
    STO -.->|en el notebook| NLTK
    LEM -.->|en el notebook| SPA

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class PRE,TOK,NOR,STE,LEM,STO tema
    class NLTK,SPA herr
    class IM1,IM2,NO dato
    class FE puente
```

## 5. Vectorización: Bag of Words y TF-IDF (§14 a §19)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart TD
    COR["Corpus<br>define el vocabulario"] -->|se divide en| DOC["Documentos<br>oración · párrafo · artículo"]
    DOC -->|se dividen en| TER["Términos<br>palabras"]
    TER -->|una dimensión por término| VEC["Vector de documento<br>tan largo como el vocabulario"]
    VEC --> CNT["Conteo<br>cuántas veces aparece"]
    VEC --> BIN["Binario<br>0 o 1 · como one-hot"]
    VEC --> TFI["TF-IDF<br>TF × log(N / DF)"]
    CNT -->|en scikit-learn| CV["CountVectorizer"]
    TFI -->|en scikit-learn| TV["TfidfVectorizer"]
    TFI -->|si el término está<br>en todos los documentos| CERO["IDF = 0<br>log(3/3) = 0"]
    CNT --> BOW["Bag of Words<br>no importa el orden"]
    BIN --> BOW
    TFI --> BOW
    CNT -->|las stopwords dominan| ZIP["Ley de Zipf<br>stopwords · palabras útiles ·<br>palabras raras"]
    ZIP -->|sirven las del| MED["Medio del ranking"]
    BOW -->|desventajas| DES["Esparsidad (más de 180.000<br>palabras en inglés) · sin orden ·<br>sin sinónimos (perro ≠ can)"]
    DES -.->|se resuelven con| EMB["Embeddings densos · clase 2"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class VEC,CNT,BIN,TFI,BOW,ZIP tema
    class CV,TV herr
    class CERO,MED,DES dato
    class EMB puente
```

## 6. Similitud entre documentos (§20)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    SIM["Similitud entre documentos"] --> EUC["Distancia euclidiana<br>distancia absoluta ·<br>depende de la magnitud"]
    SIM --> COS["Similitud coseno<br>A·B / (‖A‖ ‖B‖)<br>ángulo · ignora la magnitud"]
    EUC -->|problema| LAR["Los documentos largos parecen<br>lejos · las palabras frecuentes<br>dominan"]
    COS -->|se usa en NLP porque| DIR["La dirección codifica<br>el significado"]
    COS -->|valores| VAL["1 = misma dirección<br>0 = sin relación<br>−1 = sentido contrario"]
    VAL -->|ojo| ANT["−1 no son antónimos:<br>los antónimos están relacionados"]
    COS -->|en scikit-learn| SK["cosine_similarity"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class SIM,EUC,COS tema
    class SK herr
    class VAL,ANT dato
```

## 7. Naïve Bayes (§21 a §23)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    CLS["Clasificación de texto<br>vectores + etiquetas"] -->|modelo de ML clásico| NB["Naïve Bayes"]
    NB -->|usa el| BAY["Teorema de Bayes<br>P(clase) × Π P(palabra ∣ clase)<br>sin el denominador"]
    NB -->|naïve porque asume| IND["Palabras independientes"]
    NB -->|ejemplo| SPAM["'Dear Friend'<br>normal 0.09 vs spam 0.01"]
    NB -->|una palabra nunca vista<br>en una clase da| CER["Probabilidad 0<br>anula la clase"]
    CER -->|se evita con| LAP["Suavizado de Laplace<br>+α a los conteos"]
    NB -->|variante| MUL["Multinomial<br>cuántas veces aparece"]
    NB -->|variante| BER["Bernoulli<br>si aparece o no"]
    MUL -->|en scikit-learn| SK["MultinomialNB ·<br>ComplementNB"]
    NB -->|es un clasificador| BOW["Bag of Words<br>no depende del orden"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class CLS,NB,LAP,MUL,BER tema
    class SK herr
    class SPAM,IND dato
    class BOW puente
```

## 8. Desafío 1 (§24)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    DES["Desafío 1<br>20 Newsgroups · 20 clases"] -->|vectoriza con| TV["TfidfVectorizer<br>11.314 docs × 101.631 términos"]
    TV -->|consigna 1| C1["Similitud coseno<br>5 docs × sus 5 más similares"]
    TV -->|consigna 2| C2["Prototipos (zero-shot)<br>clase del doc de train<br>más similar"]
    TV -->|consigna 3| C3["MultinomialNB y ComplementNB<br>baseline: F1 macro 0.585"]
    TV -->|consigna 4| C4["Matriz transpuesta<br>similitud entre palabras"]
    C3 -->|no tocar| NGR["ngram_range"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef herr fill:#dcfce7,stroke:#166534,color:#052e16
    classDef dato fill:#fef9c3,stroke:#a16207,color:#3b2405
    class DES,C1,C2,C4 tema
    class TV herr
    class C3,NGR dato
```

## 9. Cierre: familias de vectorización y enfoques (§25)

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 40, "padding": 10}}}%%
flowchart LR
    FAM["Familias de vectorización"] --> FRE["Por frecuencia · esparsos<br>BoW · TF-IDF · co-ocurrencia"]
    FAM --> PRD["Por predicción · estáticos<br>Word2Vec · GloVe · FastText"]
    FAM --> CTX["Por contexto · dinámicos<br>ELMo · BERT · GPT"]
    FRE -.->|esta clase| C1["Clase 1"]
    PRD -.-> C2["Clase 2"]
    CTX -.-> C7["Transformers · clases 7 y 8"]

    TXT["Texto"] -->|en los tres enfoques| TOK["Tokenización"]
    TOK --> CL["NLP clásico<br>stopwords · stemming ·<br>BoW / TF-IDF · modelo de ML"]
    TOK --> NDL["NLP + deep learning<br>subpalabras (BPE, WordPiece) ·<br>embeddings · modelo de DL"]
    TOK --> LLM["LLMs<br>tokenizador propio del modelo ·<br>IDs de tokens · Transformer"]

    classDef tema fill:#dbeafe,stroke:#1e40af,color:#0b1b3f
    classDef puente fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray:4 3
    class FAM,FRE,PRD,CTX,TOK,CL,NDL,LLM tema
    class C1,C2,C7 puente
```
