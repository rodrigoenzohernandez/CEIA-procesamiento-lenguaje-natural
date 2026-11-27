# Procesamiento del Lenguaje Natural I — CEIA (FIUBA) 2026

Materia de la Especialización en Inteligencia Artificial (FIUBA). Repo de estudio: resúmenes de clase, redes conceptuales y resolución de los desafíos.

- **Modalidad:** virtual sincrónica, jueves 19:00–22:00, con dos recreos de 15 minutos.
- **Período:** 22 de octubre al 10 de diciembre de 2026, 8 clases. Las fechas de las clases 7 y 8 son estimadas (jueves consecutivos).
- **Docentes:** MSc. Josselyn Ordóñez (clases 1–4) y MSc. Fernando Silva (clases 5–8).
- **Material oficial:** [repo de la materia](https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/procesamiento_lenguaje_natural), rama `nov_2026` (la default; las otras ramas son cohortes anteriores).

## Cronograma

| # | Fecha | Docente | Tema | Desafío | Resumen | Grabación |
|---|-------|---------|------|---------|---------|-----------|
| 1 | 22/10 | Josselyn Ordóñez | Introducción a NLP, vectorización de documentos | [Desafío 1](desafios/01-vectorizacion/README.md) | [resumen](clases/01-introduccion-vectorizacion/resumen.md) · [red conceptual](clases/01-introduccion-vectorizacion/red-conceptual.md) | |
| 2 | 29/10 | Josselyn Ordóñez | Word embeddings | [Desafío 2](desafios/02-word-embeddings/README.md) | [resumen](clases/02-word-embeddings/resumen.md) · [red conceptual](clases/02-word-embeddings/red-conceptual.md) | |
| 3 | 05/11 | Josselyn Ordóñez | Redes recurrentes: Elman, LSTM y GRU | | [resumen](clases/03-redes-recurrentes/resumen.md) · [red conceptual](clases/03-redes-recurrentes/red-conceptual.md) | |
| 4 | 12/11 | Josselyn Ordóñez | Modelos de lenguaje y generación de secuencias | [Desafío 3](desafios/03-modelo-lenguaje-caracteres/README.md) | [resumen](clases/04-modelo-lenguaje/resumen.md) · [red conceptual](clases/04-modelo-lenguaje/red-conceptual.md) | |
| 5 | 19/11 | Fernando Silva | CNNs para PLN | | [resumen](clases/05-redes-convolucionales/resumen.md) · [red conceptual](clases/05-redes-convolucionales/red-conceptual.md) | [video](https://drive.google.com/file/d/1EKmgCjjgFSpRgurthGhRLThnvdze4_OW/view?usp=sharing) |
| 6 | 26/11 | Fernando Silva | Modelos seq2seq (encoder-decoder) | [Desafío 4](desafios/04-traductor-seq2seq/README.md) | [resumen](clases/06-seq2seq/resumen.md) · [red conceptual](clases/06-seq2seq/red-conceptual.md) | |
| 7 | 03/12 | Fernando Silva | Mecanismo de atención, Transformers | | | |
| 8 | 10/12 | Fernando Silva | Grandes modelos de lenguaje | | | |

## Evaluación

- **4 desafíos de código**, individuales o en pareja, que se presentan en las clases 1, 2, 4 y 6. La nota de la materia es el promedio de los cuatro.
- Cada desafío tiene nota máxima 10 si se entrega dentro de las dos semanas siguientes a su presentación; después la nota máxima baja.
- **Para aprobar hay que entregar los 4.** Si falta uno no se aprueba, aunque el promedio dé más de 4.
- Fechas, notas máximas, forma de entrega y estado de cada uno: [desafios/](desafios/README.md).

## Estructura

```
clases/NN-tema/
  resumen.md          resumen de la clase (transcripción + slides + notebook)
  red-conceptual.md   mapas conceptuales en Mermaid
  material/           slides, notebook de la clase, transcripción
  img/                slides que usa el resumen
desafios/
  README.md           reglas, fechas y estado de los 4 desafíos
  NN-tema/            consigna, aclaraciones y resolución de cada desafío
```

Detalle de la organización: [GUIA-ORGANIZACION.md](GUIA-ORGANIZACION.md).
