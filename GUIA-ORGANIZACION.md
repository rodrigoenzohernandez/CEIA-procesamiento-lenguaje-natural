# Guía de organización del repo

Cómo está armado este repo de estudio. Es la misma organización del repo de Agricultura Digital, adaptada a esta materia: acá no hay multiple choice ni TP final, sino 4 desafíos de código. El ejemplo concreto es la [clase 1](clases/01-introduccion-vectorizacion/resumen.md).

## Idea general

- **Un repo por materia**, todo en español.
- **Una carpeta por clase** con el material original y un resumen propio.
- El resumen tiene que cubrir **todos** los temas de la clase y dejar a mano lo que piden los desafíos.
- **Una carpeta por desafío** con la consigna, lo que se aclaró en clase y la resolución.
- El `README.md` es la portada: datos de la materia, cronograma con links y evaluación.
- El `CLAUDE.md` guarda las convenciones, así cualquier sesión con Claude (o cualquier persona) arma los resúmenes igual.

## Estructura

```
README.md                     portada: datos de la materia, cronograma, evaluación
CLAUDE.md                     convenciones del repo (las lee Claude Code)
GUIA-ORGANIZACION.md          esta guía
.gitignore                    .DS_Store, temporales de Office (~$*) y checkpoints de Jupyter
clases/
  01-introduccion-vectorizacion/
    resumen.md                resumen de la clase
    red-conceptual.md         mapas conceptuales en Mermaid
    material/                 material original, renombrado
      slides-vectorizacion-documentos.pdf
      notebook-vectorizacion-texto.ipynb
      transcripcion.txt
    img/                      slides que usa el resumen
      slide-61-tfidf.jpg
      ...
desafios/
  README.md                   reglas, fechas, notas máximas y estado de los 4 desafíos
  01-vectorizacion/
    README.md                 consigna, aclaraciones de la clase e ideas
    desafio-1.ipynb           plantilla de la materia, donde se resuelve
```

### Nombres

| Qué | Formato | Ejemplo |
|---|---|---|
| Carpeta de clase | `NN-tema-corto` (número con dos dígitos) | `01-introduccion-vectorizacion` |
| Slides | `slides-<tema>.pdf` | `slides-vectorizacion-documentos.pdf` |
| Notebook de la clase | `notebook-<tema>.ipynb` | `notebook-vectorizacion-texto.ipynb` |
| Transcripción | `transcripcion.txt` | |
| Imágenes | `slide-NN-tema.jpg` (NN = número de slide) | `slide-61-tfidf.jpg` |
| Carpeta de desafío | `NN-tema-corto` | `01-vectorizacion` |

Todo en kebab-case, sin tildes ni espacios: así los links de Markdown no se rompen.

## Flujo por clase

1. **Actualizar el repo de la materia** (`../procesamiento_lenguaje_natural`, rama `nov_2026`) con `git pull`.
2. **Crear la carpeta** `clases/NN-tema/` con `material/` e `img/`.
3. **Copiar el material** (slides y notebook de `Clase N/`) a `material/` y renombrarlo.
4. **Guardar la transcripción** de Meet en `material/transcripcion.txt`. Si viene como notas de Gemini:

   ```bash
   pandoc material/notas-gemini.docx -t plain | sed -n '/^Transcripción/,$p' > material/transcripcion.txt
   ```

5. **Exportar las slides como imágenes** para leerlas y para insertarlas en el resumen:

   ```bash
   pdftoppm -jpeg -scale-to-x 1280 -scale-to-y -1 material/slides-tema.pdf img/slide
   ```

   Después se renombran las que se usan (`slide-61-tfidf.jpg`) y se borran las demás.
6. **Escribir el resumen** (ver abajo).
7. **Armar la red conceptual** con mapas en Mermaid.
8. **Si la clase presenta un desafío:** crear `desafios/NN-tema/` con la plantilla del repo de la materia y un `README.md` con la consigna y lo que se aclaró en clase. Sumarlo a la tabla de `desafios/README.md`.
9. **Actualizar el `README.md`**: completar la fila del cronograma con los links al resumen, la red conceptual y la grabación.
10. **Commit** por paso grande: material y resumen, red conceptual, desafío.

## Fuentes: qué leer

- **La transcripción completa.** Ahí están los ejemplos, las preguntas de los compañeros y las aclaraciones sobre los desafíos, que no figuran en las slides.
- **Las slides como imágenes**, no con `pdftotext`: el texto extraído pierde diagramas, tablas y fórmulas.
- **El notebook de la clase:** los ejemplos de código y sus salidas (librerías, funciones, resultados).

## Estructura del `resumen.md`

1. **Datos de la clase:** docente, fecha, duración, link a la grabación y al material.
2. **Repaso rápido:** bullets con definiciones, fórmulas y cifras clave. Es lo que se lee 10 minutos antes de la clase siguiente o de encarar un desafío.
3. **Desarrollo por tema**, en el orden de la clase:
   - Cada tema con su timestamp de la grabación: `## 18. TF-IDF (01:56:50)`.
   - Todo lo que se dijo, no solo lo de las slides: ejemplos, preguntas, cifras, librerías y funciones.
   - Las slides con información visual insertadas con su pie:

     ```markdown
     ![Fórmula de TF-IDF](img/slide-61-tfidf.jpg)

     *Slide 61: TF-IDF = TF × IDF.*
     ```

   - Los datos de las imágenes (valores de tablas, etiquetas) también pasados a texto, y las fórmulas en LaTeX.
4. **Correcciones de la transcripción automática:** tabla "Dice la transcripción | Es" para nombres propios y términos técnicos mal reconocidos (ej. "Chamski" → Chomsky, "steming" → stemming).
5. **✎ Para los desafíos:** qué conceptos de la clase usa cada consigna y detalles prácticos.
6. **✎ Glosario.**
7. **✎ Preguntas de práctica**, con las respuestas ocultas:

   ```markdown
   <details>
   <summary>Respuestas</summary>

   1. ...
   </details>
   ```

**Regla del ✎:** todo lo que es elaboración propia (no se dijo en clase) se marca con ✎. Así queda claro qué es "lo que dijo la docente" y qué es agregado.

## Red conceptual

`red-conceptual.md` complementa el resumen con mapas en Mermaid (GitHub los renderiza):

- Una **vista general** de toda la clase y después un mapa por bloque de temas, con referencia a las secciones del resumen (§).
- Las flechas forman frases: *TF-IDF → multiplica → TF × IDF*.
- Colores fijos: azul = tema, verde = librería o herramienta, amarillo = dato para memorizar, gris punteado = concepto desarrollado en otro mapa.

## Herramientas

- `poppler` (`pdftoppm`, `pdfimages`): exportar slides e imágenes de los PDF. En macOS: `brew install poppler`.
- `pandoc`: pasar notas de Gemini (`.docx`) a texto plano, si las hay.
- Jupyter o Colab para los notebooks y los desafíos.
- Git + GitHub: un commit por paso, el README como índice navegable.
