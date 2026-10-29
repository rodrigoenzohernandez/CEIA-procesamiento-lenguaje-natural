# CLAUDE.md

Repo de estudio de la materia Procesamiento del Lenguaje Natural I (CEIA, FIUBA). Todo en español. Se usa para resúmenes de clase, redes conceptuales, resolver los desafíos y aprender.

## Fuentes

- **Repo oficial de la materia:** https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/procesamiento_lenguaje_natural, clonado en `../procesamiento_lenguaje_natural`. La rama de esta cohorte es `nov_2026` (la default); las otras son cohortes anteriores. Hacer `git pull` ahí antes de buscar material nuevo: trae las slides (`Clase N/*.pdf`), los notebooks de clase y las plantillas de los desafíos (`Desafíos/Desafio_N.ipynb`).
- **Transcripción:** la pega el usuario en el chat (transcripción de Meet). Guardarla tal cual en `material/transcripcion.txt`; para no reescribirla, se puede extraer del JSONL de la sesión (`~/.claude/projects/<proyecto>/<sesión>.jsonl`). Si en cambio llegan las notas de Gemini (`.docx`): `pandoc material/notas-gemini.docx -t plain | sed -n '/^Transcripción/,$p' > material/transcripcion.txt`.

## Convenciones

- Una carpeta por clase: `clases/NN-tema-corto/` con `resumen.md`, `red-conceptual.md`, `material/` e `img/` (imágenes de las slides que usa el resumen, `slide-NN-tema.jpg`).
- Material original renombrado en kebab-case: `slides-<tema>.pdf`, `notebook-<tema>.ipynb`, `transcripcion.txt`.
- Desafíos: `desafios/NN-tema/` con `README.md` (consigna, aclaraciones de la clase e ideas) y el notebook de la resolución. Fechas, notas máximas y estado de cada uno en `desafios/README.md`.
- Al agregar una clase: completar la fila del cronograma en `README.md` (resumen, red conceptual, grabación) y, si la clase presenta un desafío, cargar su consigna en `desafios/`.

## Resúmenes

La materia se aprueba con los 4 desafíos (no hay multiple choice): el resumen tiene que cubrir **todos** los temas de la clase y dejar a mano lo que piden los desafíos.

- Fuentes: leer la **transcripción completa**, las slides **como imágenes** (pdftotext pierde gráficos, tablas y fórmulas) y el **notebook de la clase** (ejemplos de código y sus salidas). Para recorrer muchas slides, armar hojas de 6 slides con PIL en el scratchpad y abrir a tamaño completo las de letra chica.
- Estructura de `resumen.md`: datos de la clase → repaso rápido → desarrollo por tema en el orden de la clase (slides + todo lo dicho: ejemplos, cifras, librerías y funciones) → tabla de correcciones de la transcripción automática → ✎ para los desafíos → ✎ glosario → ✎ preguntas de práctica con respuestas en `<details>`.
- Imágenes: insertar en cada tema las slides con información visual (diagramas, tablas, fórmulas, ejemplos), con un pie `*Slide N: ...*`. Exportarlas con `pdftoppm -jpeg -scale-to-x 1280 -scale-to-y -1`. Pasar a texto los datos de las imágenes (valores de las tablas, etiquetas).
- Fórmulas en LaTeX (bloques ` ```math `), que GitHub renderiza.
- Timestamps `(hh:mm:ss)` de la transcripción al inicio de cada tema.
- Marcar con ✎ todo lo que sea elaboración propia y no se dijo en clase.

## Desafíos

- Lo que más se evalúa es el análisis: cada experimento va con su interpretación y conclusiones escritas. Se puede usar IA.
- Se entregan con el link a un Colab o a un repo de GitHub por el formulario (ver `desafios/README.md`).
