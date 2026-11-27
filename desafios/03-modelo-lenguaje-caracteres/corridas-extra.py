"""Corridas aparte del desafío 3 (punto 3.7): cuatro configuraciones con lo mismo que el notebook.

Descarga, limpieza, preprocesamiento, `crear_modelo`, `PplCallback` y `entrenar` están copiados del
notebook (desafios/03-modelo-lenguaje-caracteres/desafio-3.ipynb), con el mismo learning rate, batch,
paso, épocas, early stopping y semilla. La semilla (42, la del notebook) se fija antes de cada
configuración: en el notebook se fija una vez al principio, y el estado exacto del generador al crear
cada modelo depende de las celdas anteriores, así que no se puede replicar.

Uso (con TensorFlow 2.21 y Keras 3.15, las versiones con las que corrió el notebook):
    python -I -u corridas-extra.py            # corrida completa, ~1 h en CPU
    python -I -u corridas-extra.py --smoke    # prueba de humo: 2 épocas y 4 lotes

Escribe resultados.json (resultados_smoke.json con --smoke) al lado del script después de cada
configuración. Si se corta, al relanzarlo saltea las configuraciones que ya están en el archivo; para
empezar de cero, borrarlo.

La salida de la corrida que se usó en el notebook está en corridas-extra.txt.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # menos mensajes de TensorFlow al importarlo

import re
import sys
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import tensorflow as tf
import keras
from keras import layers

SMOKE = '--smoke' in sys.argv
SALIDA = Path(__file__).with_name('resultados_smoke.json' if SMOKE else 'resultados.json')
SEMILLA = 42

tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow {tf.__version__} · Keras {keras.__version__} · {"GPU: " + gpus[0].name if gpus else "sin GPU: entrena en CPU"}')
print('Prueba de humo: 2 épocas y 4 lotes' if SMOKE else 'Corrida completa')

# ---------------------------------------------------------------- 1. Corpus (punto 1 del notebook)
URLS = [
    'https://www.gutenberg.org/files/2000/2000-0.txt',
    'https://www.gutenberg.org/cache/epub/2000/pg2000.txt',
    'https://aleph.pglaf.org/2/0/0/2000/2000-0.txt',
    'https://gutenberg.pglaf.org/2/0/0/2000/2000-0.txt',
]


def descargar(urls):
    for url in urls:
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(url, timeout=60) as respuesta:
                texto = respuesta.read().decode('utf-8')
            print(f'OK  {url} · {len(texto):,} caracteres en {time.perf_counter() - t0:.0f} s')
            return texto
        except Exception as e:
            print(f'--  {url} · {e}')
    raise RuntimeError('No se pudo descargar el libro de ninguna de las URLs')


raw_text = descargar(URLS)
texto = raw_text.replace('\r\n', '\n')
libro = texto[texto.index('\n', texto.index('*** START OF')) + 1:texto.index('*** END OF')]
parte1 = libro[libro.index('Capítulo primero.'):libro.index('\nFinis\n')]
assert 900_000 < len(parte1) < 1_100_000, 'No se encontró bien la primera parte: revisar las marcas del texto descargado'

# ---------------------------------------------------------------- 2. Preprocesamiento (punto 2)
def normalizar(texto):
    """Une las líneas de cada párrafo con un espacio, deja un salto de línea entre párrafos y pasa a minúsculas."""
    parrafos = (' '.join(p.split()) for p in re.split(r'\n\s*\n', texto))
    return '\n'.join(p for p in parrafos if p).lower()


article_text = normalizar(parte1)
max_context_size = 100

chars_vocab = sorted(set(article_text))
vocab_size = len(chars_vocab)
char2idx = {k: v for v, k in enumerate(chars_vocab)}
idx2char = {v: k for k, v in char2idx.items()}


def encode(text):
    """Texto → índices. Pasa a minúsculas y saltea los caracteres fuera del vocabulario."""
    return np.array([char2idx[ch] for ch in text.lower() if ch in char2idx], dtype=np.int32)


tokenized_text = encode(article_text)
p_val = 0.1
num_val = int(np.ceil(len(tokenized_text) * p_val / max_context_size))
train_text = tokenized_text[:-num_val * max_context_size]
val_text = tokenized_text[-num_val * max_context_size:]


def ventanas(tokens, largo, paso):
    """Ventanas de largo+1 tokens que empiezan cada `paso` tokens: X es la ventana sin el último, y sin el primero."""
    W = sliding_window_view(tokens, largo + 1)[::paso]
    return W[:, :-1].copy(), W[:, 1:].copy()


paso_train = 10
X, y = ventanas(train_text, max_context_size, paso_train)
X_val, y_val = ventanas(val_text, max_context_size, max_context_size)

# Las mismas cifras que imprime el notebook: si no coinciden, el texto descargado es otro.
assert (len(article_text), vocab_size, X.shape, X_val.shape) == (990_347, 53, (89_115, 100), (990, 100)), \
    f'el dataset no coincide con el del notebook: {len(article_text)=}, {vocab_size=}, {X.shape=}, {X_val.shape=}'
print(f'Dataset igual al del notebook: {len(article_text):,} caracteres, vocabulario de {vocab_size}, X {X.shape}, X_val {X_val.shape}')

# ---------------------------------------------------------------- 3. Modelo, callback y entrenamiento (punto 3)
UNIDADES, BATCH, LR = 256, 256, 3e-3
EPOCAS, PACIENCIA = 12, 3

if SMOKE:
    EPOCAS = 2
    X, y, X_val, y_val = X[:4 * BATCH], y[:4 * BATCH], X_val[:64], y_val[:64]


def crear_modelo(celda, unidades=UNIDADES, capas=1, dropout=0.0):
    model = keras.Sequential()
    model.add(layers.Input(shape=(None,), dtype='int32'))
    # One-hot sin parámetros entrenables: la fila i de la identidad es el vector one-hot del carácter i.
    model.add(layers.Embedding(vocab_size, vocab_size, embeddings_initializer='identity', trainable=False))
    for i in range(capas):
        model.add(celda(unidades, return_sequences=True))  # many-to-many: una salida por carácter
        if dropout and i < capas - 1:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=LR))
    return model


def parametros(model):
    """Parámetros entrenables (no cuenta la matriz identidad del one-hot)."""
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


class PplCallback(keras.callbacks.Callback):
    """Perplejidad de validación = exp(entropía cruzada de validación que calcula Keras).

    Corta el entrenamiento si no mejora en `patience` épocas (early stopping) y, al terminar, deja el modelo
    con los pesos de la mejor época (en memoria: no guarda archivos).
    """

    def __init__(self, patience=PACIENCIA):
        super().__init__()
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.history_ppl, self.min_score, self.best_epoch, self.wait = [], np.inf, 0, 0
        self.t0 = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.t_epoch = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        ppl_train, ppl_val = np.exp(logs['loss']), np.exp(logs['val_loss'])
        self.history_ppl.append((ppl_train, ppl_val))
        mejora = ppl_val < self.min_score
        if mejora:
            self.min_score, self.best_epoch, self.wait = ppl_val, epoch + 1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        print(f'  época {epoch + 1:>2} · {time.perf_counter() - self.t_epoch:3.0f} s · perplejidad '
              f'train {ppl_train:6.3f} · validación {ppl_val:6.3f}{"  ← mejor" if mejora else ""}')

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        self.train_time = time.perf_counter() - self.t0


resultados = {}


def entrenar(nombre, model):
    """Entrena con early stopping por perplejidad de validación y guarda el modelo y su historia."""
    print(f'{nombre}: {parametros(model):,} parámetros entrenables')
    cb = PplCallback()
    model.fit(X, y, validation_data=(X_val, y_val), epochs=EPOCAS, batch_size=BATCH, callbacks=[cb], verbose=0)
    print(f'  → mejor perplejidad de validación {cb.min_score:.3f} (época {cb.best_epoch}) · {cb.train_time / 60:.1f} min')
    resultados[nombre] = {'model': model, 'cb': cb}


# ---------------------------------------------------------------- 4. Las cuatro configuraciones
def simple_rnn_512_clipnorm():
    model = crear_modelo(layers.SimpleRNN, unidades=512)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate=LR, clipnorm=1.0))
    return model


def simple_rnn_512_lr_chico():
    model = crear_modelo(layers.SimpleRNN, unidades=512)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=1e-3))
    return model


CONFIGURACIONES = {
    'SimpleRNN(512) + clipnorm=1.0': simple_rnn_512_clipnorm,
    'SimpleRNN(512) con learning_rate=1e-3': simple_rnn_512_lr_chico,
    '2 × GRU(256) sin dropout': lambda: crear_modelo(layers.GRU, capas=2),
    '2 × GRU(256) + Dropout(0.4)': lambda: crear_modelo(layers.GRU, capas=2, dropout=0.4),
}

guardado = json.loads(SALIDA.read_text(encoding='utf-8')) if SALIDA.exists() and not SMOKE else {}
for nombre, construir in CONFIGURACIONES.items():
    if nombre in guardado:
        print(f'{nombre}: ya está en {SALIDA.name}, la salteo')
        continue
    keras.utils.set_random_seed(SEMILLA)
    entrenar(nombre, construir())
    cb = resultados[nombre]['cb']
    ppl = np.array(cb.history_ppl)
    guardado[nombre] = {
        'parametros': parametros(resultados[nombre]['model']),
        'mejor_epoca': cb.best_epoch,
        'mejor_perplejidad_val': round(float(cb.min_score), 4),
        'epocas_corridas': len(ppl),
        'perplejidad_train': [round(float(v), 4) for v in ppl[:, 0]],
        'perplejidad_val': [round(float(v), 4) for v in ppl[:, 1]],
        'tiempo_min': round(cb.train_time / 60, 2),
    }
    tmp = SALIDA.with_suffix('.tmp')
    tmp.write_text(json.dumps(guardado, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(SALIDA)  # reemplazo atómico: si se corta mientras escribe, queda el archivo anterior entero
    print(f'  guardado en {SALIDA}')

print('Listo.')
