# Syntax Project

Etiquetador PoS para euskera y catalán con datos de Universal Dependencies.

## Notebook `POS_Tagger.ipynb`
- Carga los CSV UD (train/dev/test) con `load_ud_csv` y valida longitudes palabra/tag.
- Explora la distribución de tags y entrena un HMM de NLTK.
- Evalúa token-level accuracy en test y muestra un ejemplo de Viterbi sobre una frase.
- Calcula la probabilidad conjunta de una secuencia etiquetada y genera muestras aleatorias del HMM.
- Compara con taggers `Default`, `Unigram`, `Bigram` y `Trigram` con backoff.

## Estructura del repositorio
- `datasets/ud_basque` y `datasets/ud_catalan`: splits `train/dev/test` en CSV.
- `POS_Tagger.ipynb`: notebook principal con entrenamiento, evaluación y demos.
- `utils/conll_to_csv.py`: script para convertir `.conllu` a CSV (usa `pyconll`, espera ficheros UD en `../../UD_Basque-BDT/`).
