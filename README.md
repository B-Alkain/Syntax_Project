# Syntax Project

Etiquetador PoS para euskera y catalán con datos de Universal Dependencies.

## Estructura rápida
- `main.py`: utilidades de evaluación (accuracy, per-tag accuracy) y función `get_data`.
- `model/hmm.py`: implementación del HMM propio (entrenamiento MLE y Viterbi).
- `POS_Tagger.ipynb`: notebook con experimentos: HMM NLTK vs HMM propio, n-gram taggers, gráficas y ejemplos cualitativos.
- `utils/conll_to_csv.py`: conversión de `.conllu` a CSV con `pyconll`.
- `datasets/ud_basque`, `datasets/ud_catalan`: splits train/dev/test en CSV.
- `tex_files/`: memoria en LaTeX (`main.tex`, capítulos en `chapters/`, figuras en `figuras/`).

## Notebook `POS_Tagger.ipynb`
- Carga los CSV UD (train/dev/test) con `load_ud_csv` y valida longitudes palabra/tag.
- Entrena HMM de NLTK y HMM propio, compara accuracies y n-gram taggers (default/unigram/bigram/trigram).
- Genera gráficas (se guardan en `tex_files/figuras/`) y ejemplos de Viterbi, probabilidad conjunta y muestras aleatorias.

## Editar y compilar la memoria LaTeX
1) Editar contenido: modificar los capítulos en `tex_files/chapters/` (introduction, methodology, results, conclusions). El archivo `tex_files/main.tex` solo orquesta el preámbulo y los `\input{...}`.
2) Generar PDF en la raíz del repo:
   - Con Makefile: `make main` crea `main.pdf`; `make summary` crea `summary.pdf`, ambos en la raíz.
   - Sin Makefile: `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. -jobname=main tex_files/main.tex` (cambia `-jobname` a `summary` si lo prefieres).

### Dependencias LaTeX (Windows/Linux)
- **Windows**: instalar MiKTeX (https://miktex.org/download) y añadir `pdflatex` al PATH. En VS Code, instalar la extensión LaTeX Workshop y compilar con el comando anterior desde el WSL/PowerShell.
- **Linux**: instalar TeX Live básico (`sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended latexmk`). Compilar con el comando anterior o `make`.
- Extensiones VS Code: LaTeX Workshop (compilación, vista previa) y, si se usa `Makefile`, la extensión Makefile Tools para ejecutar `make main`.

### Python (para los experimentos)
- Recomendado usar un venv: `python3 -m venv .venv && source .venv/bin/activate`.
- Instalar dependencias mínimas: `pip install scikit-learn numpy pandas nltk pyconll` (ajusta según necesidades del notebook/script).
