# Syntax Project

PoS tagger for Basque and Catalan using Universal Dependencies data.

## Structure
- `main.py`: evaluation utilities (accuracy, per-tag accuracy, precision, recall and F1 reports) and `get_data`.
- `model/hmm.py`: implementation of custom HMM.
- `POS.ipynb`: notebook with experiments NLTK HMM vs custom HMM n-gram taggers, plots and qualitative examples.
- `utils/conll_to_csv.py`: conversion from `.conllu` to CSV using `pyconll`.
- `datasets/ud_basque`, `datasets/ud_catalan`: train/dev/test splits in CSV.
- `tex_files/`:LaTeX report (`main.tex`, chapters in `chapters/`, figures in `figuras/`).

## Notebook `POS.ipynb`
- Loads UD CSV files (train/dev/test) with `load_ud_csv` and validates word/tag lengths.
- Trains the NLTK HMM and the custom HMM, compares accuracies and n-gram taggers (default/unigram/bigram/trigram).
- Generates plots (saved in `tex_files/figuras/`) and includes examples of Viterbi, joint probability and random samples.

## Editing and compiling the LaTeX report
1) Edit content: modify chaptes in `tex_files/chapters/` (introduction, methodology, results, conclusions). The file `tex_files/main.tex` just handles the preamble and the `\input{...}`.
2) Generate the PDF in the repo root:
   - With Makefile: `make main` creates `main.pdf`; `make summary` creates `summary.pdf`, both in the root.
   - Without Makefile:  `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=. -jobname=main tex_files/main.tex` (change `-jobname` to `summary` if preferred).

### LaTeX dependencies (Windows/Linux)
- **Windows**: install MiKTeX (https://miktex.org/download) and add `pdflatex` to PATH. In VS Code, install the LaTeX Workshop extension and compile using the command above from WSL/PowerShell.
- **Linux**: install basic TeX Live (`sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended latexmk`). Compile using the command above or `make`.
- VS Code Extensions: LaTeX Workshop (for compilation and preview) and, if using a Makefile, Makefile Tools to run `make main`.

### Python (for the experiments)
- Recommended to use a venv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install minimal dependencies: `pip install scikit-learn numpy pandas nltk pyconll` (adjust depending on the notebook/script needs).
