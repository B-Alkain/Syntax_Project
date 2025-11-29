MAIN_TEX=tex_files/main.tex
OUTDIR=.

.PHONY: main summary

# Generate main.pdf in repository root
main:
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR) -jobname=main $(MAIN_TEX)

# Optional: generate summary.pdf in repository root
summary:
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR) -jobname=summary $(MAIN_TEX)
