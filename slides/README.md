# MSE342 Presentation

This directory contains the academic Beamer version of the project presentation.

- `main.tex`: source slides.
- `main.pdf`: compiled slide deck.
- `speaker_script.md`: slide-by-slide talk script.

Build from the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=slides slides/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=slides slides/main.tex
```

