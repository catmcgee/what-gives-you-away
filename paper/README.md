# Paper and arXiv source

Compile the paper from this directory:

```bash
tectonic main.tex
```

For arXiv, upload these four files and select `main.tex` as the top-level file:

- `main.tex`
- `refs.bib`
- `fig_conversation_sensitivity.pdf`
- `fig_cross_model.pdf`

The figure PDFs are copies of the vector figures in `results/`. The test suite
checks that the manuscript uses only these flat, local paths.
