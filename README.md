# TessTrace

DAG-aware gradient fault localization for composed Tesseracts.

> **Validation status:** NumPy contract-fixture algorithm validation completed;
> GitHub Actions container validation completed with Tesseract Core 1.11.0.
> Evidence: [workflow run 32573968781](https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026/actions/runs/32573968781).

This repository currently contains the first technical validation only: a branched
WingSpectrum-shaped DAG with one intentionally incorrect UVS-vision VJP. TessTrace
checks local VJP contracts with central directional finite differences, preserves
healthy sibling branches, and marks the shared upstream fan-out as partially
contaminated.

The GitHub Actions workflow contains a separate container gate. It builds
Optics, faulty/fixed UVS, VS, Human, and Thermal as real Tesseract images using
Tesseract Core 1.11.0, invokes their `apply` and `vector_jacobian_product`
endpoints, and preserves raw reports as an Actions artifact. The first completed
container gate is recorded in workflow run `32573968781`.

## Validation

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v

# Intentionally faulty graph: prints JSON and exits nonzero.
python -m tesstrace.cli --json reports/faulty.json

# Corrected graph: all contracts pass and the process exits zero.
python -m tesstrace.cli --fixed --json reports/fixed.json
```

Status semantics are frozen as follows:

- `PASS`: the local gradient contract passed.
- `FAIL`: a local node or edge VJP error was directly confirmed.
- `CONTAMINATED`: an upstream path received a failed cotangent.
- `PARTIALLY_CONTAMINATED`: healthy and failed branch cotangents meet upstream.
- `UNTESTED`: non-smoothness, numerical instability, or execution failure prevented a verdict.
