# TessTrace

DAG-aware gradient fault localization for composed Tesseracts.

> **Validation status:** NumPy contract-fixture algorithm validation completed;
> GitHub Actions container validation completed with Tesseract Core 1.11.0.
> Evidence: [workflow run 32573968781](https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026/actions/runs/32573968781).
> The containerized WingSpectrum faulty-versus-fixed comparison is completed in
> [workflow run 32576243257](https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026/actions/runs/32576243257).

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

The repository also contains the first reduced-order WingSpectrum science model.
It maps ten smooth coating-band controls to an energy-conserving reflectance and
transmittance spectrum, evaluates UVS, VS, human-visible, and solar branches, and
optimizes a robust worst-species objective. This is explicitly a spectral surrogate,
not a fabricated multilayer or measured collision-rate claim. Its raw loss curve,
branch metrics, designs, and spectra are emitted as JSON in CI.

## Faulty-versus-fixed scientific comparison

The comparison holds the forward model, initial design, Adam settings, 180
iterations, seed, and predeclared gradient-contract tolerance fixed. The only
difference is that the faulty UVS image reverses the wavelength-axis cotangent in
its VJP. Both variants use real Tesseract Core 1.11.0 `apply` and
`vector_jacobian_product` endpoints.

At iteration 180, correcting that VJP improved model-based UVS visibility from
`0.5870` to `0.6169`, VS visibility from `0.2725` to `0.2930`, and the common
forward-evaluated objective from `-0.5750` to `-0.5876`. Human-visible
reflectance remained within the fixed `0.20` bound (`0.1388` faulty; `0.1494`
fixed). TessTrace labels only UVS `FAIL`, preserves VS/Human/Thermal as `PASS`,
and labels Optics `PARTIALLY_CONTAMINATED`; the corrected graph is entirely
`PASS`. The Actions artifact includes settings, full histories, spectra,
directional checks, endpoint evidence, image SHAs, and the workflow URL.

## Validation

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v

# Minimal four-branch science optimization and raw JSON evidence.
python scripts/run_wingspectrum_minimal.py --output reports/wingspectrum-minimal.json

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
