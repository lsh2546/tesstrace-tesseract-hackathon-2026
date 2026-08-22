# TessTrace: DAG-aware gradient fault localization for composed Tesseracts

## 1. Problem and contributions

Differentiable scientific pipelines depend on vector-Jacobian products (VJPs) crossing solver, language, and container boundaries. A VJP can have the correct shape and finite values while being numerically wrong. The forward pipeline continues to run, the optimizer may appear to converge, and the final design is silently degraded. Existing per-component checks can report a mismatch, but a composed directed acyclic graph (DAG) needs fault localization: which boundary failed, which upstream cotangent paths are affected, and which sibling branches remain trustworthy?

**TessTrace does not merely report a gradient mismatch; it localizes the faulty VJP boundary while preserving healthy sibling branches in a composed Tesseract DAG.**

Contributions: (1) a local gradient contract comparing each VJP with a central directional finite difference; (2) DAG-aware state propagation separating direct failure, upstream contamination, mixed fan-out contamination, healthy siblings, and untestable boundaries; (3) CI-compatible JSON evidence and deterministic exit codes; (4) a controlled WingSpectrum demonstration showing that correcting one silent UVS VJP changes the optimized design; and (5) reproducible validation with Tesseract Core 1.11.0, six real images, real endpoints, raw artifacts, and image SHAs.

## 2. Gradient contracts and DAG state propagation

For `y = f(x)`, input direction `d`, and output cotangent `c`, TessTrace compares `a = <VJP_f(x,c),d>` with `r = <c,[f(x+epsilon*d)-f(x-epsilon*d)]/(2*epsilon)>`. The experiment freezes `epsilon=1e-6`, `rtol=1e-4`, and `atol=1e-7`. It records analytic and reference scalars plus absolute and relative error.

States are `PASS` (local contract passed), `FAIL` (directly confirmed defect), `CONTAMINATED` (failed cotangent propagated upstream), `PARTIALLY_CONTAMINATED` (healthy and failed cotangents merge), and `UNTESTED` (no numerical verdict). Propagation follows reverse-mode direction. If UVS fails, its cotangent affects Optics; VS, Human, and Thermal remain healthy siblings. Optics becomes `PARTIALLY_CONTAMINATED`, not globally failed.

## 3. WingSpectrum experiment and fairness

WingSpectrum is a reduced-order spectral glazing inverse-design demonstration. Ten smooth controls generate reflectance and transmittance over 300-1100 nm with fixed absorptance and `R+T+A=1`. Optics fans out to UVS, VS, human-visible, and solar branches. A robust objective combines worst-species and aggregate visibility, human-visible constraint, color variation, solar-control trade-off, and regularization.

Faulty and fixed runs share the identical forward calculation, initial vector (ten values of -3), Adam, learning rate 0.16, 180 iterations, seed 20260822, objective, and tolerance. The only difference is the UVS VJP: the faulty image reverses the 81-sample wavelength-axis cotangent. UVS forward values remain identical.

Six Tesseract Core 1.11.0 images are built: Optics, faulty UVS, fixed UVS, VS, Human, and Thermal. The orchestrator calls real `apply` and `vector_jacobian_product` endpoints and stores settings, histories, spectra, checks, DAG states, endpoints, exit codes, image SHAs, and workflow URL.

## 4. Results, reproducibility, and limits

TessTrace identifies UVS as `FAIL` with relative directional error 0.7103. VS, Human, and Thermal remain `PASS`; Optics is `PARTIALLY_CONTAMINATED`. The corrected UVS error is 4.25e-12 and the whole DAG passes. Faulty/fixed exit codes are 1/0.

Correction improves model-based UVS visibility from 0.5870 to 0.6169 (+5.1%) and VS from 0.2725 to 0.2930 (+7.5%). Common forward loss improves from -0.5750 to -0.5876. Human-visible reflectance stays below 0.20 in both runs (0.1388/0.1494). Solar transmittance changes from 0.5355 to 0.5430 and is a trade-off, not an improvement.

Reproduce locally with `tesstrace --json reports/faulty.json` (exit 1) and `tesstrace --fixed --json reports/fixed.json` (exit 0). Full evidence is in Actions run 32576243257, artifact `wingspectrum-container-comparison-32576243257`.

Limits: no claim of measured collision reduction, product-ready coating, fabrication feasibility, or improved solar/thermal performance. Outputs are model-based spectral proxies. The optics model is a differentiable surrogate. Numerical checks may return `UNTESTED` near non-smooth or unstable points.
