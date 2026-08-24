# TessTrace demo — 90-second script and shot plan

## 0:00–0:12 — The silent failure

**Screen:** Open the live UI on the hero and DAG. Slowly highlight `UVS Vision — FAIL`.

**Voiceover:** “A wrong vector–Jacobian product does not have to crash a scientific pipeline. It can pass every shape check, let the optimizer converge, and silently produce the wrong design.”

## 0:12–0:30 — Localization, not just detection

**Screen:** Follow the red UVS edge upstream to Optics. Keep VS, Human, and Solar/Thermal visible in green.

**Voiceover:** “TessTrace checks local gradient contracts in a composed Tesseract DAG. It identifies the faulty UVS VJP, marks the shared Optics cotangent as partially contaminated, and—critically—preserves the healthy sibling branches as passing.”

## 0:30–0:48 — Real Tesseracts

**Screen:** Show the verified Actions run, the six-image build steps, `Tesseract 1.11.0`, and the `apply` plus `vector_jacobian_product` evidence.

**Voiceover:** “This is not a NumPy-only mock. GitHub Actions builds six real Tesseract 1.11.0 images and calls their apply and vector-Jacobian-product endpoints. The faulty fixture exits one; the corrected fixture exits zero; raw directional checks are preserved as artifacts.”

## 0:48–1:12 — Scientific impact

**Screen:** Return to the UI. Switch Outcome → Spectrum → Loss curve. Show the fixed/faulty figures.

**Voiceover:** “In WingSpectrum, the forward model, initial design, seed, Adam optimizer, learning rate, iterations, and tolerances are identical. Only the UVS backward rule changes. Fixing it improves model-based UVS visibility by 5.1 percent, VS visibility by 7.5 percent, and the common forward-evaluated loss, while human reflectance remains below its frozen bound.”

## 1:12–1:27 — Why it matters

**Screen:** Return to the DAG, then end on the repository and reproducibility command.

**Voiceover:** “TessTrace does not merely report a gradient mismatch. It localizes the faulty VJP boundary while preserving healthy sibling branches—turning a silent optimization failure into a reproducible diagnosis and recovery.”

## Recording notes

- Record at 1920×1080, 30 fps; keep the browser zoom at 100%.
- Do not claim measured collision reduction, product-ready glazing, or improved solar performance.
- Keep the verified run URL and Tesseract version readable for at least three seconds.
- Use the frozen UI and evidence; do not recompute or round values differently.
