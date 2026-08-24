# Submission copy

## Title

TessTrace — DAG-aware gradient fault localization for composed Tesseracts

## Short summary

TessTrace finds the first faulty VJP boundary in a composed scientific-computing DAG, traces its upstream cotangent contamination, and preserves healthy sibling branches. In the WingSpectrum demonstration, six real Tesseract 1.11.0 containers form a differentiable spectral-glazing pipeline. A silent UVS backward bug degrades the optimized design; TessTrace localizes it, and the corrected run improves model-based UVS visibility by 5.1%, VS visibility by 7.5%, and the common forward-evaluated loss.

## Technical description

Silent gradient defects are unusually dangerous in composed optimization pipelines: forward execution succeeds, tensor shapes agree, and the optimizer can converge despite an incorrect VJP. TessTrace introduces DAG-aware gradient contracts. At each relevant boundary it compares the supplied VJP with a central directional finite difference, records the raw analytic and reference contractions, and assigns one of five frozen states: PASS, FAIL, CONTAMINATED, PARTIALLY_CONTAMINATED, or UNTESTED. A failed cotangent is traced upstream; at fan-out joins, mixed healthy and failed contributions become partially contaminated without falsely labeling healthy sibling branches.

WingSpectrum demonstrates why this needs Tesseract. Separate Optics, UVS, VS, Human, and Solar/Thermal components cross real container boundaries and expose `apply` and `vector_jacobian_product` endpoints. The controlled faulty/fixed comparison holds the forward calculation, initial variables, Adam optimizer, learning rate, 180 iterations, seed, and numerical tolerances fixed; only the UVS wavelength-axis cotangent order differs. GitHub Actions builds six Tesseract Core 1.11.0 images, confirms faulty/fixed exit codes 1/0, validates the exact DAG states, and preserves full histories, spectra, directional errors, endpoint evidence, image SHAs, and settings as artifacts.

The result is an engineering tool and a concrete scientific demonstration: TessTrace pinpoints the silent UVS defect, keeps VS, Human, and Thermal marked PASS, and shows that correcting the gradient recovers a better design. WingSpectrum is a reduced-order visibility surrogate; it does not claim measured bird-collision reduction, fabrication readiness, or improved solar performance.

## Links

- Repository: https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026
- Live evidence UI: https://lsh2546.github.io/tesstrace-tesseract-hackathon-2026/
- Verified container experiment: https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026/actions/runs/32576243257
- SHA-pinned final audit: https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026/actions/runs/32727604858
- Pages deployment: https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026/actions/runs/32727655759
- Technical note: https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026/blob/main/docs/tesstrace-technical-note.pdf

## LinkedIn post

Silent gradient bugs are dangerous: a scientific pipeline can run, an optimizer can converge, and the result can still be wrong.

For the Tesseract Hackathon 2026, I built **TessTrace**, a DAG-aware gradient fault-localization tool for composed Tesseracts. It does more than report a mismatch: it identifies the faulty VJP boundary, traces upstream cotangent contamination, and preserves healthy sibling branches.

The WingSpectrum demo uses six real Tesseract 1.11.0 containers. With the forward model and every optimization setting held fixed, TessTrace finds a silent UVS backward bug. After correction, model-based UVS visibility improves by 5.1%, VS visibility by 7.5%, and the common forward-evaluated loss improves—while the human-reflectance constraint remains satisfied.

The repository includes the live evidence UI, a four-page technical note, reproducible GitHub Actions workflows, immutable action pins, and raw experiment artifacts.

Live demo: https://lsh2546.github.io/tesstrace-tesseract-hackathon-2026/

Code: https://github.com/lsh2546/tesstrace-tesseract-hackathon-2026

#Tesseract #ScientificComputing #DifferentiableProgramming #Optimization #OpenSource

Tag: Pasteur Labs & ISI and Tesseract.
