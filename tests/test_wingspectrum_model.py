import unittest

import numpy as np

from wingspectrum.model import BASIS_CENTERS, loss_and_gradient, optimize, optics


class WingSpectrumModelTests(unittest.TestCase):
    def test_optics_conserves_energy(self):
        spectrum = optics(np.zeros(len(BASIS_CENTERS)))
        np.testing.assert_allclose(
            spectrum.reflectance + spectrum.transmittance + 0.06,
            np.ones_like(spectrum.reflectance),
        )

    def test_analytic_gradient_matches_directional_difference(self):
        design = np.linspace(-2.0, 1.0, len(BASIS_CENTERS))
        direction = np.linspace(0.4, -0.3, len(BASIS_CENTERS))
        _, gradient, _ = loss_and_gradient(design)
        epsilon = 1e-6
        plus = loss_and_gradient(design + epsilon * direction)[0]
        minus = loss_and_gradient(design - epsilon * direction)[0]
        finite_difference = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(float(gradient @ direction), finite_difference, places=6)

    def test_minimal_optimization_improves_all_required_metrics(self):
        result = optimize()
        initial = result["history"][0]
        final = result["history"][-1]
        self.assertLess(final["loss"], initial["loss"])
        self.assertGreater(final["uvs_visibility"], initial["uvs_visibility"])
        self.assertGreater(final["vs_visibility"], initial["vs_visibility"])
        self.assertLess(final["human_reflectance"], 0.20)
        self.assertLess(final["solar_transmittance"], initial["solar_transmittance"])


if __name__ == "__main__":
    unittest.main()

