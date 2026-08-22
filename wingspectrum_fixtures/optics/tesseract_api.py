import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

W = np.arange(300.0, 1101.0, 10.0)
C = np.array([330, 370, 410, 460, 520, 590, 680, 780, 900, 1020])
S = np.array([28, 30, 34, 42, 50, 60, 70, 85, 95, 105])
B = np.stack([np.exp(-0.5 * ((W-c)/s)**2) for c, s in zip(C, S)], axis=1)
B /= np.maximum(B.sum(axis=1, keepdims=True), 1.0)

class InputSchema(BaseModel):
    x: Differentiable[Array[(10,), Float64]]
class OutputSchema(BaseModel):
    y: Differentiable[Array[(162,), Float64]]

def _values(x):
    a = 1.0 / (1.0 + np.exp(-(B @ np.asarray(x))))
    r = 0.04 + 0.70 * a
    return a, np.concatenate([r, 0.94-r])

def apply(inputs):
    return OutputSchema(y=_values(inputs.x)[1])

def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    del vjp_inputs, vjp_outputs
    a, _ = _values(inputs.x)
    c = np.asarray(cotangent_vector["y"])
    dr = c[:81] - c[81:]
    return {"x": B.T @ (0.70 * a * (1.0-a) * dr)}
