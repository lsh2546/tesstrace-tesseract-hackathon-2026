import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

MATRIX = np.array([[1.4, 0.2], [0.0, 0.6]], dtype=np.float64)

class InputSchema(BaseModel):
    x: Differentiable[Array[(2,), Float64]]

class OutputSchema(BaseModel):
    y: Differentiable[Array[(2,), Float64]]

def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(y=MATRIX @ np.asarray(inputs.x))

def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    del inputs, vjp_inputs, vjp_outputs
    return {"x": MATRIX.T @ np.asarray(cotangent_vector["y"])}

