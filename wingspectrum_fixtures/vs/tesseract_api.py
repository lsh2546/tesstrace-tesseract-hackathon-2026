import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64
W=np.arange(300.0,1101.0,10.0)
def g(c,s):
    v=np.exp(-0.5*((W-c)/s)**2); return v/v.sum()
Q=0.22*g(405,32)+0.78*g(505,55)
class InputSchema(BaseModel): x: Differentiable[Array[(162,), Float64]]
class OutputSchema(BaseModel): y: Differentiable[Array[(1,), Float64]]
def apply(inputs): return OutputSchema(y=np.array([Q @ (np.asarray(inputs.x)[:81]-0.04)]))
def vector_jacobian_product(inputs,vjp_inputs,vjp_outputs,cotangent_vector):
    del inputs,vjp_inputs,vjp_outputs
    return {"x":float(np.asarray(cotangent_vector["y"])[0])*np.concatenate([Q,np.zeros(81)])}
