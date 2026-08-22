import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64
W=np.arange(300.0,1101.0,10.0)
def g(c,s):
    v=np.exp(-0.5*((W-c)/s)**2); return v/v.sum()
Q=0.25*g(445,38)+0.55*g(555,48)+0.20*g(610,52)
class InputSchema(BaseModel): x: Differentiable[Array[(162,), Float64]]
class OutputSchema(BaseModel): y: Differentiable[Array[(2,), Float64]]
def _metrics(x):
    d=np.asarray(x)[:81]-0.04
    m=Q@d
    return d,m,np.array([m,np.sum(Q*(d-m)**2)])
def apply(inputs): return OutputSchema(y=_metrics(inputs.x)[2])
def vector_jacobian_product(inputs,vjp_inputs,vjp_outputs,cotangent_vector):
    del vjp_inputs,vjp_outputs
    d,m,_=_metrics(inputs.x); c=np.asarray(cotangent_vector["y"])
    grad=c[0]*Q+c[1]*2.0*Q*(d-m)
    return {"x":np.concatenate([grad,np.zeros(81)])}
