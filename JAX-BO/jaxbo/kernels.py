
import jax.numpy as np
from jax import jit

@jit
def RBF(x1, x2, params):
    output_scale = params[0]
    lengthscales = params[1:]
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
#     return output_scale * np.exp(-0.5 * r2)
    mat = output_scale * np.exp(-0.5 * r2)
    return mat

# @jit
# def RBF(x1, x2, params):
#     output_scale = params[0]
#     lengthscales = params[1:]
#     diffs = np.expand_dims(x1[:,0:13] / lengthscales[0:13], 1) - \
#             np.expand_dims(x2[:,0:13] / lengthscales[0:13], 0)
#     r2 = np.sum(diffs**2, axis=2)
#     return output_scale * np.exp(-0.5 * r2)

@jit
def Matern52(x1, x2, params):
    output_scale = params[0]
    lengthscales = params[1:]
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * (1.0 + np.sqrt(5.0*r2 + 1e-12) + 5.0*r2/3.0) * np.exp(-np.sqrt(5.0*r2 + 1e-12))

@jit
def Matern32(x1, x2, params):
    output_scale = params[0]
    lengthscales = params[1:]
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * (1.0 + np.sqrt(3.0*r2 + 1e-12)) * np.exp(-np.sqrt(3.0*r2 + 1e-12))
#     mat = output_scale * (1.0 + np.sqrt(3.0*r2)) * np.exp(-np.sqrt(3.0*r2))
#     return mat

@jit
def Matern12(x1, x2, params):
    output_scale = params[0]
    lengthscales = params[1:]
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum((diffs)**2, axis=2)
    return output_scale * np.exp(-np.sqrt(r2 + 1e-12))
#     diffs = np.expand_dims(x1 / lengthscales, 1) - \
#             np.expand_dims(x2 / lengthscales, 0)
#     return np.sqrt(r2+1e-12)[0,1]
#     pol = np.expand_dims(x1 / lengthscales, 1) * \
#             np.expand_dims(x2 / lengthscales, 0)
#     return output_scale * (1.0 + np.sqrt(5.0*r2) + 5.0*r2/3.0) * np.exp(-np.sqrt(5.0*r2)) *(np.sum(pol**3, axis=2))

@jit
def RatQuad(x1, x2, params):
    alpha = 1. # 1.0 ##########
    output_scale = params[0]
    lengthscales = params[1:]
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.power(1.0 + (0.5/alpha) * r2, -alpha)
