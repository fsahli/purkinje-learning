from scipy.optimize import minimize

# def minimize_lbfgs(objective, x0, verbose = False, maxfun = 15000, bnds = None):
#     if verbose:
#         def callback_fn(params):
#             print("Loss: {}".format(objective(params)[0]))
#     else:
#         callback_fn = None
        
#     result = minimize(objective, x0, jac=True,
#                       method='L-BFGS-B', bounds = bnds,
#                       callback=callback_fn, options = {'maxfun':maxfun})
#     return result.x, result.fun

def minimize_lbfgs(objective, x0, verbose = False, maxfun = 15000, bnds = None):
    if verbose:
        def callback_fn(params):
            print("Loss: {}".format(objective(params)[0]))
    else:
        callback_fn = None
        
    result = minimize(objective, x0, jac="2-point",
                      method='L-BFGS-B', bounds = bnds,
                      callback=callback_fn, options = {'maxfun':maxfun})
    print (f"optimization success: {result.success}")
    print (result.message)
    print (f"nit: {result.nit}")
    return result.x, result.fun

def minimize_lbfgs_grad(objective, x0, verbose = False, maxfun = 15000, bnds = None):
    if verbose:
        def callback_fn(params):
            print("Loss: {}".format(objective(params)[0]))
    else:
        callback_fn = None
        
    result = minimize(objective, x0, jac=True,
                      method='L-BFGS-B', bounds = bnds,
                      callback=callback_fn, options = {'maxfun':maxfun, 'gtol':1e-8})
#     result = minimize(objective, x0,
#                       method='Nelder-Mead', bounds = bnds,
#                       callback=callback_fn, options = {'maxfun':maxfun, 'gtol':1e-8})
#     print (f"optimization success: {result.success}")
#     print (result.message)
#     print (f"nit: {result.nit}")
#     print (result.info)
    return result.x, result.fun
