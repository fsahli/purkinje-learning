import numpy as onp
import jax.numpy as np
from jax import random, vmap

from jaxbo.input_priors import uniform_prior, gaussian_prior
from jaxbo.models import GP
from jaxbo.utils import normalize, compute_w_gmm

from pyDOE import lhs
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

onp.random.seed(1234)

class BO_ecg():
    # Class to find the parameters of the Purkinje Tree based on an ecg
    def __init__(self, bo_purkinje_tree):
        # bo_purkinje_tree: the initial tree
        self.bo_purkinje_tree = bo_purkinje_tree



    def extract_overlapping_section(self, ground_truth, predicted, delay):
        # Function to extract the overlapping section of two arrays, given a delay.
        # The delay is consistent with the index given by the onp.correlate() function, 
        # with the option mode = "full", for example 
        # delay = 0 the initial point of the ground_truth coincides with the last point of predicted
        # delay = len(predicted)-1 the initial point of the ground_truth coincides with the initial point of predicted
        # delay = len(ground_truth)-1 the last point of the ground_truth coincides with the last point of predicted
        len1, len2 = ground_truth.shape[0], predicted.shape[0]
#         trim = min(len1, len2)

        aux = len2-delay-1
        if len2 <= len1:
            if delay < len1:
                gt = ground_truth[max(0,-1*aux):delay+1]
                pred = predicted[max(0,aux):]
            if delay >= len1:
                gt = ground_truth[-1*aux:]
                pred = predicted[0:len1+aux]

        elif len2 > len1:
            if delay < len1:
                gt = ground_truth[0:delay+1]
                pred = predicted[aux:]
            elif delay >= len1:
                in1 = max(0,-1*aux)
                in2 = max(0,aux)
                gt = ground_truth[in1:]
                pred = predicted[in2:in2+(len1-in1)]

        assert gt.shape[0] == pred.shape[0], "the function should extract the overlapping sections"
        
        return gt, pred
        
        
        
    def plot_ecg_match(self, predicted, filename_match = None):
        # Find shift and plot best match
        _, ind_min_loss = self.calculate_loss(predicted, cross_correlation = True, return_ind = True)

        len1 = self.ground_truth.shape[0]
        len2 = predicted.shape[0]
        
#         t_min_loss = cut_in + 2 * ind_min_loss
        t_min_loss = len1 - 1 + ind_min_loss

        fig, axs = plt.subplots(3, 4, figsize = (10,13), dpi = 120, sharex = True, sharey = True)
        for ax,l in zip(axs.ravel(), self.ground_truth.dtype.names):
            # # fixed ground truth and moving predicted
            # ax.plot(ground_truth[l],'b', alpha=0.6, label="Ground truth")
            # t_pred = t_min_loss - len2 + 1 + onp.arange(len2)
            # ax.plot(t_pred, predicted[l],'r', alpha=0.6, label="BO")
            # ax.axvspan(max(0, t_pred[0]), min(len1, t_pred[-1]), alpha=0.2, color='wheat')                

            # fixed predicted and moving ground truth
            t_gt = t_min_loss - len1 + 1 + onp.arange(len1)
            ax.plot(t_gt, self.ground_truth[l], 'tab:blue', alpha = 0.6, label = "Ground truth")
            ax.plot(predicted[l], 'tab:red', alpha = 0.6, label = "BO")
            ax.axvspan(t_gt[0], t_gt[-1], alpha = 0.2, color = 'wheat')

            ax.grid(linestyle = '--', alpha = 0.4)
            ax.set_title(l)
            if l == "V2":
                ax.legend(fontsize = "8")

        fig.tight_layout()
        if filename_match is not None:
            fig.savefig(filename_match + "_ecg_match.pdf")
    
    
    
    def calculate_loss(self, predicted, cross_correlation = True, return_ind = False, ecg_pat = None):
        if ecg_pat is None:
            ground_truth = self.ground_truth
        else:
            ground_truth = ecg_pat 

        if cross_correlation:
            # In this case, we will fix the predicted ecg and move the ground truth (with only 
            # the QRS complex) to find the optimal shift
            len1 = ground_truth.shape[0]
            len2 = predicted.shape[0]

            # cut_in     = 200
            cut_fin    = 0 # 200 # to remove T wave from predicted ecg
            loss_shift = []

            for t_shift in np.arange(len1 - 1, len2 - cut_fin):
                errors = []
                for label in ground_truth.dtype.names:
                    pred, gt = self.extract_overlapping_section(predicted[label], ground_truth[label], t_shift)

                    mse = mean_squared_error(gt, pred)
                    errors.append(mse)
                
                loss = sum(errors)
                loss_shift.append(loss)

            min_loss     = min(loss_shift)            
            ind_min_loss = loss_shift.index(min_loss)

            if return_ind:
                return min_loss, ind_min_loss
            else:
                return min_loss

        else:
            errors = []
            for label in ground_truth.dtype.names:
                len1, len2 = ground_truth[label].shape[0], predicted[label].shape[0]
                trim       = min(len1, len2)
                mse        = mean_squared_error(ground_truth[label][:trim], predicted[label][:trim])
                errors.append(mse)

            loss = sum(errors)

        return loss



    def mse_jaxbo(self, ground_truth, variable_parameters):
        self.ground_truth = ground_truth
        
        # variable parameters = {"name": [lb, ub, "uniform" or "gaussian"]}
        self.variable_parameters = variable_parameters

        lb_params  = np.array([])
        ub_params  = np.array([])

        for var_name,var_value in variable_parameters.items():
            lb_params = np.append(lb_params, var_value[0])
            ub_params = np.append(ub_params, var_value[1])

        # we assume all the variable parameters have the same prior distribution
        dist_types = [value[2] for value in variable_parameters.values()]
        assert len(set(dist_types)) == 1, "The prior distribution for all variable parameters must be equal"
        if dist_types[0] == "uniform":
            p_x_params = uniform_prior(lb_params, ub_params)
        elif dist_types[0] == "gaussian":
            p_x_params = gaussian_prior(lb_params, ub_params)
        else:
            raise NotImplementedError

        # f returns the mse between the predicted and the real ecg
        def f(x):
            # parameters (to change)
            x = x.astype(float)
            print(x)

            var_params = self.set_dictionary_variables(var_parameters = variable_parameters,
                                                            x_values = x)

            # obtain predicted ecg
            try:
                predicted, propeiko, LVtree, RVtree = self.bo_purkinje_tree.run_ECG(n_sim = 0, modify = True, side = 'both', **var_params)
                loss                                = self.calculate_loss(predicted)
            except:
                print ("Error in run_ECG")
                loss = self.y_trees_non_valid
            
            print(f"Loss: {loss}")
            return loss

        self.dim        = len(lb_params)
        self.f          = f
        self.p_x_params = p_x_params
        self.lb_params  = lb_params
        self.ub_params  = ub_params

        # Domain bounds
        self.bounds = {'lb': lb_params, 'ub': ub_params}

        return f, p_x_params



    def set_initial_training_data(self, N, noise = 0.):
        self.noise = noise

        # Initial training data
        X = self.lb_params + (self.ub_params - self.lb_params) * lhs(self.dim, N)
        y = list(map(self.f, X)) # eval the function on X (N points)
        y = np.asarray(y)
        y = y + self.noise * y.std(0) * onp.random.normal(y.shape)

        return X, y



    def set_test_data(self):
        # Test data
        if self.dim == 1:
            nn           = 1000
            X_star       = np.linspace(self.lb_params[0], self.ub_params[0], nn)[:,None]

        elif self.dim == 2:
            nn = 10
            xx = np.linspace(self.lb_params[0], self.ub_params[0], nn)
            yy = np.linspace(self.lb_params[1], self.ub_params[1], nn)
            XX, YY = np.meshgrid(xx, yy)
            X_star = np.concatenate([XX.flatten()[:,None],
                                     YY.flatten()[:,None]], axis = 1)
            return X_star, XX, YY
        
        else:
            nn = 25000
            print (str(nn) + " test points")
            X_star = self.lb_params + (self.ub_params - self.lb_params) * lhs(self.dim, nn)

        return X_star



    def bo_loop(self, X, y, X_star, true_x, options, save_info = False):
        # Main Bayesian optimization loop
        # X, y: training data.
        gp_model = GP(options)
        rng_key  = random.PRNGKey(0)
        
        mean_iterations = []
        std_iterations  = []
        
        w_pred_iterations  = []
        a_pred_iterations  = []

        self.nIter = options['nIter']

        for it in range(options['nIter']):
            print('-------------------------------------------------------------------')
            print('------------------------- Iteration %d/%d -------------------------' % (it+1, options['nIter']))
            print('-------------------------------------------------------------------')

            # Fetch normalized training data
            norm_batch, norm_const = normalize(X, y, self.bounds)


            # Train GP model
            print('Train GP...')
            rng_key    = random.split(rng_key)[0]
            opt_params = gp_model.train(norm_batch,
                                        rng_key,
                                        num_restarts = 5) # 100


            # Fit GMM
            if options['criterion'] == 'LW-LCB' or options['criterion'] == 'LW-US':
                print('Fit GMM...')
                rng_key  = random.split(rng_key)[0]
                kwargs   = {'params'    : opt_params,
                            'batch'     : norm_batch,
                            'norm_const': norm_const,
                            'bounds'    : self.bounds,
                            'kappa'     : gp_model.options['kappa'],
                            'rng_key'   : rng_key}
                gmm_vars = gp_model.fit_gmm(**kwargs, N_samples = 10000)
            else:
                gmm_vars = None


            # Compute next point via minimizing the acquisition function            
            kwargs = {'params'    : opt_params,
                      'batch'     : norm_batch,
                      'norm_const': norm_const,
                      'bounds'    : self.bounds,
                      'kappa'     : gp_model.options['kappa'],
                      'gmm_vars'  : gmm_vars,
                      'rng_key'   : rng_key}


            if save_info:
                print ("Compute and save predictions (mean, std, w, acq_fun) ...")
                # Compute predicted mean and std
                mean_it, std_it = gp_model.predict(X_star, **kwargs) # of normalized data

                # Obtain y_it and sigma_it
                y_it     = mean_it * norm_const["sigma_y"] + norm_const["mu_y"]
                sigma_it = std_it * norm_const["sigma_y"]
                
                mean_iterations.append(y_it)
                std_iterations.append(sigma_it)
                                                            
                # Compute predictions (for plotting)
                if options['criterion'] == 'LW-LCB' or options['criterion'] == 'LW-US':
                    w_pred_it = compute_w_gmm(X_star, **kwargs)
                else:
                    w_pred_it = np.zeros(X_star.shape[0])

                acq_fun   = lambda x: gp_model.acquisition(x, **kwargs)
                a_pred_it = vmap(acq_fun)(X_star)

                w_pred_iterations.append(w_pred_it)
                a_pred_iterations.append(a_pred_it)                                


            # Training error
            train_error = np.mean((gp_model.predict(X , **kwargs)[0] - norm_batch['y'])**2)
            print (f"Train error: {train_error}")
            
            print('Computing next acquisition point...')
            new_X,_,_ = gp_model.compute_next_point_lbfgs(num_restarts = 50, **kwargs) # 100


            # Acquire data
            new_y = list(map(self.f, new_X))
            print("new_y", len(new_y))
            #new_ecg_array = np.asarray(ecg, dtype=object)
            new_y = np.asarray(new_y)
            new_y = new_y + self.noise*new_y.std(0)*onp.random.normal(new_y.shape)
            # Augment training data
            print('Updating data-set...')
            X = np.concatenate([X, new_X], axis = 0)
            y = np.concatenate([y, new_y], axis = 0)            


            # Print current best
            idx_best = np.argmin(y)
            best_x = X[idx_best,:]
            best_y = y.min()
            print('True location: ({}), True value: {}'.format(true_x, ""))
            print('Best location: ({}), Best value: {}'.format(best_x, best_y))
            print('New  location: ({}), New  value: {}'.format(new_X, new_y))                        

        info_iterations = [mean_iterations, std_iterations, w_pred_iterations, a_pred_iterations]
        return X, y, info_iterations



    def plot_mse(self, X, y, N, file_name):
        idx_best = np.argmin(y)
        best_x   = onp.array(X[idx_best,:])

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.nIter) + 1, y[N:])
        ax.axhline(y = np.min(y), color = 'r', linestyle = '--', alpha = 0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("mse")
        # plt.grid(True,which="both")
        ax.set_title("MSE")
        props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)

        # place a text box in upper left in axes coords
        params     = "params = " + str([value for value in self.variable_parameters.keys()])
        best_x_app = [f"{num:.2f}" for num in best_x]
        best_y_it  = " (in iteration "+str(idx_best-N+1)+")" if idx_best>=N else " (in training points)"
        textstr    = params + "\nbest_x = " + str(best_x_app) + "\ny_min = " + "{:e}".format(onp.min(y)) + best_y_it

        ax.text(0.05, 0.95, textstr, transform = ax.transAxes, fontsize = 5.5, verticalalignment = 'top', bbox = props)

        fig.tight_layout()
        fig.savefig(file_name+"_MSE.pdf")



    def percent_error(self, real, obtained):
        return abs(real - obtained) / real * 100.



    def mse_error(self, real, obtained):
        err = onp.zeros(len(obtained))
        for ind in np.arange(len(obtained)):
            err[ind] = 1./len(real) * sum((np.array(real) - obtained[ind,:])**2)
        return err



    def update_purkinje_tree(self, X, y, var_parameters):
        # Update tree with optimal parameters
        idx_best = onp.argmin(y)
        best_x   = onp.array(X[idx_best,:])

        best_var_parameters = self.set_dictionary_variables(var_parameters = var_parameters,
                                                            x_values       = best_x)

        ecg_bo, endo_bo, LVtree_bo, RVtree_bo = self.bo_purkinje_tree.run_ECG(modify = True, side = 'both', **best_var_parameters)
        return ecg_bo, endo_bo, LVtree_bo, RVtree_bo



    def set_dictionary_variables(self, var_parameters, x_values):
        dict_parameters = {}
        ind             = 0
        for var_name, _ in var_parameters.items():
            if var_name == "fascicles_length" or var_name == "fascicles_angles":
                dict_parameters[var_name] = [[x_values[ind], x_values[ind+1]],
                                             [x_values[ind+2], x_values[ind+3]]]
                ind += 4
            elif var_name == "w" or var_name == "branch_angle" or var_name == "length":
                dict_parameters[var_name] = [x_values[ind], x_values[ind]]
                ind += 1
            elif var_name == "root_time" or var_name == "cv":
                dict_parameters[var_name] = x_values[ind]
                ind += 1
            else:
                dict_parameters[var_name] = [x_values[ind], x_values[ind+1]]
                ind += 2
        
        return dict_parameters
