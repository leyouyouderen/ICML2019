"""Gaussian processes regression. """

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import pdist,cdist, squareform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated
import time;
import logging;

class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:

       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Target values in training data (also required for prediction)

    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    """
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None,previous_K=None):
        
        self.logger=logging.getLogger(__name__);
        self.logger.setLevel(logging.WARNING);
        handler = logging.FileHandler('d:/projects/compressS/GPOnlineLog.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not len(self.logger.handlers):
            self.logger.addHandler(handler);
        self.logger.info('Initiallzation of the model start...');
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.previous_K=previous_K
        
    @property
    @deprecated("Attribute rng was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def rng(self):
        return self._rng

    @property
    @deprecated("Attribute y_train_mean was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def y_train_mean(self):
        return self._y_train_mean

    def fit(self, X, y ,previous_K=None,newData=1,previous_mats=None):
        #the previous_K stores the gram mat from last step while previous_mat=[A,B] where A is the secondNorm of the x_i-x_j and B is x_i dot x_j
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        newData controls how many new datapoints has been added from last training(Used for batchMode)
        Returns
        -------
        self : returns an instance of self.
        """
        self.newData=1;
        t1=time.time();
        if self.kernel is None:  # Use an RBF kernel as default
            self.logger.info('Training with default rbf kernel');
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.logger.info('Training with customized kernel');
            #if previous_K is None:
            self.kernel_ = clone(self.kernel)
                
        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.logger.info('shape of the x is m=%d , n=%d',X.shape[0],X.shape[1]);
        # Normalize target value
        
        self.logger.info('start to normalize y value...');
        t3=time.time()
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        t4=time.time();
        self.logger.info("finish normalizing Y in----------- %s seconds", str(t4-t3))
        
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            self.logger.info('hyper parameter of the kernel will be optimized')
            self.logger.info('optimizing the hyper parameter of the kernel')
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True,pre_K=previous_K)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
            t2=time.time();
            self.logger.info("finish opt hyper-para of kernel in----------- %s seconds", str(t2-t4))
        else:
            self.logger.info('hyper parameter of the kernel will be fixed')
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        #This is the most time consuming part?
        t6=time.time()
        if(previous_K is not None):
            self.logger.info('use previous K_n and x_new to compute K_n+1');
            col=self.kernel_(X[0:X.shape[0]-newData,:],X[X.shape[0]-newData:X.shape[0],:]);
            K= np.concatenate((previous_K,col),axis=1);
            row=np.concatenate((col.T,self.kernel_(X[X.shape[0]-newData:X.shape[0],:],X[X.shape[0]-newData:X.shape[0],:])),axis=1)     
            K=np.concatenate((K,row),axis=0)
            self.K_return=K;
        elif(previous_mats is not None):
            #update A
            A=previous_mats[0]
            B=previous_mats[1]
            trainInd=range(int(X.shape[0])-newData)
            sampleInd=range(int(X.shape[0])-newData,int(X.shape[0]))
            

            Acol=cdist(X[trainInd,:],X[sampleInd,:], metric='sqeuclidean')
            Aone=cdist(X[sampleInd,:],X[sampleInd,:],metric='sqeuclidean')
            Arow=np.concatenate((Acol.transpose(),Aone),axis=1)
            #print('aaaa'+str(A.shape))
            #print('acol'+str(Acol.shape))
            A=np.concatenate((A,Acol),axis=1)
            A=np.concatenate((A,Arow),axis=0)
            #update B
            Bcol= np.inner(X[trainInd,:],X[sampleInd,:]) 
            Bone= np.inner(X[sampleInd,:],X[sampleInd,:]) 
            Brow=np.concatenate((Bcol.transpose(),Bone),axis=1)
            B=np.concatenate((B,Bcol),axis=1)
            B=np.concatenate((B,Brow),axis=0)
            #compute gram
            #note theta are in log format
            thetas=self.kernel_.theta
            thetas=np.exp(thetas)
            #rbf part
            krbf=np.exp(A*(-0.5)/(thetas[1]**2))
            np.fill_diagonal(krbf,1)
            krbf=thetas[0]*krbf
            #dot product part
            kdot=B+thetas[2]**2
            kdot=kdot*thetas[3]
            #note that we changed custom kernel, thetas[4] no longer exist
            #self.K_return=krbf+kdot+np.ones(kdot.shape)*thetas[4]
            self.K_return=krbf+kdot
            K=self.K_return
            #also save [A,B]
            self.mats_return=[A,B]

            
            
        else:
            K = self.kernel_(self.X_train_)
            self.K_return=K;
            A= pdist(self.X_train_, metric='sqeuclidean') #this is the flatten upper triangular ||xi-xj||_2
            A=squareform(A)   
            B= np.inner(self.X_train_, self.X_train_) 
            self.mats_return=[A,B]
            
            
            
        K[np.diag_indices_from(K)] += self.alpha
        t7=time.time()
        
        self.logger.info("compute matrix K takes----------- %s seconds", str(t7-t6))
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        t5=time.time()
        #print("retrain in (compute matrix K and invers and det) takes-----------  seconds:"+ str(t5-t6))
        self.logger.info("compute K.inv*t takes----------- %s seconds", str(t5-t7))
        self.logger.info('training ends in %s seconds-----------',str(t5-t1))
        return self
        
    def predict(self, X, return_std=False, return_cov=False,pre_k_trans=None,pre_test_mats=None,pre_candidate_mats=None,lastSampleIndex=0):
        
        """Predict using the Gaussian process regression model

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X = check_array(X)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = (C(1.0, constant_value_bounds="fixed") *
                          RBF(1.0, length_scale_bounds="fixed"))
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            self.logger.info('start GPR prediction');
            t1=time.time();
            if((pre_k_trans is None) and (pre_test_mats is None) and (pre_candidate_mats is None)):
            #then need to compute it
                K_trans = self.kernel_(X, self.X_train_)
                #save K_trans
                self.k_trans_ini=K_trans;
             #we also want the dotProduct mat and N2distance mat between test and train data~
                test_A= cdist(X,self.X_train_, metric='sqeuclidean') #this is the flatten upper triangular ||xi-xj||_2
                test_B= np.inner(X,self.X_train_) 
                self.test_mats_return=[test_A,test_B]
                self.candidate_mats_return=[test_A,test_B]
            elif(pre_test_mats is not None):#the candidate and test trans mat uses different update rules.
                #then use the test_A,test_B
                test_A=pre_test_mats[0]
                test_B=pre_test_mats[1]
                sampleInd=range(int(self.X_train_.shape[0])-self.newData,int(self.X_train_.shape[0]))
                #update test_A
                test_Acol=cdist(X,self.X_train_[sampleInd,:],metric='sqeuclidean');
                test_A=np.concatenate((test_A,test_Acol),axis=1)
                #update test_B
                test_Bcol=np.inner(X,self.X_train_[sampleInd,:])
                test_B=np.concatenate((test_B,test_Bcol),axis=1)
                #compute the new gram
                #note theta are in log format
                thetas=self.kernel_.theta
                thetas=np.exp(thetas)
                #rbf part
                krbf=np.exp(test_A*(-0.5)/(thetas[1]**2))
                np.fill_diagonal(krbf,1)
                krbf=thetas[0]*krbf
                #dot product part
                kdot=test_B+thetas[2]**2
                kdot=kdot*thetas[3]
                #K_trans=krbf+kdot+np.ones(kdot.shape)*thetas[4]
                K_trans=krbf+kdot
                #also save [test_A,test_B]
                self.test_mats_return=[test_A,test_B]
            elif(pre_candidate_mats is not None):
                sampleInd=range(int(self.X_train_.shape[0])-self.newData,int(self.X_train_.shape[0]))
                candidate_A=pre_candidate_mats[0]
                candidate_B=pre_candidate_mats[1]
                #first del a row who index the data that has been sampled from the candaitate pool
                candidate_A_sub=np.delete(candidate_A,lastSampleIndex,axis=0);
                candidate_B_sub=np.delete(candidate_B,lastSampleIndex,axis=0);
                #then add a col with the distance(N2 and dot) between the chosen sample and the rest of the candidate
                candidate_A_col=cdist(X,self.X_train_[sampleInd,:],metric='sqeuclidean');    
                candidate_B_col=np.inner(X,self.X_train_[sampleInd,:])            
                #concateate
                candidate_A=np.concatenate((candidate_A_sub,candidate_A_col),axis=1);
                candidate_B=np.concatenate((candidate_B_sub,candidate_B_col),axis=1);
                #compute the new gram matrix
                thetas=self.kernel_.theta
                thetas=np.exp(thetas)
                #rbf part
                krbf=np.exp(candidate_A*(-0.5)/(thetas[1]**2))
                np.fill_diagonal(krbf,1)
                krbf=thetas[0]*krbf
                #dot product part
                kdot=candidate_B+thetas[2]**2
                kdot=kdot*thetas[3]
                #K_trans=krbf+kdot+np.ones(kdot.shape)*thetas[4]    
                K_trans=krbf+kdot
                #save [candidate_A,candidate_B]
                self.candidate_mats_return=[candidate_A,candidate_B]
                
            elif(pre_k_trans is not None):#else use the k_trans passed to this method
                K_trans=pre_k_trans;
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal..
            t3=time.time();
            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # compute inverse K_inv of K based on its Cholesky
                # decomposition L and its inverse L_inv
                L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
                K_inv = L_inv.dot(L_inv.T)
                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                t2=time.time();
                self.logger.info('prediction(for candidate pool) finished in %s sec',str(t2-t1));
                self.logger.info('compute std in prediction (for candidate pool) finished in %s sec',str(t2-t3));
                
                self.logger.info('compute y_mean in prediction (for candidate pool) finished in %s sec',str(t3-t1));
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                t2=time.time();
                #because in AL, the prediction over test pool is for drawing the learning curve, so no need to returen std or cov
                
                self.logger.info('prediction(for test pool) finished in %s sec',str(t2-t1));
                
                
                return y_mean

    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like, shape = (n_samples_X, n_features)
            Query points where the GP samples are evaluated

        n_samples : int, default: 1
            The number of samples drawn from the Gaussian process

        random_state : int, RandomState instance or None, optional (default=0)
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the
            random number generator; If None, the random number
            generator is the RandomState instance used by `np.random`.

        Returns
        -------
        y_samples : array, shape = (n_samples_X, [n_output_dims], n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        rng = check_random_state(random_state)

        y_mean, y_cov = self.predict(X, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = \
                [rng.multivariate_normal(y_mean[:, i], y_cov,
                                         n_samples).T[:, np.newaxis]
                 for i in range(y_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,pre_K=None):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        t1=time.time()
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel_.clone_with_theta(theta)
        #kernel.dists_mat=self.kernel_.dists_mat
                
        if eval_gradient:
            #use incremental way of computing K
#            if(previous_K is not None):
#                self.logger.info('In computing log marginal likelihood:use previous K_n and x_new to compute K_n+1');
#                #pass previous K to kernel function kernel.__call__ to compute new K
#                K= np.concatenate((previous_K,self.kernel_(X[0:X.shape[0]-1,:],X[[X.shape[0]-1],:])),axis=1);
#                row=np.concatenate((self.kernel_(X[0:X.shape[0]-1,:],X[[X.shape[0]-1],:]).T,self.kernel_(X[[X.shape[0]-1],:],X[[X.shape[0]-1],:])),axis=1)     
#                K=np.concatenate((K,row),axis=0)
#                self.K_return=K;
            
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
            #self.kernel_.dists_mat=kernel.dists_mat;
        else:
            
            if(pre_K is None):
                K = kernel(self.X_train_)
            else:#fast compute K note that after fit() of the model has been called, the A,B returned from A,B have already have the correct size. we just need to compute the K(do not need to update,A,B like we did in fit.)
                A=pre_K[0]
                B=pre_K[1]

                #compute gram
                #note theta are in log format
                thetas=np.exp(theta)
                #rbf part
                krbf=np.exp(A*(-0.5)/(thetas[1]**2))
                np.fill_diagonal(krbf,1)
                krbf=thetas[0]*krbf
                #dot product part
                kdot=B+thetas[2]**2
                kdot=kdot*thetas[3]
                #note that we changed custom kernel, thetas[4] no longer exist
                #self.K_return=krbf+kdot+np.ones(kdot.shape)*thetas[4]
                K=krbf+kdot
                #also save [A,B]

        K[np.diag_indices_from(K)] += self.alpha
        t2=time.time()
        self.logger.info('compute K in logLikelihood in %s sec',str(t2-t1));  
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
        
        t3=time.time()    
        self.logger.info('logLikelihood computation finished in %s sec',str(t3-t1));    
        
        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds);
            self.logger.info('Number of function calls(likelihood) during the optimization: %s', str(convergence_dict["funcalls"]));
            self.logger.info('Number of iterations calls(likelihood) during the optimization: %s',str(convergence_dict["nit"]));
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict)
            
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min
