import os
from sklearn.ensemble import  RandomForestRegressor
from .RF_backend import _parallel_build_trees, _get_n_samples_bootstrap
from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble._forest import _partition_estimators, check_is_fitted
from sklearn.utils import check_random_state
from warnings import  warn
from scipy.sparse import issparse
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.exceptions import DataConversionWarning
from sklearn.tree._tree import DTYPE, DOUBLE


MAX_INT = np.iinfo(np.int32).max


def _accumulate_prediction(prediction, out):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    if prediction.ndim == 1:
        prediction = np.reshape(prediction, (-1, 1))
    out += prediction


def _accumulate_prediction_dis(prediction, out,  out_dis):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    if prediction.ndim == 1:
        prediction = np.reshape(prediction, (-1, 1))
    out += prediction
    out_dis += prediction**2


def _accumulate_prediction_smp(prediction, btp_weight, JK_sum, out, out_IJ, out_JK, out_dis):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    if prediction.ndim == 1:
        prediction = np.reshape(prediction, (-1, 1))

    out += prediction
    out_dis += prediction**2
    weights = btp_weight[None, :, None]
    out_IJ += prediction[:,None,:]*weights
    coeff = (weights == 0)
    JK_sum += coeff
    out_JK += prediction[:,None,:]*coeff


def loop_accumulate(estimators, X, JK_coeff, y_hat, y_IJ, y_JK, y_dis, mode):
    for e in estimators:
        prediction = e.predict(X, check_input=False)
        if mode == "disagreement":
            _accumulate_prediction_dis(prediction, y_hat, y_dis)
        elif mode == "sampling":
            btp_weight = e.bootstrap_weights
            _accumulate_prediction_smp(prediction, btp_weight, JK_coeff, y_hat, y_IJ, y_JK, y_dis)
        else:
            _accumulate_prediction(prediction, y_hat)
            

class MyRFRegressor(RandomForestRegressor):
    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        # Check parameters
        self._validate_estimator()
        # TODO: Remove in v1.2
        if isinstance(self, (RandomForestRegressor)):
            pass
            # if self.criterion == "mse":
            #     warn(
            #         "Criterion 'mse' was deprecated in v1.0 and will be "
            #         "removed in version 1.2. Use `criterion='squared_error'` "
            #         "which is equivalent.",
            #         FutureWarning,
            #     )
            # elif self.criterion == "mae":
            #     warn(
            #         "Criterion 'mae' was deprecated in v1.0 and will be "
            #         "removed in version 1.2. Use `criterion='absolute_error'` "
            #         "which is equivalent.",
            #         FutureWarning,
            #     )

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.

            self.sum_weights = np.zeros((len(X),))
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(
                delayed(_parallel_build_trees)(
                    t,
                    self,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    sum_weight = self.sum_weights
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self


    def predict(self, X, return_error = None):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        size_train = self.estimators_[0].size_train
        #if self.n_outputs_ > 1:
        y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        y_dis = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        y_IJ = np.zeros((X.shape[0], size_train, self.n_outputs_), dtype=np.float64)
        y_JK = np.zeros((X.shape[0], size_train, self.n_outputs_), dtype=np.float64)
        # else:
        #     y_hat = np.zeros((X.shape[0]), dtype=np.float64)
        #     y_dis = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        #     y_IJ = np.zeros((X.shape[0], size_train, self.n_outputs_), dtype=np.float64)
        #     y_JK = np.zeros((X.shape[0], size_train, self.n_outputs_), dtype=np.float64)
        JK_coeff = np.zeros((X.shape[0], size_train, self.n_outputs_), dtype=np.float64)
        # Parallel loop
        #lock = threading.Lock()
        loop_accumulate(self.estimators_, X, JK_coeff, y_hat, y_IJ, y_JK, y_dis, return_error)
        if return_error is None:
        #     Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
        #         delayed(_accumulate_prediction)(e, X, y_hat, lock)
        #         for e in self.estimators_
        #     )
            y_hat /= self.n_estimators
            return y_hat

        elif return_error == "disagreement":
        #     Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
        #         delayed(_accumulate_prediction_dis)(e, X, y_hat, y_dis, lock)
        #         for e in self.estimators_
        #     )
            y_hat /= self.n_estimators
            y_dis /= self.n_estimators
            y_dis = y_dis - y_hat**2

            error = np.sum(y_dis, axis = 1)
            return y_hat, error
            
        elif return_error == "sampling":
        #     Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
        #         delayed(_accumulate_prediction_smp)(e, X, JK_coeff, y_hat, y_IJ, y_JK, y_dis, lock)
        #         for e in self.estimators_
        #    )
            y_hat /= self.n_estimators
            y_dis /= self.n_estimators
            y_dis = np.sum((y_dis - y_hat**2), axis = 1)

            y_IJ /= self.n_estimators
            y_IJ -= y_hat[:,None,:]*self.sum_weights[None,:,None]/self.n_estimators
            y_IJ = np.sum(y_IJ**2, axis = (1, 2))#/size_train

            y_JK = y_JK/(JK_coeff)
            y_JK -= y_hat[:, None, :]
            y_JK = np.sum(y_JK**2, axis = (1, 2))

            error = 0.5*(y_IJ +y_JK - np.exp(1)*size_train*y_dis/self.n_estimators)
            return y_hat, error

      
