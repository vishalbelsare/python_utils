import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import column_or_1d, check_consistent_length
from functools import reduce
from astropy.stats import jackknife_resampling
from .utils import check_binary_array


class StrategyEvaluator:
    def __init__(self, pipeline, strategy_grid):
        self.pipeline = pipeline
        self.strategy_grid = strategy_grid
        self.outer_scoring = {}
        self.inner_scoring = {}
        self.outer_cv = KFold()
        self.inner_cv = KFold()
        self.cv_results = []

    def evaluate(self, X, y,
                 outer_cv=None, inner_cv=None,
                 outer_scoring=None,
                 inner_scoring=None,
                 outer_jobs=1, inner_jobs=1, verbose=1):

        assert isinstance(inner_scoring, str)
        self.inner_scoring = inner_scoring
        assert isinstance(outer_scoring, dict) & bool(outer_scoring)
        self.outer_scoring = outer_scoring
        if outer_cv is not None:
            self.outer_cv = outer_cv
        if inner_cv is not None:
            self.inner_cv = inner_cv

        if 1 not in (inner_jobs, outer_jobs):
            raise ValueError('Cannot parallelise inner and outer CV')

        parallel = Parallel(n_jobs=outer_jobs, verbose=verbose)
        self.cv_results = parallel(
            delayed(self._fit_and_score)(X, y, train, test,
                                         verbose=verbose, n_jobs=inner_jobs)
            for train, test in outer_cv.split(X, y))

    def _fit_and_score(self, X, y, train, test, verbose=1, n_jobs=1):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        scores_by_params = dict.fromkeys(self.strategy_grid.keys())
        for i, (strategy, strategy_values) in enumerate(
                self.strategy_grid.items()):
            print(f'{strategy} ...')

            # inner-cv model selection
            gscv = GridSearchCV(
                self.pipeline, param_grid=strategy_values,
                scoring=self.inner_scoring, cv=self.inner_cv,
                verbose=verbose, return_train_score=False, refit=True,
                n_jobs=n_jobs)
            gscv.fit(X_train, y_train)

            # outer-cv model evaluation
            scores_by_params[strategy] = self._score(gscv, X_test, y_test)

        cv_results = scores_by_params
        return cv_results

    def _score(self, estimator, X, y):
        results_by_scorer = dict.fromkeys(self.outer_scoring.keys())
        for scorer_key, scorer_func in self.outer_scoring.items():
            mu, ci = scorer_func(estimator, X, y)
            results_by_scorer[scorer_key] = np.hstack([mu, ci])
        return results_by_scorer

    def summarize(self):
        results_cols = ['cv_fold', 'scorer', 'stat',
                        *self.strategy_grid.keys()]
        results = pd.DataFrame(columns=results_cols)
        for cv_fold in range(self.outer_cv.get_n_splits()):
            for scorer in self.outer_scoring.keys():
                results_list = [self.cv_results[cv_fold][strategy][scorer]
                                for strategy in self.strategy_grid.keys()]
                results_array = reduce(
                    lambda left, right: np.column_stack([left, right]),
                    results_list)
                results_df = pd.DataFrame(results_array,
                                          columns=self.strategy_grid.keys())
                results_df['cv_fold'] = cv_fold
                results_df['scorer'] = scorer
                results_df['stat'] = ['mean', 'stderr']
                results = results.append(results_df, sort=False,
                                         ignore_index=True)
        results.cv_fold = results.cv_fold.astype(np.int8)
        return results


class DistributedStrategyEvaluator(StrategyEvaluator):
    def __init__(self, pipeline, strategy_grid):
        super().__init__(pipeline, strategy_grid)

    def distributed_evaluate(self, X, y,
                             outer_cv=None, inner_cv=None,
                             outer_scoring=None, inner_scoring=None):

        assert isinstance(inner_scoring, str)
        self.inner_scoring = inner_scoring
        assert isinstance(outer_scoring, dict) & bool(outer_scoring)
        self.outer_scoring = outer_scoring
        if outer_cv is not None:
            self.outer_cv = outer_cv
        if inner_cv is not None:
            self.inner_cv = inner_cv

        self.cv_results = [self._distributed_fit_and_score(X, y, train, test)
                           for train, test in outer_cv.split(X, y)]

    def _distributed_fit_and_score(self, X, y, train, test):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        scores_by_params = dict.fromkeys(self.strategy_grid.keys())
        for i, (strategy, strategy_values) in enumerate(
                self.strategy_grid.items()):
            print(f'{strategy} ...')

            # inner-cv model selection
            gscv = DaskGridSearchCV(self.pipeline, param_grid=strategy_values,
                                    scoring=self.inner_scoring,
                                    cv=self.inner_cv, refit=True)
            gscv.fit(X_train, y_train)

            # outer-cv model evaluation
            scores_by_params[strategy] = self._score(gscv, X_test, y_test)

        cv_results = scores_by_params
        return cv_results


def evaluate_pointwise_score(y_true, y_pred, score_func):
    pointwise_scores = score_func(y_true, y_pred)
    check_consistent_length(pointwise_scores, y_true)
    mean_score = np.mean(pointwise_scores)
    stderr = np.std(pointwise_scores) / np.sqrt(pointwise_scores.shape[0])
    return mean_score, stderr


def evaluate_composite_score(y_true, y_pred, score_func):
    """
    Evaluate composite scores based on the contingency table like precision,
    specificity or sensitivity by using jackknife re-sampling.
    :param y_true:
    :param y_pred:
    :param score_func:
    :return: score and standard error

    References:
    Efron and Stein, (1981), "The jackknife estimate of variance."
    """

    def compute_jackknife_stderr(x):
        n = x.shape[0]
        return np.sqrt((((n - 1) / n) * np.sum((x - x.mean()) ** 2)))

    composite_score = score_func(y_true, y_pred)

    # jackknifing to obtain std err estimate
    index = np.arange(y_true.shape[0])
    jack_idx = jackknife_resampling(index).astype(np.int)
    jack_scores = np.array([score_func(y_true[idx], y_pred[idx])
                            for idx in jack_idx])
    jack_stderr = compute_jackknife_stderr(jack_scores)
    return composite_score, jack_stderr


def evaluate_score_func(y_true, y_pred, func=None, pointwise=False):
    check_binary_array(y_true)
    y_true = column_or_1d(y_true)
    if pointwise:
        mu, stderr = evaluate_pointwise_score(y_true, y_pred, func)
    else:
        mu, stderr = evaluate_composite_score(y_true, y_pred, func)
    return mu, stderr
