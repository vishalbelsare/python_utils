import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from sklearn.utils import column_or_1d, check_consistent_length
from functools import reduce
from astropy.stats import jackknife_resampling
from .utils import check_binary_array


class DistributedStrategyEvaluator(object):
    def __init__(self, pipeline, strategy_grid):
        self.pipeline = pipeline
        self.strategy_grid = strategy_grid
        self.outer_scoring = {}
        self.inner_scoring = {}
        self.outer_cv = KFold(n_splits=10)
        self.inner_cv = KFold(n_splits=10)
        self.cv_results = []

    def evaluate(self, X, y, outer_cv=None, inner_cv=None,
                 outer_scoring=None, inner_scoring=None):

        assert isinstance(inner_scoring, str)
        self.inner_scoring = inner_scoring
        assert isinstance(outer_scoring, dict) & bool(outer_scoring)
        self.outer_scoring = outer_scoring
        if outer_cv is not None:
            self.outer_cv = outer_cv
        if inner_cv is not None:
            self.inner_cv = inner_cv

        self.cv_results = [self._fit_and_score(X, y, train, test)
                           for train, test in outer_cv.split(X, y)]

    def _fit_and_score(self, X, y, train, test):
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

        return scores_by_params

    def _score(self, estimator, X, y):
        results_by_scorer = dict.fromkeys(self.outer_scoring.keys())
        for scorer_key, scorer_func in self.outer_scoring.items():
            mu, stderr = scorer_func(estimator, X, y)
            results_by_scorer[scorer_key] = np.hstack([mu, stderr])
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


def evaluate_score_func(y_true, y_pred, func=None, pointwise=False):

    def _evaluate_pointwise_score(y_true, y_pred, score_func):
        pointwise_scores = score_func(y_true, y_pred)
        check_consistent_length(pointwise_scores, y_true)
        mean_score = np.mean(pointwise_scores)
        n = pointwise_scores.shape[0]
        stderr = np.std(pointwise_scores) / np.sqrt(n - 1)
        return mean_score, stderr

    def _evaluate_composite_score(y_true, y_pred, score_func):
        """
        Evaluate composite scores based on the contingency table like
        precision, specificity or sensitivity by using jackknife re-sampling.
        :param y_true:
        :param y_pred:
        :param score_func:
        :return: score and standard error

        References:
        Efron and Stein, (1981), "The jackknife estimate of variance."
        """

        def _compute_jackknife_stderr(x):
            n = x.shape[0]
            # np.sqrt((((n - 1) / n) * np.sum((x - x.mean()) ** 2)))
            return np.sqrt(n - 1) * np.std(x)

        composite_score = score_func(y_true, y_pred)

        # jackknifing to obtain std err estimate
        index = np.arange(y_true.shape[0])
        jack_idx = jackknife_resampling(index).astype(np.int)
        jack_scores = np.array([score_func(y_true[idx], y_pred[idx])
                                for idx in jack_idx])
        jack_stderr = _compute_jackknife_stderr(jack_scores)
        return composite_score, jack_stderr

    check_binary_array(y_true)
    y_true = column_or_1d(y_true)
    if pointwise:
        mu, stderr = _evaluate_pointwise_score(y_true, y_pred, func)
    else:
        mu, stderr = _evaluate_composite_score(y_true, y_pred, func)
    return mu, stderr
