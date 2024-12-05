<<<<<<< HEAD
# The Software is copyright (c) Commonwealth Scientific and Industrial Research Organisation (CSIRO) 2023-2024.

=======
>>>>>>> temp-branch
import warnings

import optuna
import pandas as pd
import sklearn
from loguru import logger
from matplotlib.figure import Figure
from optuna import Trial
from optuna.visualization import plot_optimization_history
from skelm import ELMRegressor

<<<<<<< HEAD
from hvac_gym.utils.data_utils import unique
=======
>>>>>>> temp-branch
from hvac_gym.vis.vis_tools import figs_to_html


def split_alternate_days(df: pd.DataFrame, n_sets: int = 2) -> list[pd.DataFrame]:
    """
    Splits df into two separate odd and even day-of-year sets.  Useful for creating training/validation sets without seasonaal bias
    :param df: a multi-day dataframe with a datetime index
    :param n_sets How many datasets to return. n_sets=2 will return 2 datasets containing alternate days.  n_sets will give 3 sets with every third
    day, etc.
    :return: [train, test]
    """
<<<<<<< HEAD
=======

>>>>>>> temp-branch
    df["doy"] = [d.timetuple().tm_yday for d in df.index.date]

    sets = []
    for s in range(0, n_sets):
        sets.append(df[df["doy"] % n_sets == s].sort_index().drop(columns=["doy"]))

    # no days from any set should be in any other set
    for s in range(0, n_sets - 1):
<<<<<<< HEAD
        assert len(set(unique(sets[s].index.date)).intersection(set(unique(sets[s + 1].index.date)))) == 0
=======
        assert (
            len(
                set(pd.unique(sets[s].index.date)).intersection(
                    set(pd.unique(sets[s + 1].index.date))
                )
            )
            == 0
        )
>>>>>>> temp-branch

    df.drop(columns=["doy"], inplace=True)  # noqa

    return sets


<<<<<<< HEAD
def elm_optuna_param_search(x_train: pd.DataFrame, y_train: pd.DataFrame, n_trials: int = 100) -> tuple[ELMRegressor, Figure]:
=======
def elm_optuna_param_search(
    x_train: pd.DataFrame, y_train: pd.DataFrame, n_trials: int = 100
) -> tuple[ELMRegressor, Figure]:
>>>>>>> temp-branch
    """
    Performs an optuna search for the best hyperparameters for an ELMRegressor model, trying to maximise cross-validation score on the
    provided training set.
    :param x_train: the training set features
    :param y_train: the training set target
    :param n_trials: the number of optuna trials to perform
    :return: the best model (unfitted, with best found params) and the optuna optimization history plot
    """

    def objective(trial: Trial) -> float:
        """Optuna search for the ELMRegressor hyperparams"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, y = x_train, y_train
            classifier_obj = ELMRegressor(
                n_neurons=trial.suggest_int("n_neurons", 10, 500, log=False),
<<<<<<< HEAD
                ufunc=trial.suggest_categorical("ufunc", ["tanh", "sigm", "relu", "lin"]),
                alpha=trial.suggest_loguniform("alpha", 1e-7, 1e-1),
                include_original_features=trial.suggest_categorical("include_original_features", [True, False]),
                density=trial.suggest_float("density", 1e-3, 1, step=0.1),
                pairwise_metric=trial.suggest_categorical("pairwise_metric", ["euclidean", "cityblock", "cosine", None]),
            )
            score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=5)
=======
                ufunc=trial.suggest_categorical(
                    "ufunc", ["tanh", "sigm", "relu", "lin"]
                ),
                alpha=trial.suggest_loguniform("alpha", 1e-7, 1e-1),
                include_original_features=trial.suggest_categorical(
                    "include_original_features", [True, False]
                ),
                density=trial.suggest_float("density", 1e-3, 1, step=0.1),
                pairwise_metric=trial.suggest_categorical(
                    "pairwise_metric", ["euclidean", "cityblock", "cosine", None]
                ),
            )
            score = sklearn.model_selection.cross_val_score(
                classifier_obj, x, y, n_jobs=-1, cv=5
            )
>>>>>>> temp-branch
            accuracy = float(score.mean())
        return accuracy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    trial = study.best_trial
    logger.info(f"Best model: ELMRegressor(**{trial.params})")

    fig = plot_optimization_history(study)
    figs_to_html([fig], "output/Optuna Search.html", True)

    model = ELMRegressor(**trial.params)

    return model, fig
