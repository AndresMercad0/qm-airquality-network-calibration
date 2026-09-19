"""
Machine learning calibration package (step s9)

Gathers the public functions of the package: dataset preparation, OLS
refit, model factory, training loop, evaluation, drift diagnostic,
plots and Markdown report.
"""

from .data_loader import prepare_dataset, get_feature_columns, get_train_test_masks
from .ols_refit import refit_ols_on_train
from .models import create_model
from .trainer import train_all_combinations
from .evaluator import evaluate_model, diebold_mariano_test, compute_feature_importance
from .drift import compute_monthly_drift, select_best_model_per_gas
from .plots import generate_all_plots
from .report import generate_report
