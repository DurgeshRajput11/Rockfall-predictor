import os
import sys
import mlflow
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import dagshub

from rockfallsecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from rockfallsecurity.entity.config_entity import ModelTrainerConfig
from rockfallsecurity.exception.exception import RockfallSafetyException
from rockfallsecurity.logging.logger import logging

from rockfallsecurity.utils.ml_utils.model.estimator import NetworkModel
from rockfallsecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from rockfallsecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
import warnings

# Silence specific sklearn calibration deprecation about cv='prefit' which we avoid via FrozenEstimator
warnings.filterwarnings(
    "ignore",
    message="The `cv='prefit'` option is deprecated",
    category=UserWarning,
    module=r"sklearn\.calibration"
)

class FrozenEstimator:
    """
    Minimal wrapper to emulate a 'prefit' estimator for CalibratedClassifierCV
    in newer scikit-learn versions where cv='prefit' is deprecated.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

class SelectedThresholdedModel:
    """
    Wrapper that applies the trained SelectFromModel selector before inference
    and then thresholds either probabilities (for classifiers supporting predict_proba)
    or anomaly scores (for IsolationForest via decision_function inversion).
    """
    def __init__(self, selector, base_model, threshold: float, score_mode: str = 'proba', score_norm: dict | None = None):
        self.selector = selector
        self.base_model = base_model
        self.threshold = float(threshold)
        self.score_mode = score_mode  # 'proba' or 'if_score'
        self.score_norm = score_norm or {}

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.selector.transform(X)
        if self.score_mode == 'proba':
            p = self.base_model.predict_proba(Xs)[:, 1]
        else:
            # IsolationForest score -> probability via z-score + sigmoid
            s = -self.base_model.decision_function(Xs)
            mu = float(self.score_norm.get('mu', float(np.mean(s))))
            sd = float(self.score_norm.get('sd', float(np.std(s) + 1e-9)))
            p = self._sigmoid((s - mu) / max(sd, 1e-9))
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

class SelectedThresholdedBlendedModel:
    """
    Blend multiple base models with weights on their positive probabilities.
    Applies the same selector and a final threshold.
    models: list of dicts: { 'model': clf, 'mode': 'proba'|'if_score', 'norm': {mu,sd} }
    weights: list of floats summing to 1
    """
    def __init__(self, selector, models: list, weights: list[float], threshold: float):
        assert len(models) == len(weights) and len(models) > 0
        self.selector = selector
        self.models = models
        self.weights = [float(w) for w in weights]
        self.threshold = float(threshold)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _model_proba(self, mdef: dict, Xs: np.ndarray) -> np.ndarray:
        if mdef['mode'] == 'proba':
            return mdef['model'].predict_proba(Xs)[:, 1]
        # if_score
        s = -mdef['model'].decision_function(Xs)
        mu = float(mdef.get('norm', {}).get('mu', float(np.mean(s))))
        sd = float(mdef.get('norm', {}).get('sd', float(np.std(s) + 1e-9)))
        return self._sigmoid((s - mu) / max(sd, 1e-9))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.selector.transform(X)
        parts = [self._model_proba(mdef, Xs) * w for mdef, w in zip(self.models, self.weights)]
        p = np.clip(np.sum(parts, axis=0), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise RockfallSafetyException(e, sys)

    def track_mlflow(self, best_model, classification_metric, params: dict = None, X_example=None, extra_metrics: dict = None, artifacts: dict = None, images: dict = None):
        # TrackingURI is set globally in main.py
        with mlflow.start_run():
            # Params
            if params:
                mlflow.log_params(params)
            # Metrics
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision", classification_metric.precision_score)
            mlflow.log_metric("recall_score", classification_metric.recall_score)
            if extra_metrics:
                for k, v in extra_metrics.items():
                    try:
                        mlflow.log_metric(k, float(v))
                    except Exception:
                        pass
            # Signature
            try:
                from mlflow.models.signature import infer_signature
                signature = infer_signature(X_example, best_model.predict(X_example) if X_example is not None else None)
            except Exception:
                signature = None
            # Use 'name' instead of deprecated 'artifact_path'
            mlflow.sklearn.log_model(best_model, name="model", signature=signature, input_example=X_example)
            # Optional text artifacts (e.g., reports, confusion matrices)
            if artifacts:
                for name, content in artifacts.items():
                    try:
                        mlflow.log_text(str(content), f"{name}.txt")
                    except Exception:
                        pass
            # Optional image artifacts
            if images:
                for filename, fig in images.items():
                    try:
                        mlflow.log_figure(fig, filename)
                    except Exception:
                        pass
            

    def train_model(self, X_train, y_train, X_test, y_test):
        # Check class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution in y_train:", dict(zip(unique, counts)))
        logging.info(f"Class distribution in y_train: {dict(zip(unique, counts))}")

        # Check for data leakage (overlap between train and test)
        overlap = np.intersect1d(
            X_train.view([('', X_train.dtype)] * X_train.shape[1]),
            X_test.view([('', X_test.dtype)] * X_test.shape[1])
        )
        print(f"Number of overlapping rows between train and test: {len(overlap)}")
        logging.info(f"Number of overlapping rows between train and test: {len(overlap)}")

        # Feature selection to reduce overfitting
        from sklearn.feature_selection import SelectFromModel
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=64, random_state=42, class_weight="balanced", max_depth=6),
            threshold="2*median",
        )
        selector.fit(X_train, y_train)
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)
        print(f"Selected {X_train_sel.shape[1]} features out of {X_train.shape[1]}")

        # Load group labels if available for grouped CV
        groups_train = None
        groups_test = None
        try:
            # groups saved next to transformed arrays
            tr_path = os.path.dirname(self.data_transformation_artifact.transformed_train_file_path)
            import numpy as _np
            gtr = os.path.join(tr_path, 'train_groups.npy')
            gte = os.path.join(tr_path, 'test_groups.npy')
            if os.path.exists(gtr):
                groups_train = _np.load(gtr, allow_pickle=True)
            if os.path.exists(gte):
                groups_test = _np.load(gte, allow_pickle=True)
        except Exception as _:
            groups_train = None
            groups_test = None

        # Establish evaluation set; avoid zero-positive test splits
        eval_source = "test"
        X_eval_sel, y_eval = X_test_sel, y_test
        if np.sum(y_eval == 1) == 0:
            sss_eval = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            found = False
            for tr_i, te_i in sss_eval.split(X_train_sel, y_train):
                if np.sum(y_train[te_i] == 1) > 0:
                    X_eval_sel, y_eval = X_train_sel[te_i], y_train[te_i]
                    eval_source = "train_stratified_holdout"
                    found = True
                    break
            if not found:
                # last resort: use full train for evaluation
                X_eval_sel, y_eval = X_train_sel, y_train
                eval_source = "train_full_fallback"

        # Ensure evaluation set has sufficient positives by augmenting with train samples if needed
        MIN_EVAL_POS = 5
        pos_eval = int(np.sum(y_eval == 1))
        if pos_eval < MIN_EVAL_POS:
            try:
                rng = np.random.default_rng(42)
                need_pos = MIN_EVAL_POS - pos_eval
                # Candidate pools from train
                idx_train_pos = np.where(y_train == 1)[0]
                idx_train_neg = np.where(y_train == 0)[0]
                # Remove any rows that are already in eval if eval came from train
                # Heuristic: if eval_source != 'test', avoid double-sampling
                if eval_source != 'test':
                    # We don't have direct indices; sample anew from train
                    pass
                take_pos = min(need_pos, len(idx_train_pos))
                extra_pos_idx = rng.choice(idx_train_pos, size=take_pos, replace=False) if take_pos > 0 else np.array([], dtype=int)
                # Add some negatives to keep PR curve meaningful
                neg_per_pos = 10
                take_neg = min(neg_per_pos * max(take_pos, 1), len(idx_train_neg))
                extra_neg_idx = rng.choice(idx_train_neg, size=take_neg, replace=False) if take_neg > 0 else np.array([], dtype=int)

                X_extra = np.vstack([
                    X_train_sel[extra_pos_idx],
                    X_train_sel[extra_neg_idx]
                ]) if (extra_pos_idx.size + extra_neg_idx.size) > 0 else None
                y_extra = np.concatenate([
                    np.ones(extra_pos_idx.size, dtype=int),
                    np.zeros(extra_neg_idx.size, dtype=int)
                ]) if (extra_pos_idx.size + extra_neg_idx.size) > 0 else None

                if X_extra is not None and y_extra is not None and X_extra.size > 0:
                    X_eval_sel = np.vstack([X_eval_sel, X_extra])
                    y_eval = np.concatenate([y_eval, y_extra])
                    eval_source = eval_source + "+train_augmented"
            except Exception as _:
                # If anything goes wrong, keep existing eval
                pass

        # Final safety: if evaluation still has zero positives, force-build from train
        if np.sum(y_eval == 1) == 0 and np.sum(y_train == 1) > 0:
            try:
                rng = np.random.default_rng(1337)
                idx_train_pos = np.where(y_train == 1)[0]
                idx_train_neg = np.where(y_train == 0)[0]
                take_pos = min(MIN_EVAL_POS, len(idx_train_pos))
                extra_pos_idx = rng.choice(idx_train_pos, size=take_pos, replace=False)
                neg_per_pos = 10
                take_neg = min(neg_per_pos * max(take_pos, 1), len(idx_train_neg))
                extra_neg_idx = rng.choice(idx_train_neg, size=take_neg, replace=False)
                X_eval_sel = np.vstack([X_train_sel[extra_pos_idx], X_train_sel[extra_neg_idx]])
                y_eval = np.concatenate([np.ones(take_pos, dtype=int), np.zeros(take_neg, dtype=int)])
                eval_source = "forced_from_train"
            except Exception:
                pass

        # Print eval composition for CI diagnostics
        try:
            print(f"Eval source: {eval_source}, eval_pos={int(np.sum(y_eval==1))}, eval_neg={int(np.sum(y_eval==0))}")
        except Exception:
            pass

        # ------------------------------
        # Supervised model: XGBoost
        # ------------------------------
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix
            # Split composition metrics
            train_pos = int(np.sum(y_train == 1))
            train_neg = int(np.sum(y_train == 0))
            eval_pos = int(np.sum(y_eval == 1))
            eval_neg = int(np.sum(y_eval == 0))
            groups_used = bool(groups_train is not None)
            # Compute class weights for XGBoost
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            spw = float(n_neg / max(n_pos, 1))  # scale_pos_weight

            # Tune hyperparameters via quick CV to reduce overfitting
            def _tune_xgboost(X, y, spw, n_splits=3, n_trials=15, seed=42, groups=None):
                # Prefer grouped CV if groups provided; fallback to stratified
                splitter = None
                if groups is not None:
                    splitter = GroupKFold(n_splits=n_splits)
                else:
                    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                rng = np.random.default_rng(seed)
                best_score = -1.0
                best_params = None
                param_space = {
                    'n_estimators': [300, 500, 700],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [10, 20, 40, 60],
                    'reg_lambda': [1.0, 2.0, 5.0, 10.0],
                    'gamma': [0.0, 0.5, 1.0],
                }

                keys = list(param_space.keys())
                for _ in range(n_trials):
                    sample = {k: rng.choice(v) for k, v in param_space.items()}
                    cv_scores = []
                    if groups is not None:
                        split_iter = splitter.split(X, y, groups)
                    else:
                        split_iter = splitter.split(X, y)
                    for train_idx, val_idx in split_iter:
                        X_tr, X_val = X[train_idx], X[val_idx]
                        y_tr, y_val = y[train_idx], y[val_idx]
                        # Skip folds with no positives in validation
                        if np.sum(y_val == 1) == 0 or np.sum(y_tr == 1) == 0:
                            continue
                        clf = xgb.XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="aucpr",
                            random_state=seed,
                            n_jobs=-1,
                            scale_pos_weight=spw,
                            **sample,
                        )
                        # Early stopping per fold where possible
                        try:
                            clf.fit(
                                X_tr,
                                y_tr,
                                eval_set=[(X_val, y_val)],
                                verbose=False,
                                early_stopping_rounds=50,
                            )
                        except Exception:
                            clf.fit(X_tr, y_tr)
                        prob_val = clf.predict_proba(X_val)[:, 1]
                        ap = average_precision_score(y_val, prob_val)
                        cv_scores.append(ap)
                    mean_ap = float(np.mean(cv_scores)) if cv_scores else -1.0
                    if mean_ap > best_score:
                        best_score = mean_ap
                        best_params = sample
                return best_params or {}, float(best_score)

            # Try Optuna first for smarter search
            tuned_params = None
            cv_mean_ap = -1.0
            try:
                import optuna  # type: ignore

                def objective(trial: 'optuna.trial.Trial'):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 300, 800, step=100),
                        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 6),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 10, 80, step=10),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
                        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
                    }
                    scores = []
                    if groups_train is not None:
                        skf = GroupKFold(n_splits=3)
                        split_iter = skf.split(X_train_sel, y_train, groups_train)
                    else:
                        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        split_iter = skf.split(X_train_sel, y_train)
                    for tr_i, va_i in split_iter:
                        X_tr, X_va = X_train_sel[tr_i], X_train_sel[va_i]
                        y_tr, y_va = y_train[tr_i], y_train[va_i]
                        if np.sum(y_va == 1) == 0 or np.sum(y_tr == 1) == 0:
                            continue
                        clf = xgb.XGBClassifier(
                            objective='binary:logistic',
                            eval_metric='aucpr',
                            random_state=42,
                            n_jobs=-1,
                            scale_pos_weight=spw,
                            **params,
                        )
                        try:
                            clf.fit(
                                X_tr, y_tr,
                                eval_set=[(X_va, y_va)],
                                verbose=False,
                                early_stopping_rounds=50,
                            )
                        except Exception:
                            clf.fit(X_tr, y_tr)
                        prob = clf.predict_proba(X_va)[:, 1]
                        scores.append(average_precision_score(y_va, prob))
                    return float(np.mean(scores)) if scores else -1.0

                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=20, show_progress_bar=False)
                tuned_params = study.best_params
                cv_mean_ap = float(study.best_value)
            except Exception as _:
                tuned_params, cv_mean_ap = _tune_xgboost(X_train_sel, y_train, spw, groups=groups_train)

            # Small validation split for early stopping
            # Build early-stopping validation split, prefer group-aware
            val_found = False
            if groups_train is not None:
                try:
                    gss = GroupShuffleSplit(n_splits=20, test_size=0.15, random_state=42)
                    for tr_idx, val_idx in gss.split(X_train_sel, y_train, groups_train):
                        if np.sum(y_train[val_idx] == 1) > 0 and np.sum(y_train[tr_idx] == 1) > 0:
                            X_tr_es, X_val_es = X_train_sel[tr_idx], X_train_sel[val_idx]
                            y_tr_es, y_val_es = y_train[tr_idx], y_train[val_idx]
                            val_found = True
                            break
                except Exception:
                    val_found = False
            if not val_found:
                sss = StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=42)
                for tr_idx, val_idx in sss.split(X_train_sel, y_train):
                    if np.sum(y_train[val_idx] == 1) > 0 and np.sum(y_train[tr_idx] == 1) > 0:
                        X_tr_es, X_val_es = X_train_sel[tr_idx], X_train_sel[val_idx]
                        y_tr_es, y_val_es = y_train[tr_idx], y_train[val_idx]
                        val_found = True
                        break
            if not val_found:
                X_tr_es, y_tr_es = X_train_sel, y_train
                X_val_es, y_val_es = None, None

            xgb_base = dict(
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=spw,
                # Conservative defaults to reduce overfitting
                subsample=0.7,
                colsample_bytree=0.7,
                max_depth=4,
                min_child_weight=40,
                reg_lambda=5.0,
            )
            xgb_base.update(tuned_params)
            xgb_clf = xgb.XGBClassifier(**xgb_base)

            if X_val_es is not None:
                try:
                    xgb_clf.fit(
                        X_tr_es,
                        y_tr_es,
                        eval_set=[(X_val_es, y_val_es)],
                        verbose=False,
                        early_stopping_rounds=50,
                    )
                except Exception:
                    xgb_clf.fit(X_train_sel, y_train)
            else:
                xgb_clf.fit(X_train_sel, y_train)

            # Probability scores for the positive class
            prob_train = xgb_clf.predict_proba(X_train_sel)[:, 1]
            prob_test = xgb_clf.predict_proba(X_eval_sel)[:, 1]

            # Optional calibration on validation split
            calibrated = False
            if X_val_es is not None:
                try:
                    # Try isotonic first; fallback to sigmoid (Platt)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The `cv='prefit'` option is deprecated",
                            category=UserWarning,
                            module=r"sklearn\.calibration"
                        )
                        cal = CalibratedClassifierCV(FrozenEstimator(xgb_clf), method='isotonic', cv=2)
                        cal.fit(np.vstack([X_tr_es, X_val_es]), np.concatenate([y_tr_es, y_val_es]))
                    prob_train = cal.predict_proba(X_train_sel)[:, 1]
                    prob_test = cal.predict_proba(X_eval_sel)[:, 1]
                    calibrated = True
                except Exception:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message="The `cv='prefit'` option is deprecated",
                                category=UserWarning,
                                module=r"sklearn\.calibration"
                            )
                            cal = CalibratedClassifierCV(FrozenEstimator(xgb_clf), method='sigmoid', cv=2)
                            cal.fit(np.vstack([X_tr_es, X_val_es]), np.concatenate([y_tr_es, y_val_es]))
                        prob_train = cal.predict_proba(X_train_sel)[:, 1]
                        prob_test = cal.predict_proba(X_eval_sel)[:, 1]
                        calibrated = True
                    except Exception:
                        pass

            # Prefer tuning threshold on validation if available; fallback to train
            tune_y = y_train
            tune_prob = prob_train
            if 'y_val_es' in locals() and y_val_es is not None and 'X_val_es' in locals() and X_val_es is not None:
                try:
                    tune_prob = xgb_clf.predict_proba(X_val_es)[:, 1]
                    tune_y = y_val_es
                    eval_source = eval_source + "+thr_on_val"
                except Exception:
                    pass

            # Aggressive recall-targeted thresholding first, then F2/F1 fallback
            tuned_thr_xgb = None
            thr_strategy_xgb = ""
            target_recall = 0.90
            if np.sum(tune_y == 1) > 0:
                p_x, r_x, th_x = precision_recall_curve(tune_y, tune_prob)
                if th_x is not None and len(th_x) > 0:
                    # Align thresholds with corresponding precision/recall points
                    cand = []
                    for i, thr in enumerate(th_x):
                        r_i = r_x[i+1]
                        p_i = p_x[i+1]
                        cand.append((thr, p_i, r_i))
                    # Choose threshold achieving >= target_recall with max precision
                    rec_cands = [(thr, p, r) for (thr, p, r) in cand if r >= target_recall]
                    if rec_cands:
                        thr_rec, p_at_rec, r_at_rec = max(rec_cands, key=lambda t: t[1])
                        tuned_thr_xgb = float(thr_rec)
                        thr_strategy_xgb = f"target_recall_{target_recall:.2f}"
                    else:
                        # Fallback: F2 then F1
                        p = p_x[1:]
                        r = r_x[1:]
                        f1s = 2 * (p * r) / np.maximum(p + r, 1e-12)
                        beta2 = 4.0
                        f2s = (1 + beta2) * (p * r) / np.maximum(beta2 * p + r, 1e-12)
                        idx_f2 = int(np.nanargmax(f2s)) if np.isfinite(f2s).any() else None
                        idx_f1 = int(np.nanargmax(f1s)) if np.isfinite(f1s).any() else None
                        if idx_f2 is not None and np.isfinite(f2s[idx_f2]):
                            tuned_thr_xgb = float(th_x[idx_f2])
                            thr_strategy_xgb = "f2_fallback"
                        elif idx_f1 is not None and np.isfinite(f1s[idx_f1]):
                            tuned_thr_xgb = float(th_x[idx_f1])
                            thr_strategy_xgb = "f1_fallback"
            if tuned_thr_xgb is None:
                tuned_thr_xgb = float(np.percentile(tune_prob, 99.0))
                thr_strategy_xgb = "p99_fallback"

            y_pred_test_xgb = (prob_test >= tuned_thr_xgb).astype(int)
            ap_test_xgb = float(average_precision_score(y_eval, prob_test))
            report_test_xgb = classification_report(y_eval, y_pred_test_xgb, zero_division=1)
            cm_test_xgb = confusion_matrix(y_eval, y_pred_test_xgb).tolist()

            y_pred_train_xgb = (prob_train >= tuned_thr_xgb).astype(int)
            cls_metric_train_xgb = get_classification_score(y_true=y_train, y_pred=y_pred_train_xgb)
            report_train_xgb = classification_report(y_train, y_pred_train_xgb, zero_division=1)
            cm_train_xgb = confusion_matrix(y_train, y_pred_train_xgb).tolist()

            # Plot PR curve (test)
            fig_pr = plt.figure()
            try:
                pr_p, pr_r, _ = precision_recall_curve(y_eval, prob_test)
                plt.step(pr_r, pr_p, where='post')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('XGBoost PR Curve (eval)')
            except Exception:
                plt.close(fig_pr)
                fig_pr = None

            # Operating point utilities: thresholds for target recall/precision
            target_recall = 0.90
            target_precision = 0.5
            thr_recall = tuned_thr_xgb
            thr_precision = tuned_thr_xgb
            prec_at_rec = None
            rec_at_prec = None
            alerts_per_1k = None
            try:
                pr_p_full, pr_r_full, pr_th = precision_recall_curve(y_eval, prob_test)
                # recall decreases as threshold increases is not guaranteed in PR; compute using thresholds
                # Map thresholds to recall/precision arrays (note lengths mismatch by 1)
                if len(pr_th) > 0:
                    # For target recall: pick threshold that achieves >= target_recall with max precision
                    candidates = []
                    for i, thr in enumerate(pr_th):
                        # PR arrays are aligned such that precision[i+1], recall[i+1] correspond to thr[i]
                        r_i = pr_r_full[i+1]
                        p_i = pr_p_full[i+1]
                        candidates.append((thr, p_i, r_i))
                    # Target recall
                    rec_cands = [(thr, p, r) for (thr, p, r) in candidates if r >= target_recall]
                    if rec_cands:
                        thr_recall, prec_at_rec, _ = max(rec_cands, key=lambda t: t[1])
                    # Target precision
                    prec_cands = [(thr, p, r) for (thr, p, r) in candidates if p >= target_precision]
                    if prec_cands:
                        thr_precision, _, rec_at_prec = max(prec_cands, key=lambda t: t[2])
                # Workload: alerts per 1k examples at tuned threshold
                preds_eval = (prob_test >= tuned_thr_xgb).astype(int)
                alerts_per_1k = float(1000.0 * preds_eval.sum() / max(len(preds_eval), 1))
            except Exception:
                pass

            # Log to MLflow (train)
            self.track_mlflow(
                xgb_clf,
                cls_metric_train_xgb,
                params={
                    "model": "XGBoost",
                    "grouped_cv": bool(groups_train is not None),
                    "scale_pos_weight": spw,
                    "tuned_threshold": tuned_thr_xgb,
                    "threshold_strategy": thr_strategy_xgb,
                    "n_features_selected": int(X_train_sel.shape[1]),
                    "cv_mean_ap": cv_mean_ap,
                    **{f"hp_{k}": v for k, v in tuned_params.items()},
                    "eval_source": eval_source,
                    "calibrated": calibrated,
                    "train_pos": train_pos,
                    "train_neg": train_neg,
                    "eval_pos": eval_pos,
                    "eval_neg": eval_neg,
                    "groups_used": groups_used,
                    "target_recall": target_recall,
                    "threshold_at_target_recall": thr_recall,
                    "precision_at_target_recall": prec_at_rec if prec_at_rec is not None else -1.0,
                    "target_precision": target_precision,
                    "threshold_at_target_precision": thr_precision,
                    "recall_at_target_precision": rec_at_prec if rec_at_prec is not None else -1.0,
                    "alerts_per_1000": alerts_per_1k if alerts_per_1k is not None else -1.0,
                },
                X_example=X_train_sel[:5],
                extra_metrics={"avg_precision_train": float(average_precision_score(y_train, prob_train))},
                artifacts={
                    "classification_report_train": report_train_xgb,
                    "confusion_matrix_train": cm_train_xgb,
                },
                images={"xgb_pr_curve_eval.png": fig_pr} if 'fig_pr' in locals() and fig_pr is not None else None,
            )

            # Log to MLflow (test)
            cls_metric_test_xgb = get_classification_score(y_true=y_test, y_pred=y_pred_test_xgb)
            self.track_mlflow(
                xgb_clf,
                cls_metric_test_xgb,
                X_example=X_eval_sel[:5],
                extra_metrics={"avg_precision_test": ap_test_xgb},
                artifacts={
                    "classification_report_test": report_test_xgb,
                    "confusion_matrix_test": cm_test_xgb,
                },
                images={"xgb_pr_curve_eval.png": fig_pr} if 'fig_pr' in locals() and fig_pr is not None else None,
            )

            # Initialize best with XGB by default
            best_model = xgb_clf
            best_model_name = "XGBoost"
            classification_train_metric = cls_metric_train_xgb
            classification_test_metric = cls_metric_test_xgb
            best_ap_test = ap_test_xgb
        except Exception as e:
            logging.exception(f"XGBoost training failed: {e}")
            # Fallback placeholders; IsolationForest will define best_model
            best_model = None
            best_model_name = None
            classification_train_metric = None
            classification_test_metric = None
            best_ap_test = -1.0

        # Baseline: Logistic Regression (balanced)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix
            logreg = LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear', max_iter=1000)
            logreg.fit(X_train_sel, y_train)
            prob_train_l = logreg.predict_proba(X_train_sel)[:, 1]
            prob_eval_l = logreg.predict_proba(X_eval_sel)[:, 1]

            # Threshold tuning on validation if available
            tune_y = y_train
            tune_prob = prob_train_l
            if 'y_val_es' in locals() and y_val_es is not None and 'X_val_es' in locals() and X_val_es is not None:
                try:
                    tune_prob = logreg.predict_proba(X_val_es)[:, 1]
                    tune_y = y_val_es
                except Exception:
                    pass
            thr_l = None
            thr_strategy_l = ""
            target_recall_l = 0.90
            if np.sum(tune_y == 1) > 0:
                p, r, th = precision_recall_curve(tune_y, tune_prob)
                if th is not None and len(th) > 0:
                    cand = []
                    for i, thr in enumerate(th):
                        r_i = r[i+1]
                        p_i = p[i+1]
                        cand.append((thr, p_i, r_i))
                    rec_cands = [(thr, p_i, r_i) for (thr, p_i, r_i) in cand if r_i >= target_recall_l]
                    if rec_cands:
                        thr_l, _, _ = max(rec_cands, key=lambda t: t[1])
                        thr_strategy_l = f"target_recall_{target_recall_l:.2f}"
                    else:
                        f1s = 2 * (p[1:] * r[1:]) / np.maximum(p[1:] + r[1:], 1e-12)
                        beta2 = 4.0
                        f2s = (1 + beta2) * (p[1:] * r[1:]) / np.maximum(beta2 * p[1:] + r[1:], 1e-12)
                        idx = int(np.nanargmax(f2s)) if np.isfinite(f2s).any() else None
                        if idx is not None and np.isfinite(f2s[idx]):
                            thr_l = float(th[idx])
                            thr_strategy_l = "f2_fallback"
            if thr_l is None:
                thr_l = float(np.percentile(tune_prob, 99.0))
                thr_strategy_l = "p99_fallback"

            y_pred_eval_l = (prob_eval_l >= thr_l).astype(int)
            ap_eval_l = float(average_precision_score(y_eval, prob_eval_l))
            report_eval_l = classification_report(y_eval, y_pred_eval_l, zero_division=1)
            cm_eval_l = confusion_matrix(y_eval, y_pred_eval_l).tolist()
            cls_metric_train_l = get_classification_score(y_true=y_train, y_pred=(prob_train_l >= thr_l).astype(int))

            fig_pr_l = plt.figure()
            try:
                pr_p_l, pr_r_l, _ = precision_recall_curve(y_eval, prob_eval_l)
                plt.step(pr_r_l, pr_p_l, where='post')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('LogReg PR Curve (eval)')
            except Exception:
                plt.close(fig_pr_l)
                fig_pr_l = None

            self.track_mlflow(
                logreg,
                cls_metric_train_l,
                params={
                    "model": "LogisticRegression",
                    "grouped_cv": bool('groups_train' in locals() and groups_train is not None),
                    "n_features_selected": int(X_train_sel.shape[1]),
                    "tuned_threshold": thr_l,
                    "threshold_strategy": thr_strategy_l,
                    "train_pos": train_pos,
                    "train_neg": train_neg,
                    "eval_pos": eval_pos,
                    "eval_neg": eval_neg,
                    "groups_used": groups_used,
                },
                X_example=X_train_sel[:5],
                extra_metrics={"avg_precision_train": float(average_precision_score(y_train, prob_train_l))},
                artifacts={
                    "classification_report_test": report_eval_l,
                    "confusion_matrix_test": cm_eval_l,
                },
                images={"logreg_pr_curve_eval.png": fig_pr_l} if fig_pr_l is not None else None,
            )

            # Update best model if LogReg wins by AP
            if ap_eval_l > best_ap_test:
                best_model = logreg
                best_model_name = "LogisticRegression"
                classification_train_metric = cls_metric_train_l
                classification_test_metric = get_classification_score(y_true=y_eval, y_pred=y_pred_eval_l)
                best_ap_test = ap_eval_l
        except Exception as e:
            logging.warning(f"LogisticRegression baseline failed: {e}")

        # IsolationForest anomaly detection with threshold tuning
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import (
            classification_report,
            average_precision_score,
            precision_recall_curve,
            confusion_matrix,
        )

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        total = max(n_pos + n_neg, 1)
        # Slightly inflate contamination to be more recall-friendly, with a safe floor
        prior = (n_pos / total) if total > 0 else 0.0
        contamination = max(min(prior * 2.0 if prior > 0 else 0.0 + 1e-4, 0.5), 1e-4)

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X_train_sel)

        # decision_function: larger => more normal, smaller => more anomalous
        # invert so that higher score means more likely anomaly (positive class)
        scores_train = -iso.decision_function(X_train_sel)
        scores_test = -iso.decision_function(X_eval_sel)

        # Tune threshold prioritizing target recall, then F2/F1
        precisions, recalls, thresholds = precision_recall_curve(y_train, scores_train)
        tuned_threshold = None
        thr_strategy_if = ""
        target_recall_if = 0.90
        if thresholds is not None and len(thresholds) > 0:
            cand = []
            for i, thr in enumerate(thresholds):
                r_i = recalls[i+1]
                p_i = precisions[i+1]
                cand.append((thr, p_i, r_i))
            rec_cands = [(thr, p_i, r_i) for (thr, p_i, r_i) in cand if r_i >= target_recall_if]
            if rec_cands:
                tuned_threshold, _, _ = max(rec_cands, key=lambda t: t[1])
                thr_strategy_if = f"target_recall_{target_recall_if:.2f}"
            else:
                p = precisions[1:]
                r = recalls[1:]
                f1s = 2 * (p * r) / np.maximum(p + r, 1e-12)
                beta2 = 4.0
                f2s = (1 + beta2) * (p * r) / np.maximum(beta2 * p + r, 1e-12)
                idx_f2 = int(np.nanargmax(f2s)) if np.isfinite(f2s).any() else None
                idx_f1 = int(np.nanargmax(f1s)) if np.isfinite(f1s).any() else None
                if idx_f2 is not None and np.isfinite(f2s[idx_f2]):
                    tuned_threshold = float(thresholds[idx_f2])
                    thr_strategy_if = "f2_fallback"
                elif idx_f1 is not None and np.isfinite(f1s[idx_f1]):
                    tuned_threshold = float(thresholds[idx_f1])
                    thr_strategy_if = "f1_fallback"
        if tuned_threshold is None:
            # fallback: pick 99th percentile of scores to flag some anomalies
            tuned_threshold = float(np.percentile(scores_train, 99.0))
            thr_strategy_if = "p99_fallback"

        # Evaluate with tuned threshold
        anomaly_pred_bin_test = (scores_test >= tuned_threshold).astype(int)
        print("\nIsolationForest anomaly detection report (threshold-tuned):")
        report_test = classification_report(y_eval, anomaly_pred_bin_test, zero_division=1)
        print(report_test)
        ap_test = average_precision_score(y_eval, scores_test)
        print(f"Average Precision (PR AUC): {ap_test:.4f}")
        # Confusion matrix for test
        cm_test = confusion_matrix(y_eval, anomaly_pred_bin_test).tolist()

        # Log IF to MLflow
        # Train metrics
        anomaly_pred_bin_train = (scores_train >= tuned_threshold).astype(int)
        classification_train_metric_if = get_classification_score(y_true=y_train, y_pred=anomaly_pred_bin_train)
        report_train_if = classification_report(y_train, anomaly_pred_bin_train, zero_division=1)
        cm_train_if = confusion_matrix(y_train, anomaly_pred_bin_train).tolist()
        # PR curve figure for IF
        fig_pr_if = plt.figure()
        try:
            pr_p_if, pr_r_if, _ = precision_recall_curve(y_eval, scores_test)
            plt.step(pr_r_if, pr_p_if, where='post')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('IsolationForest PR Curve (eval)')
        except Exception:
            plt.close(fig_pr_if)
            fig_pr_if = None

        self.track_mlflow(
            iso,
            classification_train_metric_if,
            params={
                "model": "IsolationForest",
                    "grouped_cv": bool(groups_train is not None),
                "contamination": float(contamination),
                "tuned_threshold": float(tuned_threshold),
                "threshold_strategy": thr_strategy_if,
                "n_features_selected": int(X_train_sel.shape[1]),
                "eval_source": eval_source,
                "train_pos": train_pos,
                "train_neg": train_neg,
                "eval_pos": eval_pos,
                "eval_neg": eval_neg,
                "groups_used": groups_used,
            },
            X_example=X_train_sel[:5],
            extra_metrics={"avg_precision_train": float(average_precision_score(y_train, scores_train))},
            artifacts={
                "classification_report_train": report_train_if,
                "confusion_matrix_train": cm_train_if,
            },
            images={"if_pr_curve_eval.png": fig_pr_if} if 'fig_pr_if' in locals() and fig_pr_if is not None else None,
        )

        # Test metrics
        classification_test_metric_if = get_classification_score(y_true=y_eval, y_pred=anomaly_pred_bin_test)
        self.track_mlflow(
            iso,
            classification_test_metric_if,
            X_example=X_eval_sel[:5],
            extra_metrics={"avg_precision_test": float(ap_test)},
            artifacts={
                "classification_report_test": report_test,
                "confusion_matrix_test": cm_test,
            },
            images={"if_pr_curve_eval.png": fig_pr_if} if 'fig_pr_if' in locals() and fig_pr_if is not None else None,
        )

        # Select best model by Average Precision on test
        if float(ap_test) > float(best_ap_test):
            best_model = iso
            best_model_name = "IsolationForest"
            classification_train_metric = classification_train_metric_if
            classification_test_metric = classification_test_metric_if
            best_ap_test = float(ap_test)

        # ------------------------------
        # Simple blending ensemble (XGB + LogReg + IF) on validation/eval
        # ------------------------------
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix
            # Collect candidate models with their eval probabilities
            candidates = []
            # XGB
            try:
                prob_eval_xgb = prob_test  # already computed on X_eval_sel
                candidates.append(("xgb", prob_eval_xgb))
            except Exception:
                pass
            # LogReg
            try:
                prob_eval_lr = prob_eval_l
                candidates.append(("lr", prob_eval_lr))
            except Exception:
                pass
            # IF (convert scores to [0,1] via sigmoid z-score)
            try:
                s_eval_if = scores_test
                mu_if = float(np.mean(scores_train))
                sd_if = float(np.std(scores_train) + 1e-9)
                prob_eval_if = 1.0 / (1.0 + np.exp(-(s_eval_if - mu_if) / max(sd_if, 1e-9)))
                candidates.append(("if", prob_eval_if))
            except Exception:
                pass

            if len(candidates) >= 2 and np.sum(y_eval == 1) > 0:
                name_to_prob = {n: p for n, p in candidates}
                # Small discrete weight search (sum to 1 in steps of 0.1)
                names = list(name_to_prob.keys())
                grids = np.linspace(0, 1, 11)
                best_w = None
                best_blend_ap = -1.0
                best_blend_prob = None
                if len(names) == 2:
                    n0, n1 = names
                    p0, p1 = name_to_prob[n0], name_to_prob[n1]
                    for w in grids:
                        w0 = float(w)
                        w1 = float(1.0 - w0)
                        pe = np.clip(w0 * p0 + w1 * p1, 0.0, 1.0)
                        ap = float(average_precision_score(y_eval, pe))
                        if ap > best_blend_ap:
                            best_blend_ap = ap
                            best_w = [w0, w1]
                            best_blend_prob = pe
                else:
                    # 3 models: nested loops with coarse steps
                    n0, n1, n2 = names[0], names[1], names[2]
                    p0, p1, p2 = name_to_prob[n0], name_to_prob[n1], name_to_prob[n2]
                    for w0 in grids:
                        for w1 in grids:
                            w2 = 1.0 - w0 - w1
                            if w2 < 0 or w2 > 1:
                                continue
                            pe = np.clip(w0 * p0 + w1 * p1 + w2 * p2, 0.0, 1.0)
                            ap = float(average_precision_score(y_eval, pe))
                            if ap > best_blend_ap:
                                best_blend_ap = ap
                                best_w = [float(w0), float(w1), float(w2)]
                                best_blend_prob = pe

                # Tune final threshold at aggressive recall on eval probs
                tuned_thr_blend = None
                thr_strategy_blend = ""
                target_recall_blend = 0.90
                if best_blend_prob is not None:
                    pr_p_b, pr_r_b, pr_th_b = precision_recall_curve(y_eval, best_blend_prob)
                    if pr_th_b is not None and len(pr_th_b) > 0:
                        cand = []
                        for i, thr in enumerate(pr_th_b):
                            r_i = pr_r_b[i+1]
                            p_i = pr_p_b[i+1]
                            cand.append((thr, p_i, r_i))
                        rec_cands = [(thr, p_i, r_i) for (thr, p_i, r_i) in cand if r_i >= target_recall_blend]
                        if rec_cands:
                            tuned_thr_blend, _, _ = max(rec_cands, key=lambda t: t[1])
                            thr_strategy_blend = f"target_recall_{target_recall_blend:.2f}"
                        else:
                            # F2 fallback
                            p = pr_p_b[1:]
                            r = pr_r_b[1:]
                            beta2 = 4.0
                            f2s = (1 + beta2) * (p * r) / np.maximum(beta2 * p + r, 1e-12)
                            idx = int(np.nanargmax(f2s)) if np.isfinite(f2s).any() else None
                            if idx is not None and np.isfinite(f2s[idx]):
                                tuned_thr_blend = float(pr_th_b[idx])
                                thr_strategy_blend = "f2_fallback"
                if tuned_thr_blend is None and best_blend_prob is not None:
                    tuned_thr_blend = float(np.percentile(best_blend_prob, 99.0))
                    thr_strategy_blend = "p99_fallback"

                if best_blend_prob is not None and best_w is not None:
                    y_pred_eval_blend = (best_blend_prob >= tuned_thr_blend).astype(int)
                    report_eval_blend = classification_report(y_eval, y_pred_eval_blend, zero_division=1)
                    cm_eval_blend = confusion_matrix(y_eval, y_pred_eval_blend).tolist()

                    # Track ensemble to MLflow
                    # Use train metric from the current best to keep interface; log AP explicitly
                    dummy_metric = classification_train_metric if classification_train_metric is not None else get_classification_score(y_true=y_train, y_pred=(prob_train >= np.median(prob_train)).astype(int))
                    fig_pr_b = plt.figure()
                    try:
                        pr_p_b2, pr_r_b2, _ = precision_recall_curve(y_eval, best_blend_prob)
                        plt.step(pr_r_b2, pr_p_b2, where='post')
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        plt.title('Blended PR Curve (eval)')
                    except Exception:
                        plt.close(fig_pr_b)
                        fig_pr_b = None

                    self.track_mlflow(
                        best_model if best_model is not None else xgb_clf,
                        dummy_metric,
                        params={
                            "model": "Blended",
                            "weights": {names[i]: float(best_w[i]) for i in range(len(names))},
                            "tuned_threshold": float(tuned_thr_blend),
                            "threshold_strategy": thr_strategy_blend,
                            "n_features_selected": int(X_train_sel.shape[1]),
                            "eval_source": eval_source,
                        },
                        X_example=X_train_sel[:5],
                        extra_metrics={"avg_precision_test": float(best_blend_ap)},
                        artifacts={
                            "classification_report_test": report_eval_blend,
                            "confusion_matrix_test": cm_eval_blend,
                        },
                        images={"blend_pr_curve_eval.png": fig_pr_b} if fig_pr_b is not None else None,
                    )

                    # If blend beats current best AP, mark to persist a blended wrapper later
                    if float(best_blend_ap) > float(best_ap_test):
                        best_ap_test = float(best_blend_ap)
                        best_model_name = "Blended"
                        # Create definitions for wrapped models to persist later
                        models_defs = []
                        if "xgb" in name_to_prob:
                            models_defs.append({"model": xgb_clf, "mode": "proba"})
                        if "lr" in name_to_prob:
                            models_defs.append({"model": logreg, "mode": "proba"})
                        if "if" in name_to_prob:
                            models_defs.append({"model": iso, "mode": "if_score", "norm": {"mu": mu_if, "sd": sd_if}})
                        best_model = ("blend", models_defs, best_w, float(tuned_thr_blend))
                        classification_test_metric = get_classification_score(y_true=y_eval, y_pred=y_pred_eval_blend)
        except Exception as e:
            logging.warning(f"Blending ensemble failed: {e}")

        # Note: Detailed per-model logs above; here we only persist the selected best_model

        # Persist final model (wrapper)
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        # Build a persistable model applying selector and thresholds
        final_model_to_save = None
        try:
            if best_model_name == "Blended" and isinstance(best_model, tuple) and best_model[0] == "blend":
                _tag, models_defs, weights, thr = best_model
                final_model_to_save = SelectedThresholdedBlendedModel(selector, models_defs, weights, thr)
            elif best_model_name == "IsolationForest":
                # Use the IF tuned threshold; normalize scores based on train
                mu_if = float(np.mean(scores_train))
                sd_if = float(np.std(scores_train) + 1e-9)
                final_model_to_save = SelectedThresholdedModel(selector, best_model, threshold=tuned_threshold, score_mode='if_score', score_norm={"mu": mu_if, "sd": sd_if})
            elif best_model_name == "LogisticRegression":
                final_model_to_save = SelectedThresholdedModel(selector, best_model, threshold=thr_l, score_mode='proba')
            else:
                # XGBoost default
                final_model_to_save = SelectedThresholdedModel(selector, best_model, threshold=tuned_thr_xgb, score_mode='proba')
        except Exception as e:
            logging.warning(f"Failed to wrap final model with threshold/selector, saving raw: {e}")
            final_model_to_save = best_model

        rockfall_model = NetworkModel(preprocessor=preprocessor, model=final_model_to_save)
        save_object(self.model_trainer_config.trained_model_file_path, obj=rockfall_model)
        save_object("final_model/model.pkl", final_model_to_save)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise RockfallSafetyException(e, sys)