#!/usr/bin/env python3
"""
============================================================================
06_HYPERPARAMETER_TUNING.PY - Optimisation des Hyperparamètres
============================================================================

🎯 OBJECTIF:
   Trouver les meilleurs hyperparamètres pour Random Forest et XGBoost
   afin de réduire le MAE en dessous de 15nm !

📥 INPUT:
   - data/processed/dataset_train.csv
   - data/processed/dataset_test.csv

📤 OUTPUT:
   - models/rf_ex_max_tuned.joblib
   - models/rf_em_max_tuned.joblib
   - models/xgb_ex_max_tuned.joblib
   - models/xgb_em_max_tuned.joblib
   - reports/tuning_results.html
   - reports/tuning_curves.png

🔧 MÉTHODES:
   - RandomizedSearchCV (exploration large)
   - GridSearchCV (affinement)
   - Cross-validation 5-fold stratifiée

============================================================================
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import json

warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import (
        RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
    )
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("❌ scikit-learn requis! pip install scikit-learn")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost non installé (optionnel)")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optionnel: scipy pour distributions
try:
    from scipy.stats import randint, uniform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration."""
    
    DATA_DIR: Path = Path("data/processed")
    MODELS_DIR: Path = Path("models")
    REPORTS_DIR: Path = Path("reports")
    LOGS_DIR: Path = Path("logs")
    
    TRAIN_FILE: str = "dataset_train.csv"
    TEST_FILE: str = "dataset_test.csv"
    
    TARGETS: List[str] = None
    EXCLUDE_COLS: List[str] = None
    
    # Paramètres de tuning
    CV_FOLDS: int = 5
    N_ITER_RANDOM: int = 50  # Nombre d'itérations pour RandomizedSearch
    RANDOM_SEED: int = 42
    N_JOBS: int = -1  # Utiliser tous les cores
    
    # Grilles d'hyperparamètres
    RF_PARAM_DIST: Dict = None
    RF_PARAM_GRID_FINE: Dict = None
    XGB_PARAM_DIST: Dict = None
    XGB_PARAM_GRID_FINE: Dict = None
    
    def __post_init__(self):
        self.TARGETS = ['ex_max', 'em_max']
        self.EXCLUDE_COLS = [
            'protein_id', 'name', 'ex_max', 'em_max', 'qy', 
            'stokes_shift', 'ext_coeff', 'brightness'
        ]
        
        # === RANDOM FOREST ===
        # Distribution large pour exploration
        self.RF_PARAM_DIST = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
            'bootstrap': [True, False],
        }
        
        # Grille fine après exploration
        self.RF_PARAM_GRID_FINE = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 0.5],
        }
        
        # === XGBOOST ===
        self.XGB_PARAM_DIST = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9, 11],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0.1, 1, 10],
        }
        
        self.XGB_PARAM_GRID_FINE = {
            'n_estimators': [200, 300],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3],
        }


CONFIG = Config()


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = CONFIG.LOGS_DIR / f"tuning_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("🔧 HYPERPARAMETER TUNING - Optimisation des Modèles")
    logger.info("="*70)
    
    return logger


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def load_data(logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les datasets."""
    
    train_path = CONFIG.DATA_DIR / CONFIG.TRAIN_FILE
    test_path = CONFIG.DATA_DIR / CONFIG.TEST_FILE
    
    if not train_path.exists() or not test_path.exists():
        logger.error("❌ Fichiers de données non trouvés!")
        return None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"   Train: {len(train_df)} | Test: {len(test_df)}")
    
    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prépare les features."""
    
    feature_cols = [col for col in train_df.columns 
                    if col not in CONFIG.EXCLUDE_COLS 
                    and train_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Imputer
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    logger.info(f"   {len(feature_cols)} features")
    
    return X_train, X_test, feature_cols


# ============================================================================
# TUNING FUNCTIONS
# ============================================================================

def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       target_name: str, logger: logging.Logger) -> Tuple[RandomForestRegressor, Dict]:
    """
    Tune Random Forest en 2 phases:
    1. RandomizedSearchCV (exploration large)
    2. GridSearchCV (affinement)
    """
    
    logger.info(f"   🌲 Tuning Random Forest pour {target_name}...")
    
    cv = KFold(n_splits=CONFIG.CV_FOLDS, shuffle=True, random_state=CONFIG.RANDOM_SEED)
    
    # Phase 1: Exploration avec RandomizedSearchCV
    logger.info(f"      Phase 1: RandomizedSearch ({CONFIG.N_ITER_RANDOM} itérations)...")
    
    rf_base = RandomForestRegressor(random_state=CONFIG.RANDOM_SEED, n_jobs=CONFIG.N_JOBS)
    
    random_search = RandomizedSearchCV(
        rf_base,
        param_distributions=CONFIG.RF_PARAM_DIST,
        n_iter=CONFIG.N_ITER_RANDOM,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_random = random_search.best_params_
    best_score_random = -random_search.best_score_
    
    logger.info(f"      Best MAE (random): {best_score_random:.2f} nm")
    
    # Phase 2: Affinement avec GridSearchCV autour des meilleurs params
    logger.info(f"      Phase 2: GridSearch (affinement)...")
    
    # Construire une grille fine autour des meilleurs params
    fine_grid = {}
    
    # n_estimators: ±100 autour du meilleur
    best_n = best_random.get('n_estimators', 200)
    fine_grid['n_estimators'] = [max(100, best_n - 100), best_n, best_n + 100]
    
    # max_depth: ±5 autour du meilleur
    best_depth = best_random.get('max_depth', 15)
    if best_depth is None:
        fine_grid['max_depth'] = [20, 25, None]
    else:
        fine_grid['max_depth'] = [max(5, best_depth - 5), best_depth, best_depth + 5]
    
    # min_samples_split: valeurs proches
    best_split = best_random.get('min_samples_split', 5)
    fine_grid['min_samples_split'] = [max(2, best_split - 2), best_split, best_split + 2]
    
    # min_samples_leaf
    best_leaf = best_random.get('min_samples_leaf', 2)
    fine_grid['min_samples_leaf'] = [max(1, best_leaf - 1), best_leaf, best_leaf + 1]
    
    # max_features: garder le meilleur + alternatives
    best_features = best_random.get('max_features', 'sqrt')
    fine_grid['max_features'] = [best_features]
    if best_features != 'sqrt':
        fine_grid['max_features'].append('sqrt')
    if best_features != 0.5:
        fine_grid['max_features'].append(0.5)
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=CONFIG.RANDOM_SEED, n_jobs=CONFIG.N_JOBS),
        param_grid=fine_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=CONFIG.N_JOBS,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    
    logger.info(f"      Best MAE (grid): {best_score:.2f} nm")
    logger.info(f"      Best params: {best_params}")
    
    # Entraîner le modèle final
    best_model = RandomForestRegressor(
        **best_params,
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS
    )
    best_model.fit(X_train, y_train)
    
    return best_model, {
        'best_params': best_params,
        'cv_mae': best_score,
        'random_search_mae': best_score_random,
        'cv_results': grid_search.cv_results_
    }


def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 target_name: str, logger: logging.Logger) -> Tuple[Optional['XGBRegressor'], Dict]:
    """Tune XGBoost en 2 phases."""
    
    if not HAS_XGBOOST:
        return None, {}
    
    logger.info(f"   🚀 Tuning XGBoost pour {target_name}...")
    
    cv = KFold(n_splits=CONFIG.CV_FOLDS, shuffle=True, random_state=CONFIG.RANDOM_SEED)
    
    # Phase 1: RandomizedSearchCV
    logger.info(f"      Phase 1: RandomizedSearch ({CONFIG.N_ITER_RANDOM} itérations)...")
    
    xgb_base = XGBRegressor(
        random_state=CONFIG.RANDOM_SEED, 
        n_jobs=CONFIG.N_JOBS,
        verbosity=0
    )
    
    random_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=CONFIG.XGB_PARAM_DIST,
        n_iter=CONFIG.N_ITER_RANDOM,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_random = random_search.best_params_
    best_score_random = -random_search.best_score_
    
    logger.info(f"      Best MAE (random): {best_score_random:.2f} nm")
    
    # Phase 2: GridSearchCV
    logger.info(f"      Phase 2: GridSearch (affinement)...")
    
    # Grille fine autour des meilleurs params
    fine_grid = {
        'n_estimators': [
            max(100, best_random.get('n_estimators', 200) - 50),
            best_random.get('n_estimators', 200),
            best_random.get('n_estimators', 200) + 50
        ],
        'max_depth': [
            max(3, best_random.get('max_depth', 7) - 2),
            best_random.get('max_depth', 7),
            best_random.get('max_depth', 7) + 2
        ],
        'learning_rate': [best_random.get('learning_rate', 0.1)],
        'subsample': [best_random.get('subsample', 0.8)],
        'colsample_bytree': [best_random.get('colsample_bytree', 0.8)],
        'min_child_weight': [best_random.get('min_child_weight', 1)],
    }
    
    # Ajouter les params de régularisation
    if 'reg_alpha' in best_random:
        fine_grid['reg_alpha'] = [best_random['reg_alpha']]
    if 'reg_lambda' in best_random:
        fine_grid['reg_lambda'] = [best_random['reg_lambda']]
    if 'gamma' in best_random:
        fine_grid['gamma'] = [best_random['gamma']]
    
    grid_search = GridSearchCV(
        XGBRegressor(random_state=CONFIG.RANDOM_SEED, n_jobs=CONFIG.N_JOBS, verbosity=0),
        param_grid=fine_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=CONFIG.N_JOBS,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    
    logger.info(f"      Best MAE (grid): {best_score:.2f} nm")
    logger.info(f"      Best params: {best_params}")
    
    # Modèle final
    best_model = XGBRegressor(
        **best_params,
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS,
        verbosity=0
    )
    best_model.fit(X_train, y_train)
    
    return best_model, {
        'best_params': best_params,
        'cv_mae': best_score,
        'random_search_mae': best_score_random,
        'cv_results': grid_search.cv_results_
    }


# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str, target_name: str, logger: logging.Logger) -> Dict:
    """Évalue le modèle tuné sur le test set."""
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"      Test MAE: {mae:.2f} nm | RMSE: {rmse:.2f} nm | R²: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'target': target_name,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }


def compare_with_baseline(tuned_results: Dict, baseline_mae: float, 
                          target: str, logger: logging.Logger):
    """Compare les résultats tunés avec le baseline."""
    
    tuned_mae = tuned_results['test_mae']
    improvement = baseline_mae - tuned_mae
    pct_improvement = (improvement / baseline_mae) * 100
    
    logger.info(f"      📊 Comparaison {target}:")
    logger.info(f"         Baseline MAE: {baseline_mae:.2f} nm")
    logger.info(f"         Tuned MAE:    {tuned_mae:.2f} nm")
    
    if improvement > 0:
        logger.info(f"         ✅ Amélioration: -{improvement:.2f} nm ({pct_improvement:.1f}%)")
    else:
        logger.info(f"         ⚠️ Pas d'amélioration: +{-improvement:.2f} nm")
    
    return improvement


# ============================================================================
# VISUALISATIONS
# ============================================================================

def create_comparison_plot(all_results: List[Dict], baseline_results: Dict,
                           output_dir: Path, logger: logging.Logger):
    """Crée un graphique comparant baseline vs tuned."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Organiser les résultats
    results_by_target = {}
    for r in all_results:
        target = r['target']
        if target not in results_by_target:
            results_by_target[target] = []
        results_by_target[target].append(r)
    
    plot_idx = 0
    for target, results in results_by_target.items():
        for r in results:
            if plot_idx >= 4:
                break
            
            ax = axes[plot_idx // 2, plot_idx % 2]
            
            y_test = r['y_test']
            y_pred = r['y_pred']
            
            ax.scatter(y_test, y_pred, alpha=0.6, s=40, c='#3498db')
            
            # Ligne parfaite
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')
            
            ax.set_xlabel(f'{target} réel (nm)', fontsize=11)
            ax.set_ylabel(f'{target} prédit (nm)', fontsize=11)
            ax.set_title(
                f"{r['model_name']} - {target}\n"
                f"MAE={r['test_mae']:.1f}nm, R²={r['test_r2']:.3f}",
                fontsize=12, fontweight='bold'
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    plt.suptitle('Modèles Optimisés - Prédictions vs Réalité', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'tuning_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_improvement_plot(improvements: Dict, output_dir: Path, logger: logging.Logger):
    """Crée un graphique des améliorations."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(improvements.keys())
    values = list(improvements.values())
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax.barh(models, values, color=colors)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Amélioration MAE (nm)', fontsize=12)
    ax.set_title('Amélioration après Tuning\n(positif = meilleur)', fontsize=14, fontweight='bold')
    
    # Annotations
    for bar, val in zip(bars, values):
        x_pos = val + 0.2 if val >= 0 else val - 0.5
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{val:+.1f} nm', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'tuning_improvement.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


# ============================================================================
# RAPPORT HTML
# ============================================================================

def generate_report(all_results: List[Dict], tuning_info: Dict,
                    improvements: Dict, output_dir: Path, logger: logging.Logger):
    """Génère le rapport HTML."""
    
    # Table des résultats
    results_rows = ""
    for r in all_results:
        color = "#27ae60" if r['test_mae'] < 18 else "#f39c12" if r['test_mae'] < 22 else "#e74c3c"
        results_rows += f"""
        <tr>
            <td>{r['model_name']}</td>
            <td>{r['target']}</td>
            <td style="color:{color};font-weight:bold">{r['test_mae']:.2f} nm</td>
            <td>{r['test_rmse']:.2f} nm</td>
            <td>{r['test_r2']:.4f}</td>
        </tr>
        """
    
    # Table des hyperparamètres
    params_html = ""
    for key, info in tuning_info.items():
        if 'best_params' in info:
            params_html += f"<h4>{key}</h4><ul>"
            for param, value in info['best_params'].items():
                params_html += f"<li><code>{param}</code>: {value}</li>"
            params_html += f"</ul><p>CV MAE: {info.get('cv_mae', 'N/A'):.2f} nm</p>"
    
    # Améliorations
    improvements_html = "<ul>"
    for key, val in improvements.items():
        icon = "✅" if val > 0 else "⚠️"
        improvements_html += f"<li>{icon} <strong>{key}</strong>: {val:+.2f} nm</li>"
    improvements_html += "</ul>"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Hyperparameter Tuning - FP Prediction</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e67e22; padding-bottom: 10px; }}
        h2 {{ color: #d35400; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background: #e67e22; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .section {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .highlight {{ background: linear-gradient(135deg, #f39c12, #e74c3c); color: white; 
                     padding: 20px; border-radius: 10px; margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 15px 0; }}
        code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🔧 Hyperparameter Tuning - Résultats</h1>
    <p><em>Généré le {datetime.now():%Y-%m-%d %H:%M}</em></p>
    
    <div class="highlight">
        <h3>📊 Résumé</h3>
        <p>Méthode: RandomizedSearchCV ({CONFIG.N_ITER_RANDOM} itérations) + GridSearchCV (affinement)</p>
        <p>Cross-validation: {CONFIG.CV_FOLDS} folds</p>
    </div>
    
    <div class="section">
        <h2>📈 Performances des Modèles Optimisés</h2>
        <table>
            <tr>
                <th>Modèle</th>
                <th>Cible</th>
                <th>Test MAE</th>
                <th>Test RMSE</th>
                <th>Test R²</th>
            </tr>
            {results_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>📉 Amélioration vs Baseline</h2>
        {improvements_html}
        <img src="tuning_improvement.png" alt="Improvement">
    </div>
    
    <div class="section">
        <h2>🔧 Meilleurs Hyperparamètres</h2>
        {params_html}
    </div>
    
    <div class="section">
        <h2>📊 Prédictions vs Réalité</h2>
        <img src="tuning_predictions.png" alt="Predictions">
    </div>
    
    <div class="section">
        <h2>🚀 Prochaines Étapes</h2>
        <ol>
            <li><strong>08_ensemble.py</strong> - Combiner RF + XGBoost (stacking)</li>
            <li><strong>09_feature_engineering.py</strong> - Ajouter ESM-2 embeddings</li>
            <li><strong>Validation finale</strong> - Test sur protéines nouvelles</li>
        </ol>
    </div>
    
</body>
</html>
"""
    
    report_path = output_dir / "tuning_results.html"
    report_path.write_text(html, encoding='utf-8')
    logger.info(f"   📄 {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not HAS_SKLEARN:
        print("❌ scikit-learn requis!")
        return
    
    logger = setup_logging()
    
    # 1. Charger les données
    logger.info("")
    logger.info("📂 ÉTAPE 1: Chargement des données")
    logger.info("-" * 50)
    
    train_df, test_df = load_data(logger)
    if train_df is None:
        return
    
    X_train, X_test, feature_names = prepare_features(train_df, test_df, logger)
    
    # Baseline MAE (depuis les résultats précédents)
    BASELINE_MAE = {
        'ex_max': 21.93,  # RF baseline
        'em_max': 19.28   # RF baseline
    }
    
    all_results = []
    tuning_info = {}
    improvements = {}
    all_models = {}
    
    # 2. Tuning pour chaque cible
    for target in CONFIG.TARGETS:
        logger.info("")
        logger.info(f"🎯 ÉTAPE 2: Tuning pour {target}")
        logger.info("-" * 50)
        
        y_train = train_df[target].values
        y_test = test_df[target].values
        
        # Random Forest
        logger.info("")
        rf_model, rf_info = tune_random_forest(X_train, y_train, target, logger)
        tuning_info[f'RF_{target}'] = rf_info
        
        rf_results = evaluate_model(rf_model, X_test, y_test, "RF Tuned", target, logger)
        all_results.append(rf_results)
        all_models[f'rf_{target}_tuned'] = rf_model
        
        imp = compare_with_baseline(rf_results, BASELINE_MAE[target], target, logger)
        improvements[f'RF_{target}'] = imp
        
        # XGBoost
        if HAS_XGBOOST:
            logger.info("")
            xgb_model, xgb_info = tune_xgboost(X_train, y_train, target, logger)
            if xgb_model:
                tuning_info[f'XGB_{target}'] = xgb_info
                
                xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGB Tuned", target, logger)
                all_results.append(xgb_results)
                all_models[f'xgb_{target}_tuned'] = xgb_model
                
                # Comparer avec baseline XGBoost
                xgb_baseline = {'ex_max': 23.28, 'em_max': 19.29}
                imp_xgb = compare_with_baseline(xgb_results, xgb_baseline[target], target, logger)
                improvements[f'XGB_{target}'] = imp_xgb
    
    # 3. Visualisations
    logger.info("")
    logger.info("📊 ÉTAPE 3: Visualisations")
    logger.info("-" * 50)
    
    CONFIG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plot(all_results, BASELINE_MAE, CONFIG.REPORTS_DIR, logger)
    create_improvement_plot(improvements, CONFIG.REPORTS_DIR, logger)
    generate_report(all_results, tuning_info, improvements, CONFIG.REPORTS_DIR, logger)
    
    # 4. Sauvegarder les modèles
    logger.info("")
    logger.info("💾 ÉTAPE 4: Sauvegarde des modèles")
    logger.info("-" * 50)
    
    if HAS_JOBLIB:
        CONFIG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, model in all_models.items():
            path = CONFIG.MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"   💾 {path.name}")
    
    # 5. Résumé final
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    
    # Meilleur modèle par cible
    for target in CONFIG.TARGETS:
        target_results = [r for r in all_results if r['target'] == target]
        if target_results:
            best = min(target_results, key=lambda x: x['test_mae'])
            baseline = BASELINE_MAE[target]
            improvement = baseline - best['test_mae']
            
            logger.info(f"\n   🎯 {target}:")
            logger.info(f"      Meilleur: {best['model_name']}")
            logger.info(f"      MAE: {best['test_mae']:.2f} nm (baseline: {baseline:.2f})")
            logger.info(f"      Amélioration: {improvement:+.2f} nm")
    
    # Objectif atteint?
    best_overall = min(all_results, key=lambda x: x['test_mae'])
    logger.info("")
    
    if best_overall['test_mae'] < 15:
        logger.info("🎉 OBJECTIF ATTEINT! MAE < 15nm")
    elif best_overall['test_mae'] < 18:
        logger.info("✅ TRÈS BON! MAE < 18nm - Proche de l'objectif!")
    else:
        logger.info("👍 Améliorations obtenues - Continuer avec feature engineering")
    
    logger.info("")
    logger.info("📁 Fichiers générés:")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/tuning_results.html")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/tuning_predictions.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/tuning_improvement.png")
    logger.info(f"   - {CONFIG.MODELS_DIR}/*_tuned.joblib")


if __name__ == "__main__":
    main()
