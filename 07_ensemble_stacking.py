#!/usr/bin/env python3
"""
============================================================================
07_ENSEMBLE_STACKING.PY - Modèle Ensemble par Stacking
============================================================================

🎯 OBJECTIF:
   Combiner les prédictions de plusieurs modèles pour obtenir
   de meilleures performances que chaque modèle individuel.

📚 PRINCIPE DU STACKING:
   
   Niveau 0 (Base models):
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ Random      │  │ XGBoost     │  │ Ridge       │
   │ Forest      │  │             │  │ Regression  │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          │                │                │
          ▼                ▼                ▼
        pred_rf         pred_xgb        pred_ridge
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
   Niveau 1 (Meta-model):
   ┌─────────────────────────────────────────────┐
   │           Ridge / Linear Regression          │
   │   (apprend à combiner les prédictions)      │
   └─────────────────────────────────────────────┘
                           │
                           ▼
                    Prédiction Finale

📥 INPUT:
   - data/processed/dataset_train.csv
   - data/processed/dataset_test.csv

📤 OUTPUT:
   - models/stacking_ex_max.joblib
   - models/stacking_em_max.joblib
   - reports/ensemble_results.html
   - reports/ensemble_comparison.png

============================================================================
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import (
        RandomForestRegressor, 
        StackingRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor
    )
    from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import cross_val_score, KFold
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
    print("⚠️ XGBoost non installé")

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
    
    CV_FOLDS: int = 5
    RANDOM_SEED: int = 42
    N_JOBS: int = -1
    
    def __post_init__(self):
        self.TARGETS = ['ex_max', 'em_max']
        self.EXCLUDE_COLS = [
            'protein_id', 'name', 'ex_max', 'em_max', 'qy', 
            'stokes_shift', 'ext_coeff', 'brightness'
        ]


CONFIG = Config()


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = CONFIG.LOGS_DIR / f"ensemble_{datetime.now():%Y%m%d_%H%M%S}.log"
    
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
    logger.info("🎭 ENSEMBLE STACKING - Combinaison de Modèles")
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
                     logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """Prépare les features avec normalisation."""
    
    feature_cols = [col for col in train_df.columns 
                    if col not in CONFIG.EXCLUDE_COLS 
                    and train_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Imputer
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Normalisation (important pour Ridge, SVR, KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"   {len(feature_cols)} features")
    
    return X_train_scaled, X_test_scaled, feature_cols, scaler


# ============================================================================
# DÉFINITION DES MODÈLES DE BASE
# ============================================================================

def get_base_estimators(logger: logging.Logger) -> List[Tuple[str, any]]:
    """
    Définit les modèles de base pour le stacking.
    
    On utilise des modèles DIVERSIFIÉS:
    - Random Forest: capture les interactions non-linéaires
    - XGBoost: boosting avec régularisation
    - Extra Trees: variance différente de RF
    - Ridge: modèle linéaire régularisé
    - KNN: approche locale
    """
    
    estimators = []
    
    # 1. Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS
    )
    estimators.append(('rf', rf))
    logger.info("   ✅ Random Forest")
    
    # 2. XGBoost
    if HAS_XGBOOST:
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=CONFIG.RANDOM_SEED,
            n_jobs=CONFIG.N_JOBS,
            verbosity=0
        )
        estimators.append(('xgb', xgb))
        logger.info("   ✅ XGBoost")
    
    # 3. Extra Trees (plus de variance que RF)
    et = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS
    )
    estimators.append(('extra_trees', et))
    logger.info("   ✅ Extra Trees")
    
    # 4. Gradient Boosting (sklearn version)
    gb = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=CONFIG.RANDOM_SEED
    )
    estimators.append(('gradient_boost', gb))
    logger.info("   ✅ Gradient Boosting")
    
    # 5. Ridge Regression (linéaire)
    ridge = Ridge(alpha=1.0)
    estimators.append(('ridge', ridge))
    logger.info("   ✅ Ridge Regression")
    
    # 6. KNN (approche locale)
    knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
    estimators.append(('knn', knn))
    logger.info("   ✅ KNN")
    
    return estimators


# ============================================================================
# CRÉATION DU STACKING
# ============================================================================

def create_stacking_model(estimators: List[Tuple[str, any]], 
                          logger: logging.Logger) -> StackingRegressor:
    """
    Crée le modèle de stacking.
    
    Le meta-model (final_estimator) apprend à combiner
    les prédictions des modèles de base.
    
    On utilise Ridge comme meta-model car:
    - Simple et rapide
    - Régularisation évite l'overfitting
    - Fonctionne bien avec peu d'inputs (nombre de base models)
    """
    
    # Meta-model: Ridge avec régularisation
    meta_model = Ridge(alpha=1.0)
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=CONFIG.CV_FOLDS,  # Cross-validation pour générer les meta-features
        n_jobs=CONFIG.N_JOBS,
        passthrough=False  # Ne pas inclure les features originaux dans le meta-model
    )
    
    logger.info(f"   📦 Stacking créé avec {len(estimators)} modèles de base")
    logger.info(f"   🎯 Meta-model: Ridge")
    
    return stacking


# ============================================================================
# MÉTHODES D'ENSEMBLE ALTERNATIVES
# ============================================================================

def simple_average_ensemble(models: List, X: np.ndarray) -> np.ndarray:
    """Moyenne simple des prédictions."""
    predictions = np.array([model.predict(X) for model in models])
    return np.mean(predictions, axis=0)


def weighted_average_ensemble(models: List, weights: List[float], X: np.ndarray) -> np.ndarray:
    """Moyenne pondérée des prédictions."""
    predictions = np.array([model.predict(X) for model in models])
    weights = np.array(weights) / np.sum(weights)
    return np.average(predictions, axis=0, weights=weights)


def train_individual_models(estimators: List[Tuple[str, any]], 
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           logger: logging.Logger) -> Tuple[List, Dict]:
    """
    Entraîne les modèles individuellement pour comparaison
    et pour l'ensemble par moyenne.
    """
    
    trained_models = []
    results = {}
    
    for name, model in estimators:
        logger.info(f"      Training {name}...")
        model.fit(X_train, y_train)
        trained_models.append(model)
        
        # Évaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae, 
            'rmse': rmse,
            'r2': r2, 
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred
        }
        logger.info(f"         MAE: {mae:.2f} nm | R²: {r2:.4f}")
    
    return trained_models, results


def optimize_weights(models: List, X_val: np.ndarray, y_val: np.ndarray,
                     logger: logging.Logger) -> np.ndarray:
    """
    Optimise les poids pour la moyenne pondérée.
    
    Utilise une approche simple: poids inversement proportionnel au MAE.
    """
    
    maes = []
    for model in models:
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        maes.append(mae)
    
    # Poids inversement proportionnel au MAE
    maes = np.array(maes)
    weights = 1.0 / (maes + 1e-6)  # Éviter division par zéro
    weights = weights / weights.sum()
    
    logger.info(f"   Poids optimisés: {dict(zip(range(len(weights)), np.round(weights, 3)))}")
    
    return weights


# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str, logger: logging.Logger) -> Dict:
    """Évalue un modèle."""
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }


def cross_validate_ensemble(model, X: np.ndarray, y: np.ndarray,
                            logger: logging.Logger) -> float:
    """Cross-validation du modèle ensemble."""
    
    cv = KFold(n_splits=CONFIG.CV_FOLDS, shuffle=True, random_state=CONFIG.RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=CONFIG.N_JOBS)
    
    cv_mae = -scores.mean()
    cv_std = scores.std()
    
    logger.info(f"      CV MAE: {cv_mae:.2f} ± {cv_std:.2f} nm")
    
    return cv_mae


# ============================================================================
# VISUALISATIONS
# ============================================================================

def create_comparison_plot(all_results: Dict[str, Dict], target: str,
                           output_dir: Path, logger: logging.Logger):
    """Crée un graphique comparant tous les modèles."""
    
    if not HAS_MATPLOTLIB:
        return
    
    # Trier par MAE
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mae'])
    
    models = [r[0] for r in sorted_results]
    maes = [r[1]['mae'] for r in sorted_results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, maes, color=colors)
    
    # Ligne de référence (baseline)
    baseline_mae = 19.28 if target == 'em_max' else 21.93
    ax.axvline(x=baseline_mae, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_mae} nm')
    
    ax.set_xlabel('MAE (nm)', fontsize=12)
    ax.set_title(f'Comparaison des Modèles - {target}', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Annotations
    for bar, mae in zip(bars, maes):
        ax.text(mae + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{mae:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / f'ensemble_comparison_{target}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_predictions_plot(best_results: Dict, target: str,
                            output_dir: Path, logger: logging.Logger):
    """Crée le scatter plot pour le meilleur modèle."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    y_test = best_results['y_test']
    y_pred = best_results['y_pred']
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=50, c='#3498db')
    
    # Ligne parfaite
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')
    
    ax.set_xlabel(f'{target} réel (nm)', fontsize=12)
    ax.set_ylabel(f'{target} prédit (nm)', fontsize=12)
    ax.set_title(
        f"{best_results['model_name']} - {target}\n"
        f"MAE={best_results['mae']:.2f}nm, R²={best_results['r2']:.3f}",
        fontsize=14, fontweight='bold'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'ensemble_best_{target}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


# ============================================================================
# RAPPORT HTML
# ============================================================================

def generate_report(results_by_target: Dict, output_dir: Path, logger: logging.Logger):
    """Génère le rapport HTML."""
    
    tables_html = ""
    
    for target, all_results in results_by_target.items():
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mae'])
        
        rows = ""
        for i, (name, r) in enumerate(sorted_results):
            color = "#27ae60" if i == 0 else "#f39c12" if r['mae'] < 20 else "#e74c3c"
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            rows += f"""
            <tr>
                <td>{medal} {name}</td>
                <td style="color:{color};font-weight:bold">{r['mae']:.2f} nm</td>
                <td>{r['rmse']:.2f} nm</td>
                <td>{r['r2']:.4f}</td>
            </tr>
            """
        
        baseline = 19.28 if target == 'em_max' else 21.93
        best_mae = sorted_results[0][1]['mae']
        improvement = baseline - best_mae
        
        tables_html += f"""
        <div class="section">
            <h2>🎯 {target}</h2>
            <p><strong>Baseline:</strong> {baseline:.2f} nm | 
               <strong>Meilleur:</strong> {best_mae:.2f} nm | 
               <strong>Amélioration:</strong> {improvement:+.2f} nm</p>
            <table>
                <tr>
                    <th>Modèle</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R²</th>
                </tr>
                {rows}
            </table>
            <img src="ensemble_comparison_{target}.png" alt="Comparison">
            <img src="ensemble_best_{target}.png" alt="Best predictions">
        </div>
        """
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ensemble Stacking - FP Prediction</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }}
        h2 {{ color: #8e44ad; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background: #9b59b6; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        tr:first-child td {{ background: #d4edda; }}
        .section {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .highlight {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                     padding: 20px; border-radius: 10px; margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 15px 0; }}
    </style>
</head>
<body>
    <h1>🎭 Ensemble Stacking - Résultats</h1>
    <p><em>Généré le {datetime.now():%Y-%m-%d %H:%M}</em></p>
    
    <div class="highlight">
        <h3>📚 Principe du Stacking</h3>
        <p>Le stacking combine les prédictions de plusieurs modèles (Random Forest, XGBoost, 
        Extra Trees, Gradient Boosting, Ridge, KNN) via un meta-model (Ridge) qui apprend 
        à pondérer optimalement chaque contribution.</p>
    </div>
    
    {tables_html}
    
    <div class="section">
        <h2>📝 Conclusions</h2>
        <ul>
            <li>Le <strong>Stacking</strong> combine la diversité des modèles pour réduire la variance</li>
            <li>La <strong>moyenne pondérée</strong> est une alternative simple et efficace</li>
            <li>Pour améliorer davantage: ajouter des features (ESM-2 embeddings)</li>
        </ul>
    </div>
    
</body>
</html>
"""
    
    report_path = output_dir / "ensemble_results.html"
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
    
    X_train, X_test, feature_names, scaler = prepare_features(train_df, test_df, logger)
    
    # Baselines
    BASELINE_MAE = {'ex_max': 21.93, 'em_max': 19.28}
    
    results_by_target = {}
    best_models = {}
    
    # 2. Pour chaque cible
    for target in CONFIG.TARGETS:
        logger.info("")
        logger.info(f"🎯 ÉTAPE 2: Ensemble pour {target}")
        logger.info("-" * 50)
        
        y_train = train_df[target].values
        y_test = test_df[target].values
        
        all_results = {}
        
        # 2a. Définir les modèles de base
        logger.info("")
        logger.info("   📦 Création des modèles de base:")
        estimators = get_base_estimators(logger)
        
        # 2b. Entraîner individuellement pour comparaison
        logger.info("")
        logger.info("   🏋️ Entraînement individuel:")
        trained_models, individual_results = train_individual_models(
            estimators, X_train, y_train, X_test, y_test, logger
        )
        
        for name, r in individual_results.items():
            all_results[name] = r
        
        # 2c. Stacking
        logger.info("")
        logger.info("   🎭 Stacking:")
        stacking = create_stacking_model(estimators, logger)
        
        logger.info("      Training stacking...")
        stacking.fit(X_train, y_train)
        
        stacking_results = evaluate_model(stacking, X_test, y_test, "Stacking", logger)
        all_results['Stacking'] = stacking_results
        logger.info(f"      MAE: {stacking_results['mae']:.2f} nm | R²: {stacking_results['r2']:.4f}")
        
        # 2d. Moyenne simple
        logger.info("")
        logger.info("   📊 Moyenne simple:")
        y_pred_avg = simple_average_ensemble(trained_models, X_test)
        mae_avg = mean_absolute_error(y_test, y_pred_avg)
        r2_avg = r2_score(y_test, y_pred_avg)
        all_results['Moyenne Simple'] = {
            'mae': mae_avg, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred_avg)),
            'r2': r2_avg, 'y_test': y_test, 'y_pred': y_pred_avg
        }
        logger.info(f"      MAE: {mae_avg:.2f} nm | R²: {r2_avg:.4f}")
        
        # 2e. Moyenne pondérée (poids optimisés sur train)
        logger.info("")
        logger.info("   ⚖️ Moyenne pondérée:")
        weights = optimize_weights(trained_models, X_train, y_train, logger)
        y_pred_weighted = weighted_average_ensemble(trained_models, weights, X_test)
        mae_weighted = mean_absolute_error(y_test, y_pred_weighted)
        r2_weighted = r2_score(y_test, y_pred_weighted)
        all_results['Moyenne Pondérée'] = {
            'mae': mae_weighted, 'rmse': np.sqrt(mean_squared_error(y_test, y_pred_weighted)),
            'r2': r2_weighted, 'y_test': y_test, 'y_pred': y_pred_weighted
        }
        logger.info(f"      MAE: {mae_weighted:.2f} nm | R²: {r2_weighted:.4f}")
        
        results_by_target[target] = all_results
        
        # Meilleur modèle
        best_name = min(all_results.keys(), key=lambda k: all_results[k]['mae'])
        best_models[target] = {
            'name': best_name,
            'results': all_results[best_name],
            'model': stacking if best_name == 'Stacking' else None
        }
        
        logger.info("")
        logger.info(f"   🏆 Meilleur: {best_name} (MAE: {all_results[best_name]['mae']:.2f} nm)")
    
    # 3. Visualisations
    logger.info("")
    logger.info("📊 ÉTAPE 3: Visualisations")
    logger.info("-" * 50)
    
    CONFIG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for target, all_results in results_by_target.items():
        create_comparison_plot(all_results, target, CONFIG.REPORTS_DIR, logger)
        
        best_name = best_models[target]['name']
        best_results = all_results[best_name]
        best_results['model_name'] = best_name
        create_predictions_plot(best_results, target, CONFIG.REPORTS_DIR, logger)
    
    generate_report(results_by_target, CONFIG.REPORTS_DIR, logger)
    
    # 4. Sauvegarder les modèles stacking
    logger.info("")
    logger.info("💾 ÉTAPE 4: Sauvegarde")
    logger.info("-" * 50)
    
    if HAS_JOBLIB:
        CONFIG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for target in CONFIG.TARGETS:
            # Recréer et sauvegarder le stacking
            estimators = get_base_estimators(logging.getLogger())
            stacking = create_stacking_model(estimators, logging.getLogger())
            y_train = train_df[target].values
            stacking.fit(X_train, y_train)
            
            path = CONFIG.MODELS_DIR / f"stacking_{target}.joblib"
            joblib.dump(stacking, path)
            logger.info(f"   💾 {path.name}")
    
    # 5. Résumé final
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    
    for target in CONFIG.TARGETS:
        best = best_models[target]
        baseline = BASELINE_MAE[target]
        improvement = baseline - best['results']['mae']
        
        logger.info(f"\n   🎯 {target}:")
        logger.info(f"      Meilleur: {best['name']}")
        logger.info(f"      MAE: {best['results']['mae']:.2f} nm")
        logger.info(f"      vs Baseline ({baseline:.2f} nm): {improvement:+.2f} nm")
    
    logger.info("")
    logger.info("📁 Fichiers générés:")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/ensemble_results.html")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/ensemble_comparison_*.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/ensemble_best_*.png")
    logger.info(f"   - {CONFIG.MODELS_DIR}/stacking_*.joblib")


if __name__ == "__main__":
    main()