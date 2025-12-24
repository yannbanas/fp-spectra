#!/usr/bin/env python3
"""
============================================================================
04_TRAIN_BASELINE.PY - Entraînement Random Forest & XGBoost
============================================================================

🎯 OBJECTIF:
   Entraîner des modèles ML pour prédire les propriétés spectrales
   des protéines fluorescentes à partir des features structuraux.

📥 INPUT:
   - data/processed/dataset_train.csv
   - data/processed/dataset_test.csv

📤 OUTPUT:
   - models/rf_ex_max.joblib (Random Forest pour excitation)
   - models/rf_em_max.joblib (Random Forest pour émission)
   - models/xgb_ex_max.joblib (XGBoost pour excitation)
   - models/xgb_em_max.joblib (XGBoost pour émission)
   - reports/training_results.html

🎯 CIBLES:
   - ex_max: longueur d'onde d'excitation (nm)
   - em_max: longueur d'onde d'émission (nm)
   - qy: rendement quantique (optionnel)

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

# Suppression warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("❌ scikit-learn non installé! pip install scikit-learn")

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost non installé (optionnel). pip install xgboost")

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
    
    # Fichiers
    TRAIN_FILE: str = "dataset_train.csv"
    TEST_FILE: str = "dataset_test.csv"
    
    # Cibles à prédire
    TARGETS: List[str] = None
    
    # Colonnes à exclure des features
    EXCLUDE_COLS: List[str] = None
    
    # Paramètres Random Forest
    RF_PARAMS: Dict = None
    
    # Paramètres XGBoost
    XGB_PARAMS: Dict = None
    
    RANDOM_SEED: int = 42
    
    def __post_init__(self):
        self.TARGETS = ['ex_max', 'em_max']
        self.EXCLUDE_COLS = [
            'protein_id', 'name', 'ex_max', 'em_max', 'qy', 
            'stokes_shift', 'ext_coeff', 'brightness'
        ]
        self.RF_PARAMS = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.RANDOM_SEED,
            'n_jobs': -1
        }
        self.XGB_PARAMS = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.RANDOM_SEED,
            'n_jobs': -1
        }


CONFIG = Config()


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = CONFIG.LOGS_DIR / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    
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
    logger.info("🌲 ENTRAÎNEMENT MODÈLES BASELINE - RF & XGBoost")
    logger.info("="*70)
    
    return logger


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def load_data(logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les datasets train et test."""
    
    train_path = CONFIG.DATA_DIR / CONFIG.TRAIN_FILE
    test_path = CONFIG.DATA_DIR / CONFIG.TEST_FILE
    
    if not train_path.exists():
        logger.error(f"❌ Fichier train non trouvé: {train_path}")
        return None, None
    
    if not test_path.exists():
        logger.error(f"❌ Fichier test non trouvé: {test_path}")
        return None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"📂 Train: {len(train_df)} échantillons")
    logger.info(f"📂 Test: {len(test_df)} échantillons")
    
    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prépare les features pour l'entraînement.
    
    - Sélectionne les colonnes numériques
    - Impute les valeurs manquantes
    - Normalise (optionnel pour RF/XGB mais bon pour la stabilité)
    """
    
    # Identifier les colonnes de features
    feature_cols = [col for col in train_df.columns 
                    if col not in CONFIG.EXCLUDE_COLS 
                    and train_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    logger.info(f"   {len(feature_cols)} features sélectionnés")
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Imputer les NaN avec la médiane
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Vérifier qu'il n'y a plus de NaN
    n_nan_train = np.isnan(X_train).sum()
    n_nan_test = np.isnan(X_test).sum()
    
    if n_nan_train > 0 or n_nan_test > 0:
        logger.warning(f"   ⚠️ NaN restants: train={n_nan_train}, test={n_nan_test}")
    
    logger.info(f"   Shape train: {X_train.shape}")
    logger.info(f"   Shape test: {X_test.shape}")
    
    return X_train, X_test, feature_cols


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        target_name: str, logger: logging.Logger) -> RandomForestRegressor:
    """Entraîne un Random Forest."""
    
    logger.info(f"   🌲 Entraînement Random Forest pour {target_name}...")
    
    rf = RandomForestRegressor(**CONFIG.RF_PARAMS)
    rf.fit(X_train, y_train)
    
    # Score sur train (pour vérifier overfitting)
    train_score = rf.score(X_train, y_train)
    logger.info(f"      R² train: {train_score:.4f}")
    
    return rf


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  target_name: str, logger: logging.Logger) -> Optional['XGBRegressor']:
    """Entraîne un XGBoost."""
    
    if not HAS_XGBOOST:
        return None
    
    logger.info(f"   🚀 Entraînement XGBoost pour {target_name}...")
    
    xgb = XGBRegressor(**CONFIG.XGB_PARAMS)
    xgb.fit(X_train, y_train, verbose=False)
    
    train_score = xgb.score(X_train, y_train)
    logger.info(f"      R² train: {train_score:.4f}")
    
    return xgb


# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str, target_name: str, logger: logging.Logger) -> Dict:
    """Évalue un modèle sur le test set."""
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Pourcentage d'erreur relative moyenne
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results = {
        'model': model_name,
        'target': target_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    logger.info(f"      MAE: {mae:.2f} nm | RMSE: {rmse:.2f} nm | R²: {r2:.4f}")
    
    return results


def get_feature_importance(model, feature_names: List[str], 
                           model_name: str, top_n: int = 15) -> pd.DataFrame:
    """Extrait les features les plus importants."""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df.head(top_n)


# ============================================================================
# VISUALISATIONS
# ============================================================================

def create_plots(all_results: List[Dict], feature_importances: Dict,
                 output_dir: Path, logger: logging.Logger):
    """Crée les visualisations."""
    
    if not HAS_MATPLOTLIB:
        logger.warning("⚠️ Matplotlib non disponible, pas de graphiques")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scatter plots: Prédit vs Réel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    plot_idx = 0
    for result in all_results:
        if plot_idx >= 4:
            break
        
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        y_test = result['y_test']
        y_pred = result['y_pred']
        
        ax.scatter(y_test, y_pred, alpha=0.5, s=30)
        
        # Ligne diagonale (prédiction parfaite)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parfait')
        
        ax.set_xlabel(f'{result["target"]} réel (nm)')
        ax.set_ylabel(f'{result["target"]} prédit (nm)')
        ax.set_title(f'{result["model"]} - {result["target"]}\nMAE={result["mae"]:.1f}nm, R²={result["r2"]:.3f}')
        ax.legend()
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 predictions_scatter.png")
    
    # 2. Feature importance (pour le meilleur modèle)
    if feature_importances:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (target, fi_df) in enumerate(feature_importances.items()):
            if idx >= 2:
                break
            ax = axes[idx]
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
            ax.barh(fi_df['feature'], fi_df['importance'], color=colors)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top Features pour {target}')
            ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   📊 feature_importance.png")


# ============================================================================
# RAPPORT HTML
# ============================================================================

def generate_report(all_results: List[Dict], feature_importances: Dict,
                    output_dir: Path, logger: logging.Logger):
    """Génère un rapport HTML."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tableau des résultats
    results_rows = ""
    for r in all_results:
        color = "#27ae60" if r['mae'] < 20 else "#f39c12" if r['mae'] < 30 else "#e74c3c"
        results_rows += f"""
        <tr>
            <td>{r['model']}</td>
            <td>{r['target']}</td>
            <td style="color:{color};font-weight:bold">{r['mae']:.2f} nm</td>
            <td>{r['rmse']:.2f} nm</td>
            <td>{r['r2']:.4f}</td>
            <td>{r['mape']:.1f}%</td>
        </tr>
        """
    
    # Feature importance HTML
    fi_html = ""
    for target, fi_df in feature_importances.items():
        fi_html += f"<h3>Top features pour {target}</h3><ol>"
        for _, row in fi_df.iterrows():
            fi_html += f"<li><strong>{row['feature']}</strong>: {row['importance']:.4f}</li>"
        fi_html += "</ol>"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Résultats Entraînement - FP Prediction</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .section {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .metric-card {{ display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
                       color: white; padding: 20px 30px; border-radius: 10px; margin: 10px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }}
        .good {{ color: #27ae60; }}
        .medium {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>🧬 Résultats Entraînement - Prédiction FP</h1>
    <p><em>Généré le {datetime.now():%Y-%m-%d %H:%M}</em></p>
    
    <div class="section">
        <h2>📊 Performances des Modèles</h2>
        <table>
            <tr>
                <th>Modèle</th>
                <th>Cible</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
                <th>MAPE</th>
            </tr>
            {results_rows}
        </table>
        
        <p><strong>Interprétation MAE:</strong></p>
        <ul>
            <li><span class="good">< 20 nm</span> = Excellent</li>
            <li><span class="medium">20-30 nm</span> = Bon</li>
            <li><span class="bad">> 30 nm</span> = À améliorer</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📈 Prédictions vs Réalité</h2>
        <img src="predictions_scatter.png" alt="Scatter plots">
    </div>
    
    <div class="section">
        <h2>🔍 Features Importants</h2>
        <img src="feature_importance.png" alt="Feature importance">
        {fi_html}
    </div>
    
    <div class="section">
        <h2>🚀 Prochaines Étapes</h2>
        <ol>
            <li><strong>06_feature_importance.py</strong> - Analyse SHAP détaillée</li>
            <li><strong>07_hyperparameter_tuning.py</strong> - Optimisation des hyperparamètres</li>
            <li><strong>08_ensemble.py</strong> - Modèle ensemble (stacking)</li>
        </ol>
    </div>
    
</body>
</html>
"""
    
    report_path = output_dir / "training_results.html"
    report_path.write_text(html, encoding='utf-8')
    logger.info(f"   📄 {report_path}")


# ============================================================================
# SAUVEGARDE DES MODÈLES
# ============================================================================

def save_models(models: Dict, output_dir: Path, logger: logging.Logger):
    """Sauvegarde les modèles entraînés."""
    
    if not HAS_JOBLIB:
        logger.warning("⚠️ joblib non disponible, modèles non sauvegardés")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        if model is not None:
            path = output_dir / f"{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"   💾 {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not HAS_SKLEARN:
        print("❌ scikit-learn requis! pip install scikit-learn")
        return
    
    logger = setup_logging()
    
    # 1. Charger les données
    logger.info("")
    logger.info("📂 ÉTAPE 1: Chargement des données")
    logger.info("-" * 50)
    
    train_df, test_df = load_data(logger)
    if train_df is None:
        return
    
    # 2. Préparer les features
    logger.info("")
    logger.info("🔧 ÉTAPE 2: Préparation des features")
    logger.info("-" * 50)
    
    X_train, X_test, feature_names = prepare_features(train_df, test_df, logger)
    
    # 3. Entraîner et évaluer pour chaque cible
    all_results = []
    all_models = {}
    feature_importances = {}
    
    for target in CONFIG.TARGETS:
        logger.info("")
        logger.info(f"🎯 ÉTAPE 3: Entraînement pour {target}")
        logger.info("-" * 50)
        
        # Extraire les cibles
        y_train = train_df[target].values
        y_test = test_df[target].values
        
        logger.info(f"   {target}: train={len(y_train)}, test={len(y_test)}")
        logger.info(f"   Range: {y_train.min():.0f} - {y_train.max():.0f} nm")
        
        # Random Forest
        rf = train_random_forest(X_train, y_train, target, logger)
        rf_results = evaluate_model(rf, X_test, y_test, "Random Forest", target, logger)
        all_results.append(rf_results)
        all_models[f"rf_{target}"] = rf
        
        # Feature importance pour RF
        fi = get_feature_importance(rf, feature_names, "RF", top_n=15)
        feature_importances[target] = fi
        
        # XGBoost
        if HAS_XGBOOST:
            xgb = train_xgboost(X_train, y_train, target, logger)
            if xgb:
                xgb_results = evaluate_model(xgb, X_test, y_test, "XGBoost", target, logger)
                all_results.append(xgb_results)
                all_models[f"xgb_{target}"] = xgb
    
    # 4. Visualisations
    logger.info("")
    logger.info("📊 ÉTAPE 4: Génération des visualisations")
    logger.info("-" * 50)
    
    create_plots(all_results, feature_importances, CONFIG.REPORTS_DIR, logger)
    generate_report(all_results, feature_importances, CONFIG.REPORTS_DIR, logger)
    
    # 5. Sauvegarder les modèles
    logger.info("")
    logger.info("💾 ÉTAPE 5: Sauvegarde des modèles")
    logger.info("-" * 50)
    
    save_models(all_models, CONFIG.MODELS_DIR, logger)
    
    # 6. Résumé final
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    
    # Meilleurs résultats par cible
    for target in CONFIG.TARGETS:
        target_results = [r for r in all_results if r['target'] == target]
        if target_results:
            best = min(target_results, key=lambda x: x['mae'])
            logger.info(f"   {target}:")
            logger.info(f"      Meilleur modèle: {best['model']}")
            logger.info(f"      MAE: {best['mae']:.2f} nm")
            logger.info(f"      R²: {best['r2']:.4f}")
    
    logger.info("")
    logger.info("📁 Fichiers générés:")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/training_results.html")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/predictions_scatter.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/feature_importance.png")
    logger.info(f"   - {CONFIG.MODELS_DIR}/*.joblib")
    
    logger.info("")
    
    # Interprétation
    best_mae = min(r['mae'] for r in all_results)
    if best_mae < 15:
        logger.info("🎉 EXCELLENT! MAE < 15nm - Objectif atteint!")
    elif best_mae < 20:
        logger.info("✅ TRÈS BIEN! MAE < 20nm - Bon résultat!")
    elif best_mae < 30:
        logger.info("👍 CORRECT! MAE < 30nm - Peut être amélioré")
    else:
        logger.info("⚠️ À AMÉLIORER - Considère plus de features ou autre architecture")


if __name__ == "__main__":
    main()
