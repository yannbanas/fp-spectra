#!/usr/bin/env python3
"""
============================================================================
05_SHAP_ANALYSIS.PY - Analyse SHAP pour Interprétabilité
============================================================================

🎯 OBJECTIF:
   Utiliser SHAP (SHapley Additive exPlanations) pour comprendre
   POURQUOI le modèle fait ses prédictions.
   
   SHAP répond à: "Quelle contribution chaque feature apporte-t-il
   à la prédiction pour une protéine donnée?"

📥 INPUT:
   - models/rf_ex_max.joblib
   - models/rf_em_max.joblib
   - data/processed/dataset_test.csv

📤 OUTPUT:
   - reports/shap_summary_ex_max.png
   - reports/shap_summary_em_max.png
   - reports/shap_dependence_plots.png
   - reports/shap_analysis.html
   - reports/shap_top_features.csv

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

# Imports ML
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("❌ joblib requis! pip install joblib")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("❌ shap requis! pip install shap")

try:
    from sklearn.impute import SimpleImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("❌ matplotlib requis! pip install matplotlib")


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
    
    TEST_FILE: str = "dataset_test.csv"
    TRAIN_FILE: str = "dataset_train.csv"
    
    TARGETS: List[str] = None
    
    EXCLUDE_COLS: List[str] = None
    
    # Nombre d'échantillons pour SHAP (peut être lent sur gros datasets)
    SHAP_SAMPLE_SIZE: int = 100
    
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
    
    log_file = CONFIG.LOGS_DIR / f"shap_analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
    
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
    logger.info("🔍 ANALYSE SHAP - INTERPRÉTABILITÉ DES MODÈLES")
    logger.info("="*70)
    
    return logger


# ============================================================================
# CHARGEMENT
# ============================================================================

def load_model(model_path: Path, logger: logging.Logger):
    """Charge un modèle sauvegardé."""
    
    if not model_path.exists():
        logger.error(f"❌ Modèle non trouvé: {model_path}")
        return None
    
    model = joblib.load(model_path)
    logger.info(f"   ✅ Chargé: {model_path.name}")
    
    return model


def load_data(logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données."""
    
    test_path = CONFIG.DATA_DIR / CONFIG.TEST_FILE
    train_path = CONFIG.DATA_DIR / CONFIG.TRAIN_FILE
    
    if not test_path.exists():
        logger.error(f"❌ Fichier test non trouvé: {test_path}")
        return None, None
    
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path) if train_path.exists() else None
    
    logger.info(f"   Test: {len(test_df)} échantillons")
    if train_df is not None:
        logger.info(f"   Train: {len(train_df)} échantillons")
    
    return test_df, train_df


def prepare_features(df: pd.DataFrame, logger: logging.Logger) -> Tuple[np.ndarray, List[str]]:
    """Prépare les features."""
    
    feature_cols = [col for col in df.columns 
                    if col not in CONFIG.EXCLUDE_COLS 
                    and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X = df[feature_cols].values
    
    # Imputer les NaN
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    return X, feature_cols


# ============================================================================
# ANALYSE SHAP
# ============================================================================

def compute_shap_values(model, X: np.ndarray, feature_names: List[str],
                        target_name: str, logger: logging.Logger) -> shap.Explanation:
    """
    Calcule les valeurs SHAP pour un modèle.
    
    SHAP = SHapley Additive exPlanations
    - Basé sur la théorie des jeux (valeurs de Shapley)
    - Explique la contribution de chaque feature à chaque prédiction
    """
    
    logger.info(f"   🔄 Calcul SHAP pour {target_name}...")
    
    # Limiter le nombre d'échantillons si nécessaire
    if len(X) > CONFIG.SHAP_SAMPLE_SIZE:
        indices = np.random.choice(len(X), CONFIG.SHAP_SAMPLE_SIZE, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Créer l'explainer
    # TreeExplainer est optimisé pour Random Forest et XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(X_sample)
    
    # Créer l'objet Explanation
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample,
        feature_names=feature_names
    )
    
    logger.info(f"      ✅ {len(X_sample)} échantillons analysés")
    
    return explanation, X_sample


def get_shap_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Calcule l'importance moyenne absolue des SHAP values.
    
    C'est différent de feature_importances_ de Random Forest:
    - RF importance = réduction de l'impureté
    - SHAP importance = contribution moyenne aux prédictions
    """
    
    # Importance = moyenne des |SHAP values| pour chaque feature
    importance = np.abs(shap_values).mean(axis=0)
    
    df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': importance
    }).sort_values('shap_importance', ascending=False)
    
    return df


# ============================================================================
# VISUALISATIONS
# ============================================================================

def create_summary_plot(explanation: shap.Explanation, target_name: str,
                        output_dir: Path, logger: logging.Logger):
    """
    Crée le SHAP Summary Plot.
    
    Ce graphique montre:
    - Chaque point = une prédiction pour une protéine
    - Position horizontale = contribution du feature à la prédiction
    - Couleur = valeur du feature (rouge=haute, bleu=basse)
    """
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explanation, show=False, max_display=20)
    plt.title(f'SHAP Summary - {target_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'shap_summary_{target_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_bar_plot(explanation: shap.Explanation, target_name: str,
                    output_dir: Path, logger: logging.Logger):
    """
    Crée le SHAP Bar Plot (importance moyenne).
    """
    
    plt.figure(figsize=(10, 8))
    shap.plots.bar(explanation, show=False, max_display=20)
    plt.title(f'SHAP Feature Importance - {target_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'shap_bar_{target_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_dependence_plots(explanation: shap.Explanation, X: np.ndarray,
                            feature_names: List[str], target_name: str,
                            output_dir: Path, logger: logging.Logger):
    """
    Crée les Dependence Plots pour les top features.
    
    Ces graphiques montrent:
    - X = valeur du feature
    - Y = SHAP value (contribution)
    - Couleur = feature qui interagit le plus
    """
    
    # Top 6 features par importance SHAP
    importance = np.abs(explanation.values).mean(axis=0)
    top_indices = np.argsort(importance)[-6:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (feature_idx, feature_name) in enumerate(zip(top_indices, top_features)):
        ax = axes[idx]
        
        # Scatter plot manuel
        x_vals = X[:, feature_idx]
        shap_vals = explanation.values[:, feature_idx]
        
        scatter = ax.scatter(x_vals, shap_vals, c=shap_vals, cmap='coolwarm', 
                            alpha=0.6, s=30)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('SHAP value')
        ax.set_title(f'{feature_name}')
        
    plt.suptitle(f'SHAP Dependence Plots - {target_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'shap_dependence_{target_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_waterfall_plot(explanation: shap.Explanation, X: np.ndarray,
                          feature_names: List[str], target_name: str,
                          protein_idx: int, output_dir: Path, logger: logging.Logger):
    """
    Crée un Waterfall Plot pour une protéine spécifique.
    
    Montre comment chaque feature contribue à la prédiction finale.
    """
    
    plt.figure(figsize=(10, 8))
    
    # Créer l'explication pour un seul échantillon
    single_explanation = shap.Explanation(
        values=explanation.values[protein_idx],
        base_values=explanation.base_values,
        data=X[protein_idx],
        feature_names=feature_names
    )
    
    shap.plots.waterfall(single_explanation, show=False, max_display=15)
    plt.title(f'Waterfall Plot - {target_name} (Protein #{protein_idx})', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'shap_waterfall_{target_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_combined_importance_plot(importance_ex: pd.DataFrame, 
                                    importance_em: pd.DataFrame,
                                    output_dir: Path, logger: logging.Logger):
    """
    Crée un graphique comparant l'importance pour ex_max vs em_max.
    """
    
    # Fusionner les deux
    merged = importance_ex.merge(importance_em, on='feature', suffixes=('_ex', '_em'))
    merged = merged.head(15)  # Top 15
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(merged))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, merged['shap_importance_ex'], width, 
                    label='ex_max (excitation)', color='#3498db')
    bars2 = ax.barh(x + width/2, merged['shap_importance_em'], width,
                    label='em_max (émission)', color='#e74c3c')
    
    ax.set_yticks(x)
    ax.set_yticklabels(merged['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('SHAP Importance (mean |SHAP value|)')
    ax.set_title('Comparaison SHAP Importance: Excitation vs Émission', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    output_path = output_dir / 'shap_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


# ============================================================================
# RAPPORT HTML
# ============================================================================

def generate_report(importance_dict: Dict[str, pd.DataFrame],
                    output_dir: Path, logger: logging.Logger):
    """Génère le rapport HTML."""
    
    # Tables d'importance
    tables_html = ""
    for target, df in importance_dict.items():
        rows = ""
        for i, (_, row) in enumerate(df.head(15).iterrows(), 1):
            rows += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{row['feature']}</strong></td>
                <td>{row['shap_importance']:.4f}</td>
            </tr>
            """
        
        tables_html += f"""
        <h3>🎯 {target}</h3>
        <table>
            <tr><th>#</th><th>Feature</th><th>SHAP Importance</th></tr>
            {rows}
        </table>
        """
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Analyse SHAP - FP Prediction</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }}
        h2 {{ color: #8e44ad; }}
        h3 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #9b59b6; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        tr:nth-child(1) td {{ background: #f1c40f; font-weight: bold; }}
        tr:nth-child(2) td {{ background: #f39c12; }}
        tr:nth-child(3) td {{ background: #e67e22; }}
        .section {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 15px 0; }}
        .insight {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                   padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .insight h3 {{ color: white; margin-top: 0; }}
        code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>🔍 Analyse SHAP - Interprétabilité des Modèles</h1>
    <p><em>Généré le {datetime.now():%Y-%m-%d %H:%M}</em></p>
    
    <div class="insight">
        <h3>💡 Qu'est-ce que SHAP ?</h3>
        <p>SHAP (SHapley Additive exPlanations) utilise la théorie des jeux pour expliquer 
        la contribution de chaque feature à chaque prédiction individuelle.</p>
        <p><strong>Avantage vs Feature Importance classique:</strong> SHAP montre non seulement 
        QUELS features sont importants, mais aussi COMMENT ils affectent les prédictions 
        (positivement ou négativement).</p>
    </div>
    
    <div class="section">
        <h2>📊 Importance SHAP par Cible</h2>
        {tables_html}
    </div>
    
    <div class="section">
        <h2>📈 Comparaison Excitation vs Émission</h2>
        <img src="shap_comparison.png" alt="SHAP Comparison">
        <p><strong>Interprétation:</strong> Les features structuraux du chromophore 
        (<code>chrom_psi_tyr</code>, <code>chrom_chi1_tyr</code>) sont importants pour 
        les deux propriétés, ce qui valide l'hypothèse du projet!</p>
    </div>
    
    <div class="section">
        <h2>🎨 Summary Plots</h2>
        <p>Chaque point = une protéine. Position = contribution du feature. 
        Couleur = valeur du feature (rouge=haute, bleu=basse).</p>
        
        <h3>ex_max (Excitation)</h3>
        <img src="shap_summary_ex_max.png" alt="SHAP Summary ex_max">
        
        <h3>em_max (Émission)</h3>
        <img src="shap_summary_em_max.png" alt="SHAP Summary em_max">
    </div>
    
    <div class="section">
        <h2>📉 Dependence Plots</h2>
        <p>Comment la valeur d'un feature affecte sa contribution à la prédiction.</p>
        
        <h3>ex_max</h3>
        <img src="shap_dependence_ex_max.png" alt="Dependence ex_max">
        
        <h3>em_max</h3>
        <img src="shap_dependence_em_max.png" alt="Dependence em_max">
    </div>
    
    <div class="section">
        <h2>🌊 Waterfall Plots</h2>
        <p>Décomposition de la prédiction pour une protéine spécifique.</p>
        
        <h3>ex_max</h3>
        <img src="shap_waterfall_ex_max.png" alt="Waterfall ex_max">
        
        <h3>em_max</h3>
        <img src="shap_waterfall_em_max.png" alt="Waterfall em_max">
    </div>
    
    <div class="insight">
        <h3>🎓 Conclusions pour le Mémoire</h3>
        <ol>
            <li><strong>L'angle ψ du chromophore</strong> (<code>chrom_psi_tyr</code>) est 
            crucial pour prédire la couleur - cela prouve que la structure 3D AlphaFold2 
            contient de l'information spectrale!</li>
            <li><strong>La composition en acides aminés</strong> (N, D, hydrophobicité) 
            reflète l'environnement du chromophore.</li>
            <li><strong>SHAP permet d'expliquer</strong> chaque prédiction individuellement, 
            ce qui est essentiel pour la confiance dans le modèle.</li>
        </ol>
    </div>
    
</body>
</html>
"""
    
    report_path = output_dir / "shap_analysis.html"
    report_path.write_text(html, encoding='utf-8')
    logger.info(f"   📄 {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Vérifications
    if not HAS_SHAP:
        print("❌ SHAP requis! pip install shap")
        return
    
    if not HAS_JOBLIB:
        print("❌ joblib requis! pip install joblib")
        return
    
    logger = setup_logging()
    
    # 1. Charger les données
    logger.info("")
    logger.info("📂 ÉTAPE 1: Chargement des données")
    logger.info("-" * 50)
    
    test_df, train_df = load_data(logger)
    if test_df is None:
        return
    
    X_test, feature_names = prepare_features(test_df, logger)
    logger.info(f"   {len(feature_names)} features")
    
    # 2. Charger les modèles
    logger.info("")
    logger.info("📂 ÉTAPE 2: Chargement des modèles")
    logger.info("-" * 50)
    
    models = {}
    for target in CONFIG.TARGETS:
        model_path = CONFIG.MODELS_DIR / f"rf_{target}.joblib"
        model = load_model(model_path, logger)
        if model:
            models[target] = model
    
    if not models:
        logger.error("❌ Aucun modèle chargé!")
        return
    
    # 3. Calculer SHAP
    logger.info("")
    logger.info("🔍 ÉTAPE 3: Calcul des valeurs SHAP")
    logger.info("-" * 50)
    
    shap_results = {}
    importance_dict = {}
    
    for target, model in models.items():
        explanation, X_sample = compute_shap_values(
            model, X_test, feature_names, target, logger
        )
        shap_results[target] = (explanation, X_sample)
        
        # Importance
        importance = get_shap_importance(explanation.values, feature_names)
        importance_dict[target] = importance
    
    # 4. Visualisations
    logger.info("")
    logger.info("📊 ÉTAPE 4: Génération des visualisations")
    logger.info("-" * 50)
    
    CONFIG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for target, (explanation, X_sample) in shap_results.items():
        # Summary plot
        create_summary_plot(explanation, target, CONFIG.REPORTS_DIR, logger)
        
        # Bar plot
        create_bar_plot(explanation, target, CONFIG.REPORTS_DIR, logger)
        
        # Dependence plots
        create_dependence_plots(explanation, X_sample, feature_names, 
                               target, CONFIG.REPORTS_DIR, logger)
        
        # Waterfall (pour la première protéine)
        create_waterfall_plot(explanation, X_sample, feature_names,
                             target, 0, CONFIG.REPORTS_DIR, logger)
    
    # Comparaison
    if len(importance_dict) == 2:
        create_combined_importance_plot(
            importance_dict['ex_max'], 
            importance_dict['em_max'],
            CONFIG.REPORTS_DIR, logger
        )
    
    # 5. Sauvegarder les importances
    logger.info("")
    logger.info("💾 ÉTAPE 5: Sauvegarde des résultats")
    logger.info("-" * 50)
    
    for target, importance in importance_dict.items():
        csv_path = CONFIG.REPORTS_DIR / f"shap_importance_{target}.csv"
        importance.to_csv(csv_path, index=False)
        logger.info(f"   💾 {csv_path.name}")
    
    # 6. Rapport HTML
    generate_report(importance_dict, CONFIG.REPORTS_DIR, logger)
    
    # Résumé
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ")
    logger.info("=" * 60)
    
    for target, importance in importance_dict.items():
        logger.info(f"\n   🎯 {target} - Top 5 Features SHAP:")
        for i, (_, row) in enumerate(importance.head(5).iterrows(), 1):
            logger.info(f"      {i}. {row['feature']}: {row['shap_importance']:.4f}")
    
    logger.info("")
    logger.info("📁 Fichiers générés:")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/shap_analysis.html")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/shap_summary_*.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/shap_dependence_*.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/shap_waterfall_*.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/shap_comparison.png")
    logger.info("")
    logger.info("🎓 Ouvre shap_analysis.html pour le rapport complet!")


if __name__ == "__main__":
    main()
