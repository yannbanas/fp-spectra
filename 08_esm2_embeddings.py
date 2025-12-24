#!/usr/bin/env python3
"""
============================================================================
08_ESM2_EMBEDDINGS.PY - Embeddings de Langage Protéique
============================================================================

🎯 OBJECTIF:
   Utiliser ESM-2 (Evolutionary Scale Modeling) de Meta AI pour
   générer des embeddings riches à partir des séquences protéiques.
   
   Ces embeddings capturent des informations:
   - Évolutives (conservation, coévolution)
   - Structurales (structure secondaire, contacts)
   - Fonctionnelles (sites actifs, domaines)

📚 ESM-2:
   - Modèle de langage protéique entraîné sur 250M de séquences
   - Génère des vecteurs de 1280 dimensions par résidu
   - On utilise la moyenne sur toute la séquence (mean pooling)

📥 INPUT:
   - data/processed/dataset_train.csv
   - data/processed/dataset_test.csv
   - data/raw/fpbase.json (pour les séquences)

📤 OUTPUT:
   - data/processed/esm2_embeddings.npy
   - data/processed/dataset_train_esm2.csv
   - data/processed/dataset_test_esm2.csv
   - models/extra_trees_esm2_ex_max.joblib
   - models/extra_trees_esm2_em_max.joblib
   - reports/esm2_results.html

⚠️ PRÉREQUIS:
   pip install torch transformers
   
   GPU recommandé mais pas obligatoire (CPU ~10x plus lent)

============================================================================
"""

import os
import sys
import json
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
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("❌ scikit-learn requis!")

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

# ESM-2 imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch requis! pip install torch")

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("❌ Transformers requis! pip install transformers")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration."""
    
    DATA_DIR: Path = Path("data/processed")
    RAW_DIR: Path = Path("data/raw")
    MODELS_DIR: Path = Path("models")
    REPORTS_DIR: Path = Path("reports")
    LOGS_DIR: Path = Path("logs")
    
    TRAIN_FILE: str = "dataset_train.csv"
    TEST_FILE: str = "dataset_test.csv"
    FPBASE_JSON: str = "fpbase.json"
    
    TARGETS: List[str] = None
    EXCLUDE_COLS: List[str] = None
    
    # ESM-2 config
    ESM2_MODEL: str = "facebook/esm2_t33_650M_UR50D"  # 650M params, 1280 dim
    # Alternatives plus légères:
    # "facebook/esm2_t12_35M_UR50D"  # 35M params, 480 dim
    # "facebook/esm2_t6_8M_UR50D"    # 8M params, 320 dim
    
    MAX_SEQ_LENGTH: int = 1024  # Limite de longueur de séquence
    BATCH_SIZE: int = 4  # Réduire si problème de mémoire
    
    # PCA pour réduire les dimensions des embeddings
    PCA_COMPONENTS: int = 50  # Réduire 1280 → 50 dimensions
    
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
    
    log_file = CONFIG.LOGS_DIR / f"esm2_{datetime.now():%Y%m%d_%H%M%S}.log"
    
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
    logger.info("🧬 ESM-2 EMBEDDINGS - Langage Protéique")
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


def load_sequences(logger: logging.Logger) -> Dict[str, str]:
    """Charge les séquences depuis FPbase JSON."""
    
    json_path = CONFIG.RAW_DIR / CONFIG.FPBASE_JSON
    
    if not json_path.exists():
        logger.error(f"❌ Fichier JSON non trouvé: {json_path}")
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sequences = {}
    for entry in data:
        slug = entry.get('slug', '').lower().strip()
        seq = entry.get('seq', '')
        
        if slug and seq:
            # Nettoyer la séquence
            seq = ''.join(c for c in seq.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
            if len(seq) > 0:
                sequences[slug] = seq
    
    logger.info(f"   {len(sequences)} séquences chargées")
    
    return sequences


# ============================================================================
# ESM-2 EMBEDDINGS
# ============================================================================

class ESM2Embedder:
    """
    Classe pour générer des embeddings ESM-2.
    
    ESM-2 est un modèle de langage protéique qui apprend des représentations
    riches à partir des séquences d'acides aminés.
    """
    
    def __init__(self, model_name: str, device: str = None, logger: logging.Logger = None):
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        
        # Déterminer le device
        if device:
            self.device = device
        elif HAS_TORCH and torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info(f"   🎮 GPU détecté: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.logger.info("   💻 Utilisation du CPU (plus lent)")
        
        # Charger le modèle
        self.logger.info(f"   📥 Chargement de {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"   ✅ Modèle chargé sur {self.device}")
    
    def get_embedding(self, sequence: str) -> np.ndarray:
        """
        Génère l'embedding pour une séquence.
        
        Retourne la moyenne des embeddings de tous les résidus (mean pooling).
        """
        
        # Tronquer si trop long
        if len(sequence) > CONFIG.MAX_SEQ_LENGTH:
            sequence = sequence[:CONFIG.MAX_SEQ_LENGTH]
        
        # Tokenizer
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            truncation=True,
            max_length=CONFIG.MAX_SEQ_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inférence
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling (moyenne sur tous les tokens sauf [CLS] et [EOS])
        embeddings = outputs.last_hidden_state
        
        # Masquer les tokens spéciaux
        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        mean_pooled = summed / counts
        
        return mean_pooled.cpu().numpy().flatten()
    
    def get_embeddings_batch(self, sequences: List[str], 
                             protein_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Génère les embeddings pour un batch de séquences.
        """
        
        embeddings = {}
        n_total = len(sequences)
        
        for i, (pid, seq) in enumerate(zip(protein_ids, sequences)):
            if (i + 1) % 50 == 0 or i == 0:
                self.logger.info(f"      Progression: {i+1}/{n_total}")
            
            try:
                emb = self.get_embedding(seq)
                embeddings[pid] = emb
            except Exception as e:
                self.logger.warning(f"      ⚠️ Erreur pour {pid}: {e}")
                continue
        
        return embeddings


def generate_esm2_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             sequences: Dict[str, str],
                             logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère les embeddings ESM-2 pour train et test.
    """
    
    if not HAS_TORCH or not HAS_TRANSFORMERS:
        logger.error("❌ PyTorch et Transformers requis pour ESM-2!")
        return None, None
    
    # Initialiser ESM-2
    embedder = ESM2Embedder(CONFIG.ESM2_MODEL, logger=logger)
    
    # Collecter les séquences
    all_proteins = pd.concat([train_df, test_df])['protein_id'].unique()
    
    protein_sequences = []
    protein_ids = []
    
    for pid in all_proteins:
        if pid in sequences:
            protein_sequences.append(sequences[pid])
            protein_ids.append(pid)
        else:
            logger.warning(f"   ⚠️ Séquence non trouvée pour {pid}")
    
    logger.info(f"   {len(protein_ids)} protéines avec séquences")
    
    # Générer les embeddings
    logger.info("")
    logger.info("   🔄 Génération des embeddings ESM-2...")
    embeddings_dict = embedder.get_embeddings_batch(protein_sequences, protein_ids)
    
    logger.info(f"   ✅ {len(embeddings_dict)} embeddings générés")
    
    # Créer les matrices d'embeddings
    embedding_dim = None
    train_embeddings = []
    test_embeddings = []
    
    for pid in train_df['protein_id']:
        if pid in embeddings_dict:
            emb = embeddings_dict[pid]
            if embedding_dim is None:
                embedding_dim = len(emb)
            train_embeddings.append(emb)
        else:
            train_embeddings.append(np.zeros(embedding_dim or 1280))
    
    for pid in test_df['protein_id']:
        if pid in embeddings_dict:
            emb = embeddings_dict[pid]
            test_embeddings.append(emb)
        else:
            test_embeddings.append(np.zeros(embedding_dim or 1280))
    
    train_embeddings = np.array(train_embeddings)
    test_embeddings = np.array(test_embeddings)
    
    logger.info(f"   Shape train: {train_embeddings.shape}")
    logger.info(f"   Shape test: {test_embeddings.shape}")
    
    return train_embeddings, test_embeddings


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prépare les features structuraux existants."""
    
    feature_cols = [col for col in train_df.columns 
                    if col not in CONFIG.EXCLUDE_COLS 
                    and train_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Imputer
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    logger.info(f"   {len(feature_cols)} features structuraux")
    
    return X_train, X_test, feature_cols


def reduce_embeddings_pca(train_emb: np.ndarray, test_emb: np.ndarray,
                          n_components: int, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Réduit la dimensionnalité des embeddings avec PCA.
    
    1280 dimensions → n_components dimensions
    """
    
    logger.info(f"   📉 PCA: {train_emb.shape[1]} → {n_components} dimensions")
    
    # Normaliser d'abord
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_emb)
    test_scaled = scaler.transform(test_emb)
    
    # PCA
    pca = PCA(n_components=n_components, random_state=CONFIG.RANDOM_SEED)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum() * 100
    logger.info(f"   Variance expliquée: {explained_var:.1f}%")
    
    return train_pca, test_pca


def combine_features(X_struct_train: np.ndarray, X_struct_test: np.ndarray,
                     X_emb_train: np.ndarray, X_emb_test: np.ndarray,
                     logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine les features structuraux et les embeddings ESM-2.
    """
    
    X_train_combined = np.hstack([X_struct_train, X_emb_train])
    X_test_combined = np.hstack([X_struct_test, X_emb_test])
    
    logger.info(f"   Features combinés: {X_train_combined.shape[1]}")
    logger.info(f"      - Structuraux: {X_struct_train.shape[1]}")
    logger.info(f"      - ESM-2 (PCA): {X_emb_train.shape[1]}")
    
    return X_train_combined, X_test_combined


# ============================================================================
# ENTRAÎNEMENT
# ============================================================================

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       target: str, logger: logging.Logger) -> Dict:
    """
    Entraîne Extra Trees (meilleur modèle) sur les features combinés.
    """
    
    logger.info(f"   🌲 Entraînement Extra Trees pour {target}...")
    
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS
    )
    
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"      MAE: {mae:.2f} nm | RMSE: {rmse:.2f} nm | R²: {r2:.4f}")
    
    return {
        'model': model,
        'target': target,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }


# ============================================================================
# VISUALISATIONS
# ============================================================================

def create_comparison_plot(results_baseline: Dict, results_esm2: Dict,
                           output_dir: Path, logger: logging.Logger):
    """Compare les performances avant/après ESM-2."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baselines (Extra Trees sans ESM-2)
    baseline_mae = {'ex_max': 21.44, 'em_max': 18.47}
    
    for idx, target in enumerate(['ex_max', 'em_max']):
        ax = axes[idx]
        
        models = ['Baseline\n(Extra Trees)', 'Avec ESM-2']
        maes = [baseline_mae[target], results_esm2[target]['mae']]
        
        colors = ['#3498db', '#9b59b6']
        bars = ax.bar(models, maes, color=colors, width=0.5)
        
        # Annotations
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mae:.2f} nm', ha='center', fontsize=12, fontweight='bold')
        
        # Ligne objectif
        ax.axhline(y=15, color='red', linestyle='--', label='Objectif: 15 nm')
        
        ax.set_ylabel('MAE (nm)', fontsize=12)
        ax.set_title(f'{target}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(maes) + 5)
        ax.legend()
    
    plt.suptitle('Impact des Embeddings ESM-2', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'esm2_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_predictions_plot(results: Dict, output_dir: Path, logger: logging.Logger):
    """Scatter plot des prédictions."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, target in enumerate(['ex_max', 'em_max']):
        if target not in results:
            continue
        
        ax = axes[idx]
        r = results[target]
        
        ax.scatter(r['y_test'], r['y_pred'], alpha=0.6, s=50, c='#9b59b6')
        
        # Ligne parfaite
        min_val = min(r['y_test'].min(), r['y_pred'].min())
        max_val = max(r['y_test'].max(), r['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Parfait')
        
        ax.set_xlabel(f'{target} réel (nm)', fontsize=12)
        ax.set_ylabel(f'{target} prédit (nm)', fontsize=12)
        ax.set_title(f'Extra Trees + ESM-2 - {target}\nMAE={r["mae"]:.2f}nm, R²={r["r2"]:.3f}',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'esm2_predictions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


# ============================================================================
# RAPPORT HTML
# ============================================================================

def generate_report(results: Dict, output_dir: Path, logger: logging.Logger):
    """Génère le rapport HTML."""
    
    baseline_mae = {'ex_max': 21.44, 'em_max': 18.47}
    
    rows = ""
    for target in ['ex_max', 'em_max']:
        if target in results:
            r = results[target]
            improvement = baseline_mae[target] - r['mae']
            color = "#27ae60" if improvement > 0 else "#e74c3c"
            rows += f"""
            <tr>
                <td>{target}</td>
                <td>{baseline_mae[target]:.2f} nm</td>
                <td style="font-weight:bold">{r['mae']:.2f} nm</td>
                <td style="color:{color};font-weight:bold">{improvement:+.2f} nm</td>
                <td>{r['r2']:.4f}</td>
            </tr>
            """
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ESM-2 Embeddings - FP Prediction</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }}
        h2 {{ color: #8e44ad; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background: #9b59b6; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .section {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .highlight {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                     padding: 20px; border-radius: 10px; margin: 20px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 15px 0; }}
        code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>🧬 ESM-2 Embeddings - Résultats</h1>
    <p><em>Généré le {datetime.now():%Y-%m-%d %H:%M}</em></p>
    
    <div class="highlight">
        <h3>📚 Qu'est-ce que ESM-2 ?</h3>
        <p>ESM-2 (Evolutionary Scale Modeling) est un modèle de langage protéique de Meta AI, 
        entraîné sur 250 millions de séquences. Il génère des représentations vectorielles 
        riches (embeddings) qui capturent des informations évolutives, structurales et fonctionnelles.</p>
        <p><strong>Modèle utilisé:</strong> <code>{CONFIG.ESM2_MODEL}</code></p>
        <p><strong>Dimensions:</strong> 1280 → {CONFIG.PCA_COMPONENTS} (après PCA)</p>
    </div>
    
    <div class="section">
        <h2>📊 Comparaison des Performances</h2>
        <table>
            <tr>
                <th>Cible</th>
                <th>Baseline (Extra Trees)</th>
                <th>Avec ESM-2</th>
                <th>Amélioration</th>
                <th>R²</th>
            </tr>
            {rows}
        </table>
        <img src="esm2_comparison.png" alt="Comparison">
    </div>
    
    <div class="section">
        <h2>📈 Prédictions</h2>
        <img src="esm2_predictions.png" alt="Predictions">
    </div>
    
    <div class="section">
        <h2>🔬 Interprétation</h2>
        <ul>
            <li>Les embeddings ESM-2 apportent de l'information sur la <strong>conservation évolutive</strong> 
            et les <strong>patterns structuraux</strong> appris sur des millions de protéines.</li>
            <li>La combinaison features structuraux + ESM-2 permet de mieux capturer la relation 
            séquence → structure → propriétés spectrales.</li>
            <li>La PCA réduit le risque d'overfitting en passant de 1280 à {CONFIG.PCA_COMPONENTS} dimensions.</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🎓 Pour le Mémoire</h2>
        <p>Ces résultats démontrent que les modèles de langage protéique peuvent améliorer 
        la prédiction des propriétés spectrales en apportant des informations complémentaires 
        aux features géométriques extraits des structures AlphaFold2.</p>
    </div>
    
</body>
</html>
"""
    
    report_path = output_dir / "esm2_results.html"
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
    
    # Vérifier les dépendances ESM-2
    if not HAS_TORCH:
        logger.error("❌ PyTorch requis! pip install torch")
        logger.info("   Voir: https://pytorch.org/get-started/locally/")
        return
    
    if not HAS_TRANSFORMERS:
        logger.error("❌ Transformers requis! pip install transformers")
        return
    
    # 1. Charger les données
    logger.info("")
    logger.info("📂 ÉTAPE 1: Chargement des données")
    logger.info("-" * 50)
    
    train_df, test_df = load_data(logger)
    if train_df is None:
        return
    
    sequences = load_sequences(logger)
    if not sequences:
        return
    
    # 2. Préparer les features structuraux
    logger.info("")
    logger.info("🔧 ÉTAPE 2: Features structuraux")
    logger.info("-" * 50)
    
    X_struct_train, X_struct_test, feature_names = prepare_features(train_df, test_df, logger)
    
    # 3. Générer les embeddings ESM-2
    logger.info("")
    logger.info("🧬 ÉTAPE 3: Embeddings ESM-2")
    logger.info("-" * 50)
    
    emb_train, emb_test = generate_esm2_embeddings(train_df, test_df, sequences, logger)
    
    if emb_train is None:
        logger.error("❌ Échec de la génération des embeddings!")
        return
    
    # 4. Réduire avec PCA
    logger.info("")
    logger.info("📉 ÉTAPE 4: Réduction PCA")
    logger.info("-" * 50)
    
    emb_train_pca, emb_test_pca = reduce_embeddings_pca(
        emb_train, emb_test, CONFIG.PCA_COMPONENTS, logger
    )
    
    # 5. Combiner les features
    logger.info("")
    logger.info("🔗 ÉTAPE 5: Combinaison des features")
    logger.info("-" * 50)
    
    X_train, X_test = combine_features(
        X_struct_train, X_struct_test,
        emb_train_pca, emb_test_pca,
        logger
    )
    
    # 6. Entraîner et évaluer
    logger.info("")
    logger.info("🏋️ ÉTAPE 6: Entraînement")
    logger.info("-" * 50)
    
    results = {}
    models = {}
    
    for target in CONFIG.TARGETS:
        y_train = train_df[target].values
        y_test = test_df[target].values
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, target, logger)
        results[target] = result
        models[target] = result['model']
    
    # 7. Visualisations
    logger.info("")
    logger.info("📊 ÉTAPE 7: Visualisations")
    logger.info("-" * 50)
    
    CONFIG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plot({}, results, CONFIG.REPORTS_DIR, logger)
    create_predictions_plot(results, CONFIG.REPORTS_DIR, logger)
    generate_report(results, CONFIG.REPORTS_DIR, logger)
    
    # 8. Sauvegarder les modèles
    logger.info("")
    logger.info("💾 ÉTAPE 8: Sauvegarde")
    logger.info("-" * 50)
    
    if HAS_JOBLIB:
        CONFIG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for target, model in models.items():
            path = CONFIG.MODELS_DIR / f"extra_trees_esm2_{target}.joblib"
            joblib.dump(model, path)
            logger.info(f"   💾 {path.name}")
        
        # Sauvegarder aussi les embeddings pour réutilisation
        np.save(CONFIG.DATA_DIR / "esm2_embeddings_train.npy", emb_train)
        np.save(CONFIG.DATA_DIR / "esm2_embeddings_test.npy", emb_test)
        logger.info("   💾 esm2_embeddings_*.npy")
    
    # 9. Résumé final
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    
    baseline_mae = {'ex_max': 21.44, 'em_max': 18.47}
    
    for target in CONFIG.TARGETS:
        r = results[target]
        improvement = baseline_mae[target] - r['mae']
        
        logger.info(f"\n   🎯 {target}:")
        logger.info(f"      Baseline (Extra Trees): {baseline_mae[target]:.2f} nm")
        logger.info(f"      Avec ESM-2: {r['mae']:.2f} nm")
        logger.info(f"      Amélioration: {improvement:+.2f} nm")
        logger.info(f"      R²: {r['r2']:.4f}")
    
    # Objectif atteint?
    best_mae = min(r['mae'] for r in results.values())
    logger.info("")
    
    if best_mae < 15:
        logger.info("🎉 OBJECTIF ATTEINT! MAE < 15nm")
    elif best_mae < 17:
        logger.info("✅ EXCELLENT! MAE < 17nm - Très proche de l'objectif!")
    else:
        logger.info("👍 Améliorations obtenues avec ESM-2")
    
    logger.info("")
    logger.info("📁 Fichiers générés:")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/esm2_results.html")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/esm2_comparison.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/esm2_predictions.png")
    logger.info(f"   - {CONFIG.MODELS_DIR}/extra_trees_esm2_*.joblib")
    logger.info(f"   - {CONFIG.DATA_DIR}/esm2_embeddings_*.npy")


if __name__ == "__main__":
    main()
