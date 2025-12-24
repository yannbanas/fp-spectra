#!/usr/bin/env python3
"""
============================================================================
09_DEEP_LEARNING_GNN.PY - Graph Neural Networks & Multi-Task Learning
============================================================================

🎯 OBJECTIF:
   Explorer des approches Deep Learning pour améliorer les prédictions:
   1. Multi-task learning (prédire ex_max, em_max, QY simultanément)
   2. Graph Neural Network (GNN) sur la structure 3D

📚 GNN PRINCIPE:
   - Chaque résidu = un noeud
   - Features noeud = type AA + pLDDT + position relative au chromophore
   - Arêtes = contacts (Cα-Cα < 8Å)
   - Le réseau apprend à propager l'information dans le graphe

⚠️ PRÉREQUIS:
   pip install torch torch-geometric
   
   Pour torch-geometric, voir: https://pytorch-geometric.readthedocs.io/

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
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
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

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("❌ PyTorch requis! pip install torch")

# PyTorch Geometric
try:
    from torch_geometric.data import Data, DataLoader, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("⚠️ PyTorch Geometric non installé (optionnel pour GNN)")
    print("   pip install torch-geometric")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration."""
    
    DATA_DIR: Path = Path("data/processed")
    RAW_DIR: Path = Path("data/raw")
    STRUCTURES_DIR: Path = Path("data/structures")
    MODELS_DIR: Path = Path("models")
    REPORTS_DIR: Path = Path("reports")
    LOGS_DIR: Path = Path("logs")
    
    TRAIN_FILE: str = "dataset_train.csv"
    TEST_FILE: str = "dataset_test.csv"
    FPBASE_JSON: str = "fpbase.json"
    
    TARGETS: List[str] = None
    EXCLUDE_COLS: List[str] = None
    
    # GNN Config
    CONTACT_THRESHOLD: float = 8.0  # Angstroms pour définir un contact
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.2
    
    # Training Config
    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 20  # Early stopping
    
    RANDOM_SEED: int = 42
    
    def __post_init__(self):
        self.TARGETS = ['ex_max', 'em_max']
        self.EXCLUDE_COLS = [
            'protein_id', 'name', 'ex_max', 'em_max', 'qy', 
            'stokes_shift', 'ext_coeff', 'brightness'
        ]


CONFIG = Config()

# Seed pour reproductibilité
np.random.seed(CONFIG.RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(CONFIG.RANDOM_SEED)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = CONFIG.LOGS_DIR / f"deep_learning_{datetime.now():%Y%m%d_%H%M%S}.log"
    
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
    logger.info("🧠 DEEP LEARNING - GNN & Multi-Task")
    logger.info("="*70)
    
    if HAS_TORCH:
        logger.info(f"   Device: {DEVICE}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"   {len(feature_cols)} features")
    
    return X_train, X_test, feature_cols


# ============================================================================
# PARTIE 1: MULTI-TASK LEARNING
# ============================================================================

def train_multitask_sklearn(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            target_names: List[str],
                            logger: logging.Logger) -> Dict:
    """
    Multi-task learning avec scikit-learn.
    
    Utilise MultiOutputRegressor pour entraîner un seul modèle
    qui prédit toutes les cibles simultanément.
    """
    
    logger.info("")
    logger.info("   🎯 Multi-Task Learning (Extra Trees)")
    logger.info("-" * 50)
    
    # Modèle multi-output
    base_model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(base_model)
    
    logger.info("      Training...")
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Évaluation par cible
    results = {'model': model, 'targets': {}}
    
    for i, target in enumerate(target_names):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        results['targets'][target] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test[:, i],
            'y_pred': y_pred[:, i]
        }
        
        logger.info(f"      {target}: MAE={mae:.2f} nm | R²={r2:.4f}")
    
    return results


# ============================================================================
# PARTIE 2: MULTI-TASK NEURAL NETWORK (MLP)
# ============================================================================

class MultiTaskMLP(nn.Module):
    """
    Multi-Layer Perceptron pour multi-task learning.
    
    Architecture:
    - Shared layers (tronc commun)
    - Task-specific heads (têtes spécialisées)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_targets: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Task-specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(num_targets)
        ])
    
    def forward(self, x):
        shared_features = self.shared(x)
        outputs = [head(shared_features) for head in self.heads]
        return torch.cat(outputs, dim=1)


def train_multitask_nn(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       target_names: List[str],
                       logger: logging.Logger) -> Dict:
    """
    Multi-task learning avec Neural Network.
    """
    
    if not HAS_TORCH:
        logger.warning("PyTorch non disponible, skip NN")
        return None
    
    logger.info("")
    logger.info("   🧠 Multi-Task Neural Network")
    logger.info("-" * 50)
    
    # Convertir en tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)
    
    # Normaliser les cibles
    y_mean = y_train_t.mean(dim=0)
    y_std = y_train_t.std(dim=0)
    y_train_norm = (y_train_t - y_mean) / y_std
    
    # Modèle
    model = MultiTaskMLP(
        input_dim=X_train.shape[1],
        hidden_dim=CONFIG.HIDDEN_DIM,
        num_targets=len(target_names),
        dropout=CONFIG.DROPOUT
    ).to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=CONFIG.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    logger.info(f"      Training for {CONFIG.EPOCHS} epochs...")
    
    for epoch in range(CONFIG.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_norm)
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        train_losses.append(loss.item())
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG.PATIENCE:
            logger.info(f"      Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 50 == 0:
            logger.info(f"      Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # Charger le meilleur modèle
    model.load_state_dict(best_state)
    
    # Évaluation
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test_t)
        y_pred = y_pred_norm * y_std + y_mean
        y_pred = y_pred.cpu().numpy()
    
    results = {'model': model, 'targets': {}, 'train_losses': train_losses}
    
    for i, target in enumerate(target_names):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        results['targets'][target] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test[:, i],
            'y_pred': y_pred[:, i]
        }
        
        logger.info(f"      {target}: MAE={mae:.2f} nm | R²={r2:.4f}")
    
    return results


# ============================================================================
# PARTIE 3: GRAPH NEURAL NETWORK
# ============================================================================

# Mapping acides aminés
AA_TO_IDX = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_TO_IDX['X'] = 20  # Unknown


def parse_pdb_for_graph(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Parse un fichier PDB pour construire un graphe.
    
    Returns:
        ca_coords: Coordonnées des Cα
        features: Features par résidu (one-hot AA + pLDDT)
        info: Informations supplémentaires
    """
    
    AA_MAP = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    
    residues = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_id = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    b_factor = float(line[60:66]) if len(line) > 65 else 0.0
                    
                    if res_id not in residues:
                        residues[res_id] = {
                            'name': res_name,
                            'aa': AA_MAP.get(res_name, 'X'),
                            'ca_coord': None,
                            'b_factors': []
                        }
                    
                    residues[res_id]['b_factors'].append(b_factor)
                    
                    if atom_name == 'CA':
                        residues[res_id]['ca_coord'] = np.array([x, y, z])
    except:
        return None, None, None
    
    # Filtrer les résidus avec Cα
    valid_residues = {k: v for k, v in residues.items() if v['ca_coord'] is not None}
    
    if len(valid_residues) < 10:
        return None, None, None
    
    # Construire les arrays
    sorted_ids = sorted(valid_residues.keys())
    n_residues = len(sorted_ids)
    
    ca_coords = np.zeros((n_residues, 3))
    features = np.zeros((n_residues, 22))  # 21 AA one-hot + pLDDT
    
    for i, res_id in enumerate(sorted_ids):
        res = valid_residues[res_id]
        
        ca_coords[i] = res['ca_coord']
        
        # One-hot encoding
        aa_idx = AA_TO_IDX.get(res['aa'], 20)
        features[i, aa_idx] = 1.0
        
        # pLDDT (normalisé)
        plddt = np.mean(res['b_factors']) if res['b_factors'] else 50.0
        features[i, 21] = plddt / 100.0
    
    info = {
        'n_residues': n_residues,
        'sequence': ''.join(valid_residues[k]['aa'] for k in sorted_ids)
    }
    
    return ca_coords, features, info


def build_graph(ca_coords: np.ndarray, features: np.ndarray,
                threshold: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit le graphe de contacts.
    
    Arête entre deux résidus si distance Cα-Cα < threshold.
    """
    
    n = len(ca_coords)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist < threshold:
                edges.append([i, j])
                edges.append([j, i])  # Bidirectionnel
    
    if not edges:
        # Fallback: connecter séquentiellement
        for i in range(n - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
    
    edge_index = np.array(edges).T
    
    return edge_index


def create_pyg_data(ca_coords: np.ndarray, features: np.ndarray,
                    target: np.ndarray, threshold: float = 8.0) -> 'Data':
    """
    Crée un objet Data PyTorch Geometric.
    """
    
    edge_index = build_graph(ca_coords, features, threshold)
    
    data = Data(
        x=torch.FloatTensor(features),
        edge_index=torch.LongTensor(edge_index),
        y=torch.FloatTensor(target).unsqueeze(0)
    )
    
    return data


class ProteinGNN(nn.Module):
    """
    Graph Neural Network pour les structures protéiques.
    
    Architecture:
    - Plusieurs couches GCN/GAT
    - Global pooling (mean + max)
    - MLP final
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 3, num_targets: int = 2,
                 dropout: float = 0.2, use_gat: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_gat = use_gat
        
        # Couches de convolution
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Première couche
        if use_gat:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Couches cachées
        for _ in range(num_layers - 1):
            if use_gat:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout
        
        # MLP final (mean + max pooling = 2 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_targets)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # MLP
        out = self.mlp(x)
        
        return out


def prepare_gnn_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     target_names: List[str], structures_dir: Path,
                     logger: logging.Logger) -> Tuple[List, List]:
    """
    Prépare les données pour le GNN.
    """
    
    logger.info("   📊 Préparation des graphes...")
    
    train_graphs = []
    test_graphs = []
    
    # Trouver les fichiers PDB
    pdb_files = {f.stem.split('_')[1].lower(): f 
                 for f in structures_dir.glob('*.pdb')}
    
    logger.info(f"      {len(pdb_files)} fichiers PDB trouvés")
    
    for df, graphs, name in [(train_df, train_graphs, 'train'), 
                              (test_df, test_graphs, 'test')]:
        n_success = 0
        
        for _, row in df.iterrows():
            protein_id = row['protein_id'].lower()
            
            # Chercher le fichier PDB
            pdb_path = pdb_files.get(protein_id)
            
            if pdb_path is None:
                continue
            
            # Parser le PDB
            ca_coords, features, info = parse_pdb_for_graph(pdb_path)
            
            if ca_coords is None:
                continue
            
            # Cibles
            targets = np.array([row[t] for t in target_names])
            
            # Créer le graphe
            data = create_pyg_data(
                ca_coords, features, targets, 
                threshold=CONFIG.CONTACT_THRESHOLD
            )
            
            graphs.append(data)
            n_success += 1
        
        logger.info(f"      {name}: {n_success}/{len(df)} graphes créés")
    
    return train_graphs, test_graphs


def train_gnn(train_graphs: List, test_graphs: List,
              target_names: List[str], logger: logging.Logger) -> Dict:
    """
    Entraîne le Graph Neural Network.
    """
    
    if not HAS_PYG:
        logger.warning("PyTorch Geometric non disponible")
        return None
    
    if len(train_graphs) < 10:
        logger.warning("Pas assez de graphes pour entraîner")
        return None
    
    logger.info("")
    logger.info("   🕸️ Graph Neural Network")
    logger.info("-" * 50)
    
    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=CONFIG.BATCH_SIZE)
    
    # Normalisation des cibles
    all_targets = torch.cat([g.y for g in train_graphs], dim=0)
    y_mean = all_targets.mean(dim=0)
    y_std = all_targets.std(dim=0)
    
    # Modèle
    input_dim = train_graphs[0].x.shape[1]
    model = ProteinGNN(
        input_dim=input_dim,
        hidden_dim=CONFIG.HIDDEN_DIM,
        num_layers=CONFIG.NUM_LAYERS,
        num_targets=len(target_names),
        dropout=CONFIG.DROPOUT,
        use_gat=False  # GCN par défaut
    ).to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=CONFIG.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    logger.info(f"      Modèle: {sum(p.numel() for p in model.parameters())} paramètres")
    logger.info(f"      Training for {CONFIG.EPOCHS} epochs...")
    
    # Training
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    for epoch in range(CONFIG.EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Normaliser les cibles
            y_norm = (batch.y - y_mean.to(DEVICE)) / y_std.to(DEVICE)
            
            outputs = model(batch)
            loss = criterion(outputs, y_norm)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG.PATIENCE:
            logger.info(f"      Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 50 == 0:
            logger.info(f"      Epoch {epoch+1}: Loss = {epoch_loss:.4f}")
    
    # Charger le meilleur modèle
    model.load_state_dict(best_state)
    
    # Évaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            
            outputs = model(batch)
            # Dénormaliser
            outputs = outputs * y_std.to(DEVICE) + y_mean.to(DEVICE)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    y_pred = np.vstack(all_preds)
    y_test = np.vstack(all_targets)
    
    results = {'model': model, 'targets': {}, 'train_losses': train_losses}
    
    for i, target in enumerate(target_names):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        results['targets'][target] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test[:, i],
            'y_pred': y_pred[:, i]
        }
        
        logger.info(f"      {target}: MAE={mae:.2f} nm | R²={r2:.4f}")
    
    return results


# ============================================================================
# VISUALISATIONS
# ============================================================================

def create_comparison_plot(all_results: Dict, output_dir: Path, logger: logging.Logger):
    """Compare tous les modèles."""
    
    if not HAS_MATPLOTLIB:
        return
    
    # Baselines
    baseline = {'ex_max': 21.44, 'em_max': 18.47}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, target in enumerate(['ex_max', 'em_max']):
        ax = axes[idx]
        
        models = ['Baseline\n(Extra Trees)']
        maes = [baseline[target]]
        colors = ['#3498db']
        
        for name, result in all_results.items():
            if result and target in result['targets']:
                models.append(name)
                maes.append(result['targets'][target]['mae'])
                
                if name == 'Multi-Task\n(sklearn)':
                    colors.append('#9b59b6')
                elif name == 'Multi-Task\n(NN)':
                    colors.append('#e74c3c')
                else:
                    colors.append('#27ae60')
        
        bars = ax.bar(models, maes, color=colors)
        
        # Ligne objectif
        ax.axhline(y=15, color='red', linestyle='--', label='Objectif: 15 nm')
        
        # Annotations
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mae:.2f}', ha='center', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('MAE (nm)', fontsize=12)
        ax.set_title(f'{target}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(maes) + 5)
        ax.legend()
    
    plt.suptitle('Comparaison des Approches Deep Learning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'deep_learning_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_training_curves(results: Dict, output_dir: Path, logger: logging.Logger):
    """Courbes d'apprentissage."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, result in results.items():
        if result and 'train_losses' in result:
            ax.plot(result['train_losses'], label=name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Courbes d\'Apprentissage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'deep_learning_training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


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
    
    # Préparer les cibles multi-task (ex_max, em_max)
    target_names = ['ex_max', 'em_max']
    y_train = train_df[target_names].values
    y_test = test_df[target_names].values
    
    # Gérer les NaN dans QY (optionnel)
    logger.info(f"   Cibles: {target_names}")
    
    all_results = {}
    
    # 2. Multi-Task sklearn
    logger.info("")
    logger.info("🎯 ÉTAPE 2: Multi-Task Learning")
    logger.info("=" * 50)
    
    mt_sklearn = train_multitask_sklearn(
        X_train, y_train, X_test, y_test, target_names, logger
    )
    all_results['Multi-Task\n(sklearn)'] = mt_sklearn
    
    # 3. Multi-Task Neural Network
    if HAS_TORCH:
        mt_nn = train_multitask_nn(
            X_train, y_train, X_test, y_test, target_names, logger
        )
        all_results['Multi-Task\n(NN)'] = mt_nn
    
    # 4. Graph Neural Network
    if HAS_PYG and CONFIG.STRUCTURES_DIR.exists():
        logger.info("")
        logger.info("🕸️ ÉTAPE 3: Graph Neural Network")
        logger.info("=" * 50)
        
        train_graphs, test_graphs = prepare_gnn_data(
            train_df, test_df, target_names, CONFIG.STRUCTURES_DIR, logger
        )
        
        if train_graphs and test_graphs:
            gnn_results = train_gnn(train_graphs, test_graphs, target_names, logger)
            all_results['GNN'] = gnn_results
    else:
        logger.info("")
        logger.info("⚠️ GNN skipped (PyTorch Geometric non disponible ou structures manquantes)")
    
    # 5. Visualisations
    logger.info("")
    logger.info("📊 ÉTAPE 4: Visualisations")
    logger.info("-" * 50)
    
    CONFIG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plot(all_results, CONFIG.REPORTS_DIR, logger)
    create_training_curves(all_results, CONFIG.REPORTS_DIR, logger)
    
    # 6. Résumé final
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    
    baseline = {'ex_max': 21.44, 'em_max': 18.47}
    
    logger.info("\n   Comparaison des MAE (nm):")
    logger.info("-" * 50)
    logger.info(f"   {'Modèle':<25} {'ex_max':>10} {'em_max':>10}")
    logger.info("-" * 50)
    logger.info(f"   {'Baseline (Extra Trees)':<25} {baseline['ex_max']:>10.2f} {baseline['em_max']:>10.2f}")
    
    for name, result in all_results.items():
        if result:
            ex_mae = result['targets']['ex_max']['mae']
            em_mae = result['targets']['em_max']['mae']
            logger.info(f"   {name.replace(chr(10), ' '):<25} {ex_mae:>10.2f} {em_mae:>10.2f}")
    
    # Meilleur modèle
    best_em_mae = baseline['em_max']
    best_model = 'Baseline'
    
    for name, result in all_results.items():
        if result and result['targets']['em_max']['mae'] < best_em_mae:
            best_em_mae = result['targets']['em_max']['mae']
            best_model = name.replace('\n', ' ')
    
    logger.info("")
    logger.info(f"   🏆 Meilleur modèle: {best_model}")
    logger.info(f"   📊 Meilleur MAE (em_max): {best_em_mae:.2f} nm")
    
    if best_em_mae < 15:
        logger.info("   🎉 OBJECTIF ATTEINT! MAE < 15nm")
    elif best_em_mae < 18:
        logger.info("   ✅ TRÈS BON! MAE < 18nm")
    else:
        logger.info("   👍 Résultats obtenus")
    
    logger.info("")
    logger.info("📁 Fichiers générés:")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/deep_learning_comparison.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/deep_learning_training_curves.png")


if __name__ == "__main__":
    main()
