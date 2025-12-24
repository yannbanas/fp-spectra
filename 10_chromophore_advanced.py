#!/usr/bin/env python3
"""
============================================================================
10_CHROMOPHORE_ADVANCED.PY - Features Avancés du Chromophore + ESM-2 Local
============================================================================

🎯 OBJECTIF:
   Améliorer les prédictions en se concentrant sur le chromophore:
   1. Features géométriques avancés du chromophore
   2. ESM-2 local (embeddings sur ±15 résidus autour du chromophore)

📚 NOUVEAUX FEATURES:
   
   GÉOMÉTRIE DU CHROMOPHORE:
   - Distances O-H (liaisons hydrogène avec résidus voisins)
   - Angles dihédraux supplémentaires (ω, φ)
   - Planarité du système conjugué
   - Distances interatomiques clés
   
   ENVIRONNEMENT:
   - Volume de la cavité (approximé par convex hull)
   - Densité de résidus polaires/aromatiques
   - Réseau de liaisons H
   
   ESM-2 LOCAL:
   - Embeddings sur fenêtre [chromophore-15 : chromophore+15]
   - Capture l'environnement immédiat sans dilution

📤 OUTPUT:
   - data/processed/advanced_features.csv
   - models/extra_trees_advanced.joblib
   - reports/advanced_results.html

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
import re

warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import ExtraTreesRegressor
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
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False
    DEVICE = None

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Scipy pour calculs géométriques
try:
    from scipy.spatial import ConvexHull, distance
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy recommandé: pip install scipy")


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
    
    # ESM-2 config
    ESM2_MODEL: str = "facebook/esm2_t33_650M_UR50D"
    LOCAL_WINDOW: int = 15  # ±15 résidus autour du chromophore
    PCA_COMPONENTS: int = 30  # Réduire les embeddings locaux
    
    # Chromophore config
    HBOND_DISTANCE: float = 3.5  # Distance max pour liaison H (Å)
    CAVITY_RADIUS: float = 6.0   # Rayon pour calcul de cavité (Å)
    
    RANDOM_SEED: int = 42
    N_JOBS: int = -1
    
    def __post_init__(self):
        self.TARGETS = ['ex_max', 'em_max']


CONFIG = Config()


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = CONFIG.LOGS_DIR / f"advanced_{datetime.now():%Y%m%d_%H%M%S}.log"
    
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
    logger.info("🔬 FEATURES AVANCÉS - Chromophore + ESM-2 Local")
    logger.info("="*70)
    
    return logger


# ============================================================================
# PARSING PDB AVANCÉ
# ============================================================================

AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# Résidus polaires/aromatiques/donneurs H
POLAR_RESIDUES = set('STNQYDE')
AROMATIC_RESIDUES = set('FYW')
HBOND_DONORS = set('STNQKRHWY')
HBOND_ACCEPTORS = set('STNQDEHY')
HYDROPHOBIC_RESIDUES = set('AVILMFYW')


class PDBParser:
    """Parser PDB avancé pour extraction de features."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.atoms = []
        self.residues = {}
        self.sequence = ""
        self._parse()
    
    def _parse(self):
        """Parse le fichier PDB."""
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        atom = self._parse_atom_line(line)
                        if atom:
                            self.atoms.append(atom)
                            
                            res_id = atom['res_id']
                            if res_id not in self.residues:
                                self.residues[res_id] = {
                                    'name': atom['res_name'],
                                    'aa': AA_MAP.get(atom['res_name'], 'X'),
                                    'atoms': {}
                                }
                            self.residues[res_id]['atoms'][atom['name']] = atom
            
            # Construire la séquence
            sorted_ids = sorted(self.residues.keys())
            self.sequence = ''.join(self.residues[i]['aa'] for i in sorted_ids)
            
        except Exception as e:
            print(f"Erreur parsing {self.filepath}: {e}")
    
    def _parse_atom_line(self, line: str) -> Optional[Dict]:
        """Parse une ligne ATOM."""
        try:
            return {
                'name': line[12:16].strip(),
                'res_name': line[17:20].strip(),
                'res_id': int(line[22:26]),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'b_factor': float(line[60:66]) if len(line) > 65 else 0.0,
                'coord': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            }
        except:
            return None
    
    def get_atom_coord(self, res_id: int, atom_name: str) -> Optional[np.ndarray]:
        """Récupère les coordonnées d'un atome."""
        if res_id in self.residues and atom_name in self.residues[res_id]['atoms']:
            return self.residues[res_id]['atoms'][atom_name]['coord']
        return None
    
    def get_residue_atoms(self, res_id: int) -> Dict:
        """Récupère tous les atomes d'un résidu."""
        if res_id in self.residues:
            return self.residues[res_id]['atoms']
        return {}
    
    def find_chromophore(self) -> Optional[Tuple[int, int, int]]:
        """
        Trouve le chromophore (tripeptide X-Y-G).
        Retourne (res_id_X, res_id_Y, res_id_G).
        """
        sorted_ids = sorted(self.residues.keys())
        
        # Patterns connus
        patterns = ['SYG', 'TYG', 'GYG', 'AYG', 'CYG', 'MYG', 'VYG', 'LYG', 'QYG']
        
        for i in range(len(sorted_ids) - 2):
            triplet = ''.join(self.residues[sorted_ids[j]]['aa'] for j in range(i, i+3))
            
            if triplet in patterns:
                return (sorted_ids[i], sorted_ids[i+1], sorted_ids[i+2])
        
        # Fallback: chercher Y-G
        for i in range(len(sorted_ids) - 1):
            aa1 = self.residues[sorted_ids[i]]['aa']
            aa2 = self.residues[sorted_ids[i+1]]['aa']
            
            if aa1 == 'Y' and aa2 == 'G':
                if i > 0:
                    return (sorted_ids[i-1], sorted_ids[i], sorted_ids[i+1])
        
        return None


# ============================================================================
# FEATURES GÉOMÉTRIQUES AVANCÉS
# ============================================================================

def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, 
                       p3: np.ndarray, p4: np.ndarray) -> float:
    """Calcule l'angle dihédral entre 4 points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)
    
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.degrees(np.arctan2(y, x))


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calcule l'angle entre 3 points (en degrés)."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    return np.degrees(np.arccos(cos_angle))


def calculate_planarity(coords: List[np.ndarray]) -> float:
    """
    Calcule la planarité d'un ensemble de points (RMSD par rapport au plan).
    """
    if len(coords) < 4:
        return 0.0
    
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    
    # SVD pour trouver le plan
    _, s, vh = np.linalg.svd(centered)
    
    # Le vecteur normal au plan est la dernière composante de vh
    normal = vh[-1]
    
    # RMSD des distances au plan
    distances = np.abs(np.dot(centered, normal))
    rmsd = np.sqrt(np.mean(distances**2))
    
    return rmsd


def extract_chromophore_features(pdb: PDBParser, chrom_ids: Tuple[int, int, int]) -> Dict:
    """
    Extrait les features avancés du chromophore.
    """
    features = {}
    
    res_x, res_y, res_g = chrom_ids
    
    # =====================================================================
    # 1. ANGLES DIHÉDRAUX
    # =====================================================================
    
    # Phi (φ) de la Tyrosine: C(i-1)-N-Cα-C
    c_prev = pdb.get_atom_coord(res_x, 'C')
    n_y = pdb.get_atom_coord(res_y, 'N')
    ca_y = pdb.get_atom_coord(res_y, 'CA')
    c_y = pdb.get_atom_coord(res_y, 'C')
    
    if all(x is not None for x in [c_prev, n_y, ca_y, c_y]):
        features['chrom_phi_tyr'] = calculate_dihedral(c_prev, n_y, ca_y, c_y)
    else:
        features['chrom_phi_tyr'] = np.nan
    
    # Psi (ψ) de la Tyrosine: N-Cα-C-N(i+1)
    n_next = pdb.get_atom_coord(res_g, 'N')
    if all(x is not None for x in [n_y, ca_y, c_y, n_next]):
        features['chrom_psi_tyr'] = calculate_dihedral(n_y, ca_y, c_y, n_next)
    else:
        features['chrom_psi_tyr'] = np.nan
    
    # Omega (ω) entre X et Y: Cα(i-1)-C(i-1)-N-Cα
    ca_x = pdb.get_atom_coord(res_x, 'CA')
    c_x = pdb.get_atom_coord(res_x, 'C')
    
    if all(x is not None for x in [ca_x, c_x, n_y, ca_y]):
        features['chrom_omega_xy'] = calculate_dihedral(ca_x, c_x, n_y, ca_y)
    else:
        features['chrom_omega_xy'] = np.nan
    
    # Chi1 (χ1) de la Tyrosine: N-Cα-Cβ-Cγ
    cb_y = pdb.get_atom_coord(res_y, 'CB')
    cg_y = pdb.get_atom_coord(res_y, 'CG')
    
    if all(x is not None for x in [n_y, ca_y, cb_y, cg_y]):
        features['chrom_chi1_tyr'] = calculate_dihedral(n_y, ca_y, cb_y, cg_y)
    else:
        features['chrom_chi1_tyr'] = np.nan
    
    # Chi2 (χ2) de la Tyrosine: Cα-Cβ-Cγ-CD1
    cd1_y = pdb.get_atom_coord(res_y, 'CD1')
    
    if all(x is not None for x in [ca_y, cb_y, cg_y, cd1_y]):
        features['chrom_chi2_tyr'] = calculate_dihedral(ca_y, cb_y, cg_y, cd1_y)
    else:
        features['chrom_chi2_tyr'] = np.nan
    
    # =====================================================================
    # 2. DISTANCES INTERATOMIQUES
    # =====================================================================
    
    # Distance Cα-Cα du tripeptide
    ca_g = pdb.get_atom_coord(res_g, 'CA')
    
    if ca_x is not None and ca_y is not None:
        features['chrom_dist_ca_xy'] = np.linalg.norm(ca_x - ca_y)
    else:
        features['chrom_dist_ca_xy'] = np.nan
    
    if ca_y is not None and ca_g is not None:
        features['chrom_dist_ca_yg'] = np.linalg.norm(ca_y - ca_g)
    else:
        features['chrom_dist_ca_yg'] = np.nan
    
    if ca_x is not None and ca_g is not None:
        features['chrom_dist_ca_xg'] = np.linalg.norm(ca_x - ca_g)
    else:
        features['chrom_dist_ca_xg'] = np.nan
    
    # Distance OH (phénol) - backbone
    oh_y = pdb.get_atom_coord(res_y, 'OH')
    
    if oh_y is not None:
        # Distance OH - N du Gly
        if n_next is not None:
            features['chrom_dist_oh_n_gly'] = np.linalg.norm(oh_y - n_next)
        else:
            features['chrom_dist_oh_n_gly'] = np.nan
        
        # Distance OH - C=O du résidu X
        o_x = pdb.get_atom_coord(res_x, 'O')
        if o_x is not None:
            features['chrom_dist_oh_o_x'] = np.linalg.norm(oh_y - o_x)
        else:
            features['chrom_dist_oh_o_x'] = np.nan
    else:
        features['chrom_dist_oh_n_gly'] = np.nan
        features['chrom_dist_oh_o_x'] = np.nan
    
    # =====================================================================
    # 3. PLANARITÉ DU CHROMOPHORE
    # =====================================================================
    
    # Atomes du cycle aromatique de la Tyrosine
    ring_atoms = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
    ring_coords = []
    
    for atom_name in ring_atoms:
        coord = pdb.get_atom_coord(res_y, atom_name)
        if coord is not None:
            ring_coords.append(coord)
    
    if len(ring_coords) >= 4:
        features['chrom_ring_planarity'] = calculate_planarity(ring_coords)
    else:
        features['chrom_ring_planarity'] = np.nan
    
    # Planarité du système conjugué étendu (inclut le backbone)
    conjugated_coords = ring_coords.copy()
    for atom_name in ['N', 'CA', 'C', 'O']:
        coord = pdb.get_atom_coord(res_y, atom_name)
        if coord is not None:
            conjugated_coords.append(coord)
    
    if len(conjugated_coords) >= 6:
        features['chrom_conjugated_planarity'] = calculate_planarity(conjugated_coords)
    else:
        features['chrom_conjugated_planarity'] = np.nan
    
    # =====================================================================
    # 4. ANGLES AVEC LE BACKBONE
    # =====================================================================
    
    # Angle Cα-Cβ-Cγ (inclinaison du cycle)
    if all(x is not None for x in [ca_y, cb_y, cg_y]):
        features['chrom_angle_ca_cb_cg'] = calculate_angle(ca_y, cb_y, cg_y)
    else:
        features['chrom_angle_ca_cb_cg'] = np.nan
    
    # Angle N-Cα-C (angle du backbone)
    if all(x is not None for x in [n_y, ca_y, c_y]):
        features['chrom_angle_n_ca_c'] = calculate_angle(n_y, ca_y, c_y)
    else:
        features['chrom_angle_n_ca_c'] = np.nan
    
    return features


# ============================================================================
# FEATURES D'ENVIRONNEMENT AVANCÉS
# ============================================================================

def calculate_cavity_volume(pdb: PDBParser, center: np.ndarray, 
                            radius: float = 6.0) -> float:
    """
    Estime le volume de la cavité autour du chromophore.
    Utilise le convex hull des atomes dans le rayon.
    """
    if not HAS_SCIPY:
        return np.nan
    
    # Collecter les atomes dans le rayon
    nearby_coords = []
    for atom in pdb.atoms:
        dist = np.linalg.norm(atom['coord'] - center)
        if dist < radius:
            nearby_coords.append(atom['coord'])
    
    if len(nearby_coords) < 4:
        return np.nan
    
    try:
        hull = ConvexHull(nearby_coords)
        return hull.volume
    except:
        return np.nan


def find_hbonds(pdb: PDBParser, chrom_ids: Tuple[int, int, int],
                max_dist: float = 3.5) -> Dict:
    """
    Trouve les liaisons hydrogène potentielles avec le chromophore.
    """
    features = {}
    
    res_x, res_y, res_g = chrom_ids
    
    # Atomes du chromophore pouvant former des H-bonds
    chrom_hbond_atoms = []
    
    # OH de la Tyrosine (donneur ET accepteur)
    oh = pdb.get_atom_coord(res_y, 'OH')
    if oh is not None:
        chrom_hbond_atoms.append(('OH_tyr', oh, 'both'))
    
    # N du backbone (donneur)
    for res_id, label in [(res_x, 'X'), (res_y, 'Y'), (res_g, 'G')]:
        n = pdb.get_atom_coord(res_id, 'N')
        if n is not None:
            chrom_hbond_atoms.append((f'N_{label}', n, 'donor'))
    
    # O du backbone (accepteur)
    for res_id, label in [(res_x, 'X'), (res_y, 'Y')]:
        o = pdb.get_atom_coord(res_id, 'O')
        if o is not None:
            chrom_hbond_atoms.append((f'O_{label}', o, 'acceptor'))
    
    # Compter les H-bonds potentielles
    n_hbonds = 0
    hbond_distances = []
    
    sorted_ids = sorted(pdb.residues.keys())
    chrom_set = set(chrom_ids)
    
    for res_id in sorted_ids:
        if res_id in chrom_set:
            continue
        
        res = pdb.residues[res_id]
        aa = res['aa']
        
        # Vérifier chaque atome du chromophore
        for chrom_atom_name, chrom_coord, hbond_type in chrom_hbond_atoms:
            
            # Atomes potentiels de l'autre résidu
            test_atoms = []
            
            if hbond_type in ['donor', 'both']:
                # Le chromophore donne, chercher des accepteurs
                if aa in HBOND_ACCEPTORS:
                    for atom_name in ['O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'ND1', 'NE2']:
                        coord = pdb.get_atom_coord(res_id, atom_name)
                        if coord is not None:
                            test_atoms.append(coord)
            
            if hbond_type in ['acceptor', 'both']:
                # Le chromophore accepte, chercher des donneurs
                if aa in HBOND_DONORS:
                    for atom_name in ['N', 'NZ', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'OG', 'OG1', 'OH']:
                        coord = pdb.get_atom_coord(res_id, atom_name)
                        if coord is not None:
                            test_atoms.append(coord)
            
            # Calculer les distances
            for test_coord in test_atoms:
                dist = np.linalg.norm(chrom_coord - test_coord)
                if dist < max_dist:
                    n_hbonds += 1
                    hbond_distances.append(dist)
    
    features['env_n_hbonds'] = n_hbonds
    features['env_hbond_dist_mean'] = np.mean(hbond_distances) if hbond_distances else np.nan
    features['env_hbond_dist_min'] = np.min(hbond_distances) if hbond_distances else np.nan
    
    return features


def extract_environment_features(pdb: PDBParser, chrom_ids: Tuple[int, int, int]) -> Dict:
    """
    Extrait les features d'environnement avancés.
    """
    features = {}
    
    res_x, res_y, res_g = chrom_ids
    
    # Centre du chromophore (Cα de la Tyrosine)
    ca_y = pdb.get_atom_coord(res_y, 'CA')
    if ca_y is None:
        return features
    
    # =====================================================================
    # 1. VOLUME DE CAVITÉ
    # =====================================================================
    
    features['env_cavity_volume_6A'] = calculate_cavity_volume(pdb, ca_y, radius=6.0)
    features['env_cavity_volume_8A'] = calculate_cavity_volume(pdb, ca_y, radius=8.0)
    
    # =====================================================================
    # 2. COMPOSITION DE L'ENVIRONNEMENT
    # =====================================================================
    
    sorted_ids = sorted(pdb.residues.keys())
    chrom_set = set(chrom_ids)
    
    neighbors = {'polar': 0, 'aromatic': 0, 'hydrophobic': 0, 'total': 0}
    neighbor_distances = []
    
    for res_id in sorted_ids:
        if res_id in chrom_set:
            continue
        
        # Distance au chromophore
        ca = pdb.get_atom_coord(res_id, 'CA')
        if ca is None:
            continue
        
        dist = np.linalg.norm(ca - ca_y)
        
        if dist < 8.0:
            neighbors['total'] += 1
            neighbor_distances.append(dist)
            
            aa = pdb.residues[res_id]['aa']
            
            if aa in POLAR_RESIDUES:
                neighbors['polar'] += 1
            if aa in AROMATIC_RESIDUES:
                neighbors['aromatic'] += 1
            if aa in HYDROPHOBIC_RESIDUES:
                neighbors['hydrophobic'] += 1
    
    features['env_n_neighbors_8A'] = neighbors['total']
    features['env_n_polar_8A'] = neighbors['polar']
    features['env_n_aromatic_8A'] = neighbors['aromatic']
    features['env_n_hydrophobic_8A'] = neighbors['hydrophobic']
    
    if neighbors['total'] > 0:
        features['env_ratio_polar'] = neighbors['polar'] / neighbors['total']
        features['env_ratio_aromatic'] = neighbors['aromatic'] / neighbors['total']
        features['env_ratio_hydrophobic'] = neighbors['hydrophobic'] / neighbors['total']
    else:
        features['env_ratio_polar'] = np.nan
        features['env_ratio_aromatic'] = np.nan
        features['env_ratio_hydrophobic'] = np.nan
    
    features['env_neighbor_dist_mean'] = np.mean(neighbor_distances) if neighbor_distances else np.nan
    
    # =====================================================================
    # 3. LIAISONS HYDROGÈNE
    # =====================================================================
    
    hbond_features = find_hbonds(pdb, chrom_ids, max_dist=CONFIG.HBOND_DISTANCE)
    features.update(hbond_features)
    
    # =====================================================================
    # 4. pLDDT LOCAL
    # =====================================================================
    
    plddt_local = []
    for res_id in sorted_ids:
        ca = pdb.get_atom_coord(res_id, 'CA')
        if ca is None:
            continue
        
        dist = np.linalg.norm(ca - ca_y)
        if dist < 8.0:
            atoms = pdb.get_residue_atoms(res_id)
            for atom in atoms.values():
                plddt_local.append(atom['b_factor'])
    
    features['env_plddt_local_mean'] = np.mean(plddt_local) if plddt_local else np.nan
    features['env_plddt_local_std'] = np.std(plddt_local) if plddt_local else np.nan
    
    return features


# ============================================================================
# ESM-2 LOCAL
# ============================================================================

class LocalESM2Embedder:
    """
    Génère des embeddings ESM-2 sur une fenêtre locale autour du chromophore.
    """
    
    def __init__(self, model_name: str, window: int = 15, logger: logging.Logger = None):
        self.model_name = model_name
        self.window = window
        self.logger = logger or logging.getLogger(__name__)
        
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            self.model = None
            return
        
        self.logger.info(f"   📥 Chargement de {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()
        
        self.logger.info(f"   ✅ ESM-2 chargé sur {DEVICE}")
    
    def get_local_embedding(self, sequence: str, chrom_pos: int) -> Optional[np.ndarray]:
        """
        Génère l'embedding pour la fenêtre locale autour du chromophore.
        
        Args:
            sequence: Séquence complète
            chrom_pos: Position du chromophore (résidu central, la Tyrosine)
        
        Returns:
            Embedding de la fenêtre locale (moyenne)
        """
        if self.model is None:
            return None
        
        # Extraire la fenêtre locale
        start = max(0, chrom_pos - self.window)
        end = min(len(sequence), chrom_pos + self.window + 1)
        
        local_seq = sequence[start:end]
        
        if len(local_seq) < 5:
            return None
        
        # Tokenizer
        inputs = self.tokenizer(
            local_seq, 
            return_tensors="pt", 
            truncation=True,
            max_length=100,
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Inférence
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        mean_pooled = summed / counts
        
        return mean_pooled.cpu().numpy().flatten()


def generate_local_esm2_embeddings(protein_ids: List[str], sequences: Dict[str, str],
                                   chrom_positions: Dict[str, int],
                                   logger: logging.Logger) -> Dict[str, np.ndarray]:
    """
    Génère les embeddings ESM-2 locaux pour toutes les protéines.
    """
    
    if not HAS_TORCH or not HAS_TRANSFORMERS:
        logger.warning("ESM-2 non disponible")
        return {}
    
    embedder = LocalESM2Embedder(CONFIG.ESM2_MODEL, CONFIG.LOCAL_WINDOW, logger)
    
    if embedder.model is None:
        return {}
    
    embeddings = {}
    n_total = len(protein_ids)
    
    logger.info(f"   🔄 Génération des embeddings locaux...")
    
    for i, pid in enumerate(protein_ids):
        if (i + 1) % 50 == 0:
            logger.info(f"      Progression: {i+1}/{n_total}")
        
        if pid not in sequences or pid not in chrom_positions:
            continue
        
        seq = sequences[pid]
        chrom_pos = chrom_positions[pid]
        
        try:
            emb = embedder.get_local_embedding(seq, chrom_pos)
            if emb is not None:
                embeddings[pid] = emb
        except Exception as e:
            continue
    
    logger.info(f"   ✅ {len(embeddings)} embeddings locaux générés")
    
    return embeddings


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def extract_all_advanced_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                   structures_dir: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrait tous les features avancés pour train et test.
    """
    
    logger.info("   📊 Extraction des features avancés...")
    
    # Trouver les fichiers PDB
    pdb_files = {}
    for f in structures_dir.glob('*.pdb'):
        # Extraire le protein_id du nom de fichier
        stem = f.stem.lower()
        if '_' in stem:
            pid = stem.split('_')[1]
        else:
            pid = stem
        pdb_files[pid] = f
    
    logger.info(f"      {len(pdb_files)} fichiers PDB trouvés")
    
    all_features = []
    sequences = {}
    chrom_positions = {}
    
    all_proteins = pd.concat([train_df, test_df])
    n_total = len(all_proteins)
    n_success = 0
    
    for idx, (_, row) in enumerate(all_proteins.iterrows()):
        if (idx + 1) % 100 == 0:
            logger.info(f"      Progression: {idx+1}/{n_total}")
        
        pid = row['protein_id'].lower()
        
        if pid not in pdb_files:
            all_features.append({'protein_id': row['protein_id']})
            continue
        
        # Parser le PDB
        pdb = PDBParser(pdb_files[pid])
        
        if not pdb.residues:
            all_features.append({'protein_id': row['protein_id']})
            continue
        
        # Trouver le chromophore
        chrom = pdb.find_chromophore()
        
        if chrom is None:
            all_features.append({'protein_id': row['protein_id']})
            continue
        
        # Stocker la séquence et position du chromophore
        sequences[row['protein_id']] = pdb.sequence
        sorted_ids = sorted(pdb.residues.keys())
        chrom_idx = sorted_ids.index(chrom[1])  # Position de la Tyrosine
        chrom_positions[row['protein_id']] = chrom_idx
        
        # Extraire les features
        features = {'protein_id': row['protein_id']}
        
        # Features chromophore
        chrom_features = extract_chromophore_features(pdb, chrom)
        features.update(chrom_features)
        
        # Features environnement
        env_features = extract_environment_features(pdb, chrom)
        features.update(env_features)
        
        all_features.append(features)
        n_success += 1
    
    logger.info(f"      {n_success}/{n_total} protéines traitées")
    
    # Créer le DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Séparer train/test
    train_features = features_df[features_df['protein_id'].isin(train_df['protein_id'])]
    test_features = features_df[features_df['protein_id'].isin(test_df['protein_id'])]
    
    return train_features, test_features, sequences, chrom_positions


# ============================================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       target: str, logger: logging.Logger) -> Dict:
    """
    Entraîne Extra Trees et évalue.
    """
    
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=CONFIG.N_JOBS
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"      {target}: MAE={mae:.2f} nm | RMSE={rmse:.2f} nm | R²={r2:.4f}")
    
    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }


# ============================================================================
# VISUALISATIONS
# ============================================================================

def create_comparison_plot(results: Dict, output_dir: Path, logger: logging.Logger):
    """Compare avec les baselines."""
    
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    baselines = {
        'ex_max': [
            ('RF Baseline', 21.93),
            ('Extra Trees', 21.44),
            ('Advanced', results.get('ex_max', {}).get('mae', np.nan))
        ],
        'em_max': [
            ('RF Baseline', 19.28),
            ('Extra Trees', 18.47),
            ('Advanced', results.get('em_max', {}).get('mae', np.nan))
        ]
    }
    
    for idx, target in enumerate(['ex_max', 'em_max']):
        ax = axes[idx]
        
        models = [b[0] for b in baselines[target]]
        maes = [b[1] for b in baselines[target]]
        
        colors = ['#3498db', '#9b59b6', '#27ae60']
        bars = ax.bar(models, maes, color=colors)
        
        # Annotations
        for bar, mae in zip(bars, maes):
            if not np.isnan(mae):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{mae:.2f}', ha='center', fontsize=12, fontweight='bold')
        
        # Objectif
        ax.axhline(y=15, color='red', linestyle='--', label='Objectif: 15 nm')
        
        ax.set_ylabel('MAE (nm)', fontsize=12)
        ax.set_title(f'{target}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(m for m in maes if not np.isnan(m)) + 5)
        ax.legend()
    
    plt.suptitle('Impact des Features Avancés + ESM-2 Local', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'advanced_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   📊 {output_path.name}")


def create_feature_importance_plot(model, feature_names: List[str], 
                                   target: str, output_dir: Path, logger: logging.Logger):
    """Importance des features."""
    
    if not HAS_MATPLOTLIB:
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(indices)), importances[indices], color='#27ae60')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top 20 Features - {target}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / f'advanced_importance_{target}.png'
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
    
    train_path = CONFIG.DATA_DIR / CONFIG.TRAIN_FILE
    test_path = CONFIG.DATA_DIR / CONFIG.TEST_FILE
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"   Train: {len(train_df)} | Test: {len(test_df)}")
    
    # 2. Extraire les features avancés
    logger.info("")
    logger.info("🔬 ÉTAPE 2: Extraction des features avancés")
    logger.info("-" * 50)
    
    train_adv, test_adv, sequences, chrom_positions = extract_all_advanced_features(
        train_df, test_df, CONFIG.STRUCTURES_DIR, logger
    )
    
    # 3. ESM-2 local
    logger.info("")
    logger.info("🧬 ÉTAPE 3: ESM-2 Local")
    logger.info("-" * 50)
    
    all_pids = list(train_df['protein_id']) + list(test_df['protein_id'])
    local_embeddings = generate_local_esm2_embeddings(
        all_pids, sequences, chrom_positions, logger
    )
    
    # 4. Préparer les features combinés
    logger.info("")
    logger.info("🔗 ÉTAPE 4: Combinaison des features")
    logger.info("-" * 50)
    
    # Merger avec les features existants
    train_merged = train_df.merge(train_adv, on='protein_id', how='left')
    test_merged = test_df.merge(test_adv, on='protein_id', how='left')
    
    # Colonnes de features (exclure les métadonnées et cibles)
    exclude = ['protein_id', 'name', 'ex_max', 'em_max', 'qy', 'stokes_shift', 'ext_coeff', 'brightness']
    feature_cols = [c for c in train_merged.columns if c not in exclude 
                    and train_merged[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X_train = train_merged[feature_cols].values
    X_test = test_merged[feature_cols].values
    
    # Imputer et scaler
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"   Features structuraux: {len(feature_cols)}")
    
    # Ajouter les embeddings ESM-2 locaux
    if local_embeddings:
        # Créer les matrices d'embeddings
        emb_dim = len(next(iter(local_embeddings.values())))
        
        train_emb = np.zeros((len(train_df), emb_dim))
        test_emb = np.zeros((len(test_df), emb_dim))
        
        for i, pid in enumerate(train_df['protein_id']):
            if pid in local_embeddings:
                train_emb[i] = local_embeddings[pid]
        
        for i, pid in enumerate(test_df['protein_id']):
            if pid in local_embeddings:
                test_emb[i] = local_embeddings[pid]
        
        # PCA sur les embeddings
        logger.info(f"   ESM-2 local: {emb_dim} → {CONFIG.PCA_COMPONENTS} (PCA)")
        
        pca = PCA(n_components=CONFIG.PCA_COMPONENTS, random_state=CONFIG.RANDOM_SEED)
        train_emb_pca = pca.fit_transform(train_emb)
        test_emb_pca = pca.transform(test_emb)
        
        # Combiner
        X_train = np.hstack([X_train, train_emb_pca])
        X_test = np.hstack([X_test, test_emb_pca])
        
        # Ajouter les noms de features
        feature_cols = feature_cols + [f'esm2_local_{i}' for i in range(CONFIG.PCA_COMPONENTS)]
    
    logger.info(f"   Total features: {X_train.shape[1]}")
    
    # 5. Entraînement
    logger.info("")
    logger.info("🏋️ ÉTAPE 5: Entraînement")
    logger.info("-" * 50)
    
    results = {}
    
    for target in CONFIG.TARGETS:
        y_train = train_df[target].values
        y_test = test_df[target].values
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, target, logger)
        results[target] = result
    
    # 6. Visualisations
    logger.info("")
    logger.info("📊 ÉTAPE 6: Visualisations")
    logger.info("-" * 50)
    
    CONFIG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plot(results, CONFIG.REPORTS_DIR, logger)
    
    for target in CONFIG.TARGETS:
        if target in results:
            create_feature_importance_plot(
                results[target]['model'], feature_cols, target, CONFIG.REPORTS_DIR, logger
            )
    
    # 7. Sauvegarder les modèles
    logger.info("")
    logger.info("💾 ÉTAPE 7: Sauvegarde")
    logger.info("-" * 50)
    
    if HAS_JOBLIB:
        CONFIG.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        for target, result in results.items():
            path = CONFIG.MODELS_DIR / f"extra_trees_advanced_{target}.joblib"
            joblib.dump(result['model'], path)
            logger.info(f"   💾 {path.name}")
    
    # 8. Résumé
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    
    baseline = {'ex_max': 21.44, 'em_max': 18.47}
    
    for target in CONFIG.TARGETS:
        if target in results:
            r = results[target]
            improvement = baseline[target] - r['mae']
            
            logger.info(f"\n   🎯 {target}:")
            logger.info(f"      Baseline (Extra Trees): {baseline[target]:.2f} nm")
            logger.info(f"      Advanced + ESM-2 Local: {r['mae']:.2f} nm")
            logger.info(f"      Amélioration: {improvement:+.2f} nm")
            logger.info(f"      R²: {r['r2']:.4f}")
    
    # Meilleur résultat
    best_mae = min(r['mae'] for r in results.values())
    
    logger.info("")
    if best_mae < 15:
        logger.info("🎉 OBJECTIF ATTEINT! MAE < 15nm")
    elif best_mae < 17:
        logger.info("✅ EXCELLENT! MAE < 17nm")
    elif best_mae < 18:
        logger.info("✅ TRÈS BON! MAE < 18nm")
    else:
        logger.info("👍 Résultats obtenus")
    
    logger.info("")
    logger.info("📁 Fichiers générés:")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/advanced_comparison.png")
    logger.info(f"   - {CONFIG.REPORTS_DIR}/advanced_importance_*.png")
    logger.info(f"   - {CONFIG.MODELS_DIR}/extra_trees_advanced_*.joblib")


if __name__ == "__main__":
    main()
