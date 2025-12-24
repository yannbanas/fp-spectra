#!/usr/bin/env python3

import os
import sys
import json
import re
import logging
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration."""
    
    FPBASE_JSON: Path = Path("data/raw/fpbase.json")
    STRUCTURES_DIR: Path = Path("data/structures")
    DATA_DIR: Path = Path("data/processed")
    LOGS_DIR: Path = Path("logs")
    
    CHROMOPHORE_CUTOFF: float = 5.0
    TEST_SIZE: float = 0.2
    RANDOM_SEED: int = 42
    
    CHROMOPHORE_PATTERNS: List[str] = field(default_factory=lambda: [
        "SYG", "TYG", "GYG", "AYG", "MYG", "CYG", "QYG",
        "DYG", "HYG", "FYG", "NYG", "VYG", "LYG", "IYG",
        "WYG", "RYG", "KYG", "EYG"
    ])
    
    HYDROPHOBIC: set = field(default_factory=lambda: {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'})
    POLAR: set = field(default_factory=lambda: {'S', 'T', 'N', 'Q', 'Y', 'C'})
    CHARGED_POS: set = field(default_factory=lambda: {'K', 'R', 'H'})
    CHARGED_NEG: set = field(default_factory=lambda: {'D', 'E'})
    AROMATIC: set = field(default_factory=lambda: {'F', 'Y', 'W', 'H'})


CONFIG = Config()


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = CONFIG.LOGS_DIR / f"extraction_{datetime.now():%Y%m%d_%H%M%S}.log"
    
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
    logger.info("🧬 EXTRACTION FEATURES - PROJET MASTER FP PREDICTION")
    logger.info("="*70)
    
    return logger


# ============================================================================
# LECTURE DU JSON FPBASE
# ============================================================================

def load_fpbase_json(json_file: Path, logger: logging.Logger) -> Dict[str, Dict]:
    """Charge fpbase.json."""
    
    if not json_file.exists():
        logger.error(f"❌ Fichier JSON non trouvé: {json_file}")
        return {}
    
    logger.info(f"📂 Chargement de {json_file}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"   {len(data)} entrées dans le JSON")
    
    proteins = {}
    n_with_spectra = 0
    
    for entry in data:
        slug = entry.get('slug', '').lower().strip()
        name = entry.get('name', slug)
        seq = entry.get('seq', '')
        
        if not slug:
            continue
        
        states = entry.get('states', [])
        ex_max, em_max, qy, ext_coeff, brightness = None, None, None, None, None
        
        if states:
            state = states[0]
            ex_max = state.get('ex_max')
            em_max = state.get('em_max')
            qy = state.get('qy')
            ext_coeff = state.get('ext_coeff')
            brightness = state.get('brightness')
        
        proteins[slug] = {
            'name': name,
            'slug': slug,
            'sequence': seq,
            'ex_max': ex_max,
            'em_max': em_max,
            'qy': qy,
            'ext_coeff': ext_coeff,
            'brightness': brightness,
        }
        
        if ex_max and em_max:
            n_with_spectra += 1
    
    logger.info(f"   ✅ {len(proteins)} protéines parsées")
    logger.info(f"   📊 {n_with_spectra} avec propriétés spectrales")
    
    return proteins


# ============================================================================
# SCAN DES STRUCTURES
# ============================================================================

def scan_structure_files(structures_dir: Path, logger: logging.Logger) -> Dict[str, Dict]:
    """
    Scanne les fichiers PDB à la racine.
    
    Format: NomProtéine_slug_exXXX_emYYY_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
    """
    
    if not structures_dir.exists():
        logger.error(f"❌ Dossier non trouvé: {structures_dir}")
        return {}
    
    # Pattern mis à jour pour le vrai format
    # Exemple: AcGFP1_acgfp1_ex475_em505_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
    pattern = re.compile(
        r'^(.+?)_(.+?)_ex(\d+)_em(\d+)_.+\.pdb$',
        re.IGNORECASE
    )
    
    structures = {}
    
    for filepath in structures_dir.iterdir():
        if not filepath.is_file():
            continue
        
        if filepath.suffix.lower() != '.pdb':
            continue
        
        filename = filepath.name
        
        match = pattern.match(filename)
        if match:
            display_name = match.group(1)
            slug = match.group(2).lower()
            ex_from_name = int(match.group(3))
            em_from_name = int(match.group(4))
            
            structures[slug] = {
                'pdb_file': filepath,
                'display_name': display_name,
                'ex_from_name': ex_from_name,
                'em_from_name': em_from_name,
            }
    
    logger.info(f"📂 {len(structures)} fichiers PDB trouvés et parsés")
    
    if structures:
        logger.info("   Exemples:")
        for i, (slug, info) in enumerate(list(structures.items())[:5]):
            logger.info(f"      {slug} → ex={info['ex_from_name']} em={info['em_from_name']}")
    
    return structures


def match_with_fpbase(proteins: Dict[str, Dict], 
                      structures: Dict[str, Dict],
                      logger: logging.Logger) -> List[Dict]:
    """Matche les protéines FPbase avec les structures par slug."""
    
    matches = []
    
    for slug, struct_info in structures.items():
        if slug in proteins:
            fp_info = proteins[slug]
            
            match = {
                **fp_info,
                'structure_file': struct_info['pdb_file'],
                'display_name': struct_info['display_name'],
            }
            
            # Utiliser les propriétés du nom de fichier si pas dans FPbase
            if not match.get('ex_max') and struct_info.get('ex_from_name'):
                match['ex_max'] = struct_info['ex_from_name']
            if not match.get('em_max') and struct_info.get('em_from_name'):
                match['em_max'] = struct_info['em_from_name']
            
            matches.append(match)
    
    logger.info(f"🔗 {len(matches)} protéines matchées (FPbase ↔ structure)")
    
    n_with_spectra = sum(1 for m in matches if m.get('ex_max') and m.get('em_max'))
    logger.info(f"   ✅ {n_with_spectra} avec propriétés spectrales complètes")
    
    return matches


# ============================================================================
# PARSING PDB
# ============================================================================

def parse_pdb(filepath: Path) -> Tuple[Dict, str, Dict]:
    """Parse un fichier PDB."""
    
    AA_MAP = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    }
    
    atoms_by_residue = {}
    residue_info = {}
    b_factors = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        res_id = int(line[22:26])
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        b_factor = float(line[60:66]) if len(line) > 65 else 0.0
                        
                        coords = np.array([x, y, z])
                        
                        if res_id not in atoms_by_residue:
                            atoms_by_residue[res_id] = []
                            residue_info[res_id] = {'name': res_name}
                            b_factors[res_id] = []
                        
                        atoms_by_residue[res_id].append((atom_name, coords))
                        b_factors[res_id].append(b_factor)
                    except:
                        continue
    except:
        return {}, "", {}
    
    # Construire séquence et pLDDT
    sequence = ""
    for res_id in sorted(residue_info.keys()):
        res_name = residue_info[res_id]['name']
        sequence += AA_MAP.get(res_name, 'X')
        if res_id in b_factors and b_factors[res_id]:
            residue_info[res_id]['plddt'] = np.mean(b_factors[res_id])
    
    return atoms_by_residue, sequence, residue_info


# ============================================================================
# FONCTIONS GÉOMÉTRIQUES
# ============================================================================

def calc_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    return np.linalg.norm(c1 - c2)

def calc_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
    return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

def calc_rmsd_to_plane(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 0.0
    centered = coords - np.mean(coords, axis=0)
    _, _, vh = np.linalg.svd(centered)
    return np.sqrt(np.mean(np.dot(centered, vh[-1])**2))


# ============================================================================
# DÉTECTION CHROMOPHORE
# ============================================================================

def find_chromophore(sequence: str, atoms_by_residue: Dict) -> Dict:
    """Détecte le chromophore."""
    
    info = {'found': False, 'sequence': '', 'position': 0, 'residue_ids': [], 'atoms': {}, 'center': np.zeros(3)}
    sorted_ids = sorted(atoms_by_residue.keys())
    
    if not sorted_ids:
        return info
    
    # Chercher patterns connus
    for pattern in CONFIG.CHROMOPHORE_PATTERNS:
        pos = sequence.find(pattern)
        if pos != -1 and pos + 2 < len(sorted_ids):
            info['found'] = True
            info['sequence'] = pattern
            info['position'] = pos + 1
            info['residue_ids'] = [sorted_ids[pos], sorted_ids[pos+1], sorted_ids[pos+2]]
            break
    
    # Fallback
    if not info['found']:
        for i in range(max(0, 55), min(len(sequence)-1, 80)):
            if sequence[i] == 'Y' and i+1 < len(sequence) and sequence[i+1] == 'G' and i > 0:
                if i+1 < len(sorted_ids):
                    info['found'] = True
                    info['sequence'] = sequence[i-1:i+2]
                    info['position'] = i
                    info['residue_ids'] = [sorted_ids[i-1], sorted_ids[i], sorted_ids[i+1]]
                    break
    
    # Extraire atomes
    if info['found'] and len(info['residue_ids']) == 3:
        all_coords = []
        for idx, res_id in enumerate(info['residue_ids']):
            if res_id in atoms_by_residue:
                for atom_name, coords in atoms_by_residue[res_id]:
                    info['atoms'][f"res{idx}_{atom_name}"] = coords
                    all_coords.append(coords)
        if all_coords:
            info['center'] = np.mean(all_coords, axis=0)
    
    return info


# ============================================================================
# EXTRACTION FEATURES
# ============================================================================

def extract_features(atoms_by_residue: Dict, sequence: str, 
                     residue_info: Dict, chromophore: Dict) -> Dict[str, float]:
    """Extrait tous les features."""
    
    features = {}
    
    # ===== CHROMOPHORE =====
    features['chrom_found'] = 1 if chromophore['found'] else 0
    
    if chromophore['found']:
        features['chrom_position'] = chromophore['position']
        atoms = chromophore['atoms']
        
        # Angles dihédraux
        try:
            if all(k in atoms for k in ['res0_C', 'res1_N', 'res1_CA', 'res1_C']):
                features['chrom_phi_tyr'] = calc_dihedral(
                    atoms['res0_C'], atoms['res1_N'], atoms['res1_CA'], atoms['res1_C'])
        except: pass
        
        try:
            if all(k in atoms for k in ['res1_N', 'res1_CA', 'res1_C', 'res2_N']):
                features['chrom_psi_tyr'] = calc_dihedral(
                    atoms['res1_N'], atoms['res1_CA'], atoms['res1_C'], atoms['res2_N'])
        except: pass
        
        try:
            if all(k in atoms for k in ['res1_N', 'res1_CA', 'res1_CB', 'res1_CG']):
                features['chrom_chi1_tyr'] = calc_dihedral(
                    atoms['res1_N'], atoms['res1_CA'], atoms['res1_CB'], atoms['res1_CG'])
        except: pass
        
        # Planarité
        try:
            ring = ['res1_CG', 'res1_CD1', 'res1_CD2', 'res1_CE1', 'res1_CE2', 'res1_CZ']
            coords = [atoms[k] for k in ring if k in atoms]
            if len(coords) >= 4:
                features['chrom_ring_planarity'] = calc_rmsd_to_plane(np.array(coords))
        except: pass
        
        # pLDDT chromophore
        plddt = [residue_info[r]['plddt'] for r in chromophore['residue_ids'] 
                 if r in residue_info and 'plddt' in residue_info[r]]
        if plddt:
            features['chrom_plddt_mean'] = np.mean(plddt)
            features['chrom_plddt_min'] = np.min(plddt)
    
    # ===== ENVIRONNEMENT =====
    if chromophore['found']:
        center = chromophore['center']
        cutoff = CONFIG.CHROMOPHORE_CUTOFF
        
        neighbors = []
        for res_id in atoms_by_residue:
            if res_id in chromophore['residue_ids']:
                continue
            for _, coords in atoms_by_residue[res_id]:
                if calc_distance(coords, center) < cutoff:
                    neighbors.append(res_id)
                    break
        
        features['env_n_neighbors'] = len(neighbors)
        
        aa_map = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
                  'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
                  'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
        
        n_hydro = n_polar = n_pos = n_neg = n_arom = 0
        for res_id in neighbors:
            if res_id in residue_info:
                aa = aa_map.get(residue_info[res_id]['name'], 'X')
                if aa in CONFIG.HYDROPHOBIC: n_hydro += 1
                if aa in CONFIG.POLAR: n_polar += 1
                if aa in CONFIG.CHARGED_POS: n_pos += 1
                if aa in CONFIG.CHARGED_NEG: n_neg += 1
                if aa in CONFIG.AROMATIC: n_arom += 1
        
        total = max(len(neighbors), 1)
        features['env_n_hydrophobic'] = n_hydro
        features['env_n_polar'] = n_polar
        features['env_n_charged_pos'] = n_pos
        features['env_n_charged_neg'] = n_neg
        features['env_n_aromatic'] = n_arom
        features['env_ratio_hydrophobic'] = n_hydro / total
        features['env_ratio_polar'] = n_polar / total
        features['env_net_charge'] = n_pos - n_neg
        
        plddt_n = [residue_info[r]['plddt'] for r in neighbors 
                   if r in residue_info and 'plddt' in residue_info[r]]
        if plddt_n:
            features['env_plddt_mean'] = np.mean(plddt_n)
    
    # ===== GLOBAL =====
    ca_coords = []
    all_coords = []
    for res_id in atoms_by_residue:
        for atom_name, coords in atoms_by_residue[res_id]:
            all_coords.append(coords)
            if atom_name == 'CA':
                ca_coords.append(coords)
    
    all_coords = np.array(all_coords) if all_coords else np.zeros((1, 3))
    ca_coords = np.array(ca_coords) if ca_coords else np.zeros((1, 3))
    
    features['glob_n_residues'] = len(residue_info)
    features['glob_n_atoms'] = len(all_coords)
    
    if len(ca_coords) > 1:
        center = np.mean(ca_coords, axis=0)
        features['glob_radius_gyration'] = np.sqrt(np.mean(np.sum((ca_coords - center)**2, axis=1)))
    
    features['glob_extent_x'] = np.ptp(all_coords[:, 0])
    features['glob_extent_y'] = np.ptp(all_coords[:, 1])
    features['glob_extent_z'] = np.ptp(all_coords[:, 2])
    
    plddt_all = [residue_info[r]['plddt'] for r in residue_info if 'plddt' in residue_info[r]]
    if plddt_all:
        features['glob_plddt_mean'] = np.mean(plddt_all)
        features['glob_plddt_std'] = np.std(plddt_all)
        features['glob_plddt_min'] = np.min(plddt_all)
        features['glob_frac_plddt_70'] = np.mean(np.array(plddt_all) > 70)
        features['glob_frac_plddt_90'] = np.mean(np.array(plddt_all) > 90)
    
    # ===== SÉQUENCE =====
    n = max(len(sequence), 1)
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        features[f'seq_freq_{aa}'] = sequence.count(aa) / n
    
    features['seq_frac_hydrophobic'] = sum(sequence.count(a) for a in CONFIG.HYDROPHOBIC) / n
    features['seq_frac_polar'] = sum(sequence.count(a) for a in CONFIG.POLAR) / n
    features['seq_frac_aromatic'] = sum(sequence.count(a) for a in CONFIG.AROMATIC) / n
    features['seq_length'] = len(sequence)
    
    hydro = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'E':-3.5,'Q':-3.5,'G':-0.4,
             'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
             'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    features['seq_gravy'] = sum(hydro.get(aa, 0) for aa in sequence) / n
    
    return features


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger = setup_logging()
    
    # 1. Charger FPbase
    logger.info("")
    logger.info("📂 ÉTAPE 1: Chargement FPbase JSON")
    logger.info("-" * 50)
    proteins = load_fpbase_json(CONFIG.FPBASE_JSON, logger)
    if not proteins:
        return
    
    # 2. Scanner les PDB
    logger.info("")
    logger.info("📂 ÉTAPE 2: Scan des structures PDB")
    logger.info("-" * 50)
    structures = scan_structure_files(CONFIG.STRUCTURES_DIR, logger)
    if not structures:
        logger.error("❌ Aucun fichier PDB trouvé!")
        return
    
    # 3. Matcher
    logger.info("")
    logger.info("🔗 ÉTAPE 3: Matching FPbase ↔ Structures")
    logger.info("-" * 50)
    matches = match_with_fpbase(proteins, structures, logger)
    
    matches_valid = [m for m in matches if m.get('ex_max') and m.get('em_max')]
    logger.info(f"   📊 {len(matches_valid)} avec ex_max ET em_max")
    
    if not matches_valid:
        logger.error("Aucune protéine avec propriétés spectrales!")
        return
    
    # 4. Extraire features
    logger.info("")
    logger.info("🧬 ÉTAPE 4: Extraction des features")
    logger.info("-" * 50)
    
    results = []
    n_chrom = 0
    
    for i, match in enumerate(matches_valid, 1):
        if i % 100 == 0:
            logger.info(f"   Progression: {i}/{len(matches_valid)}")
        
        try:
            atoms, seq, res_info = parse_pdb(match['structure_file'])
            if not atoms:
                continue
            
            sequence = seq if seq else match.get('sequence', '')
            chrom = find_chromophore(sequence, atoms)
            feats = extract_features(atoms, sequence, res_info, chrom)
            
            result = {
                'protein_id': match['slug'],
                'name': match['name'],
                'ex_max': match['ex_max'],
                'em_max': match['em_max'],
                'qy': match.get('qy'),
                'stokes_shift': match['em_max'] - match['ex_max'],
                'ext_coeff': match.get('ext_coeff'),
                'brightness': match.get('brightness'),
                **feats
            }
            results.append(result)
            
            if feats.get('chrom_found', 0) == 1:
                n_chrom += 1
                
        except Exception as e:
            pass
    
    logger.info(f"   ✅ {len(results)} protéines traitées")
    logger.info(f"   🧬 {n_chrom} chromophores détectés ({100*n_chrom/max(len(results),1):.1f}%)")
    
    # 5. DataFrame et filtrage
    df = pd.DataFrame(results)
    
    mask = (
        (df['ex_max'] >= 300) & (df['ex_max'] <= 750) &
        (df['em_max'] >= 350) & (df['em_max'] <= 850) &
        (df['stokes_shift'] >= 0) & (df['stokes_shift'] <= 250)
    )
    df = df[mask].copy()
    logger.info(f"   Après filtrage: {len(df)} protéines")
    
    # 6. Split train/test
    logger.info("")
    logger.info("✂️ ÉTAPE 5: Split train/test (80/20)")
    logger.info("-" * 50)
    
    df['_bin'] = pd.cut(df['em_max'], bins=5, labels=False)
    np.random.seed(CONFIG.RANDOM_SEED)
    
    train_l, test_l = [], []
    for b in df['_bin'].dropna().unique():
        sub = df[df['_bin'] == b].sample(frac=1, random_state=CONFIG.RANDOM_SEED)
        n_test = max(1, int(len(sub) * CONFIG.TEST_SIZE))
        test_l.append(sub.iloc[:n_test])
        train_l.append(sub.iloc[n_test:])
    
    train_df = pd.concat(train_l).drop(columns=['_bin'])
    test_df = pd.concat(test_l).drop(columns=['_bin'])
    df = df.drop(columns=['_bin'])
    
    logger.info(f"   Train: {len(train_df)} | Test: {len(test_df)}")
    
    # 7. Sauvegarder
    logger.info("")
    logger.info("💾 ÉTAPE 6: Sauvegarde")
    logger.info("-" * 50)
    
    CONFIG.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(CONFIG.DATA_DIR / "dataset_final.csv", index=False)
    train_df.to_csv(CONFIG.DATA_DIR / "dataset_train.csv", index=False)
    test_df.to_csv(CONFIG.DATA_DIR / "dataset_test.csv", index=False)
    
    logger.info(f"   ✅ dataset_final.csv ({len(df)} lignes)")
    logger.info(f"   ✅ dataset_train.csv ({len(train_df)} lignes)")
    logger.info(f"   ✅ dataset_test.csv ({len(test_df)} lignes)")
    
    # 8. Stats
    n_features = len([c for c in df.columns if c.startswith(('chrom_', 'env_', 'glob_', 'seq_'))])
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RÉSUMÉ FINAL")
    logger.info("=" * 60)
    logger.info(f"   Protéines: {len(df)}")
    logger.info(f"   Features: {n_features}")
    logger.info(f"   λ excitation: {df['ex_max'].min():.0f} - {df['ex_max'].max():.0f} nm")
    logger.info(f"   λ émission: {df['em_max'].min():.0f} - {df['em_max'].max():.0f} nm")
    logger.info(f"   Avec QY: {df['qy'].notna().sum()}")
    logger.info("")
    logger.info("🚀 PROCHAINE ÉTAPE: python 05_train_baseline.py")


if __name__ == "__main__":
    main()