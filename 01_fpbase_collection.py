#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PROJET MASTER : Prediction des proprietes spectrales des proteines fluorescentes
PHASE 1 : Collecte et preparation des donnees FPbase
=============================================================================

Ce script telecharge, parse et filtre les donnees de la base FPbase
(https://www.fpbase.org/) pour creer un dataset de proteines fluorescentes
utilisable pour l'apprentissage automatique.

Auteur: BANAS Yann
Version: 1.0.0 (Windows compatible)
Date: Decembre 2025

Reference:
    Lambert, T. J. (2019). FPbase: a community-editable fluorescent protein 
    database. Nature Methods, 16(4), 277-278.

Usage:
    python 01_fpbase_collection.py [--output-dir OUTPUT_DIR] [--no-download]
    python 01_fpbase_collection.py --input-file fpbase.json

Sortie:
    - data/raw/fpbase_raw.json           : Donnees brutes de l'API
    - data/processed/fpbase_curated.csv  : Dataset filtre et nettoye
    - data/processed/fpbase_stats.json   : Statistiques du dataset
    - data/processed/dataset_overview.png : Visualisation

=============================================================================
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

# Optionnel: tqdm pour la barre de progression
try:
    from tqdm import tqdm
except ImportError:
    # Fallback si tqdm n'est pas installe
    def tqdm(iterable, desc="", **kwargs):
        print(f"   {desc}...")
        return iterable

# =============================================================================
# CONFIGURATION
# =============================================================================

# API FPbase
FPBASE_API_URL = "https://www.fpbase.org/api/proteins/"
FPBASE_STATES_URL = "https://www.fpbase.org/api/states/"
API_TIMEOUT = 60  # secondes
API_DELAY = 0.5   # delai entre requetes (respecter le rate limiting)

# Criteres de filtrage scientifiques
FILTER_CRITERIA = {
    # Proprietes spectrales
    'min_excitation_nm': 300,
    'max_excitation_nm': 800,
    'min_emission_nm': 400,
    'max_emission_nm': 900,
    'min_stokes_shift_nm': 5,      # Physiquement realiste
    'max_stokes_shift_nm': 200,    # Exclure les outliers
    
    # Sequence
    'min_sequence_length': 100,    # Exclure fragments
    'max_sequence_length': 500,    # Exclure fusions
    
    # Qualite des donnees
    'require_excitation': True,
    'require_emission': True,
    'require_sequence': True,
    
    # Exclusions
    'exclude_cofactors': [
        'BV',      # Biliverdine (bacteriophytochromes)
        'PCB',     # Phycocyanobiline
        'PEB',     # Phycoerythrobiline
        'FAD',     # Flavin adenine dinucleotide
        'FMN',     # Flavin mononucleotide
    ],
}

# Logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SpectralState:
    """Represente un etat spectral d'une proteine fluorescente."""
    state_id: int
    state_name: str
    ex_max: Optional[float]      # nm
    em_max: Optional[float]      # nm
    ext_coeff: Optional[float]   # M-1 cm-1
    qy: Optional[float]          # 0-1
    brightness: Optional[float]  # ext_coeff x qy / 1000
    lifetime: Optional[float]    # ns
    pka: Optional[float]
    maturation: Optional[float]  # minutes
    is_dark: bool = False


@dataclass
class FluorescentProtein:
    """Represente une proteine fluorescente complete."""
    # Identifiants
    uuid: str
    name: str
    slug: str
    
    # Sequence
    seq: str
    seq_length: int
    
    # Etats spectraux
    states: List[SpectralState]
    default_state: Optional[SpectralState]
    
    # Proprietes du state par defaut (pour faciliter l'analyse)
    ex_max: Optional[float]
    em_max: Optional[float]
    ext_coeff: Optional[float]
    qy: Optional[float]
    brightness: Optional[float]
    stokes_shift: Optional[float]
    
    # Metadonnees
    chromophore: Optional[str]
    cofactor: Optional[str]
    switch_type: Optional[str]
    oligomerization: Optional[str]
    
    # References structures
    pdb_ids: List[str]
    uniprot_id: Optional[str]
    genbank_id: Optional[str]
    
    # Origine
    parent_organism: Optional[str]
    
    # Qualite
    has_complete_spectrum: bool
    data_quality_score: float


# =============================================================================
# FONCTIONS DE TELECHARGEMENT
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure le logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('fpbase_collector')
    logger.setLevel(logging.DEBUG)
    
    # Handler fichier (UTF-8)
    fh = logging.FileHandler(log_dir / 'collection.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    
    # Handler console (ASCII safe)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def download_fpbase_data(logger: logging.Logger) -> List[Dict]:
    """
    Telecharge toutes les proteines depuis l'API FPbase.
    
    L'API est paginee, cette fonction gere automatiquement la pagination.
    
    Returns:
        Liste des proteines (dictionnaires bruts de l'API)
    """
    logger.info("=" * 60)
    logger.info("[DOWNLOAD] TELECHARGEMENT DEPUIS FPBASE")
    logger.info("=" * 60)
    
    all_proteins = []
    url = FPBASE_API_URL + "?format=json"
    page = 1
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'FP-Predict-ML-Project/1.0 (Academic Research)',
        'Accept': 'application/json',
    })
    
    while url:
        logger.info(f"   Telechargement page {page}...")
        
        try:
            response = session.get(url, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # L'API retourne soit une liste, soit un dict avec 'results'
            if isinstance(data, list):
                all_proteins.extend(data)
                url = None  # Pas de pagination
            elif isinstance(data, dict):
                results = data.get('results', [])
                all_proteins.extend(results)
                url = data.get('next')  # URL de la page suivante
                logger.debug(f"      {len(results)} proteines recuperees")
            else:
                logger.error(f"Format de reponse inattendu: {type(data)}")
                break
            
            page += 1
            time.sleep(API_DELAY)
            
        except requests.exceptions.Timeout:
            logger.error(f"   Timeout sur la page {page}")
            break
        except requests.exceptions.HTTPError as e:
            logger.error(f"   Erreur HTTP: {e}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"   Erreur de connexion: {e}")
            break
        except json.JSONDecodeError as e:
            logger.error(f"   Erreur JSON: {e}")
            break
    
    logger.info(f"[OK] {len(all_proteins)} proteines telechargees")
    
    return all_proteins


def load_local_data(filepath: Path, logger: logging.Logger) -> List[Dict]:
    """Charge les donnees depuis un fichier local."""
    logger.info(f"[LOAD] Chargement depuis {filepath}")
    
    # IMPORTANT: Specifier encoding UTF-8 pour Windows
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        proteins = data['results']
    elif isinstance(data, list):
        proteins = data
    else:
        raise ValueError(f"Format de donnees non reconnu")
    
    logger.info(f"   [OK] {len(proteins)} proteines chargees")
    
    return proteins


# =============================================================================
# FONCTIONS DE PARSING
# =============================================================================

def parse_spectral_state(state_data: Dict) -> SpectralState:
    """Parse un etat spectral depuis les donnees brutes."""
    return SpectralState(
        state_id=state_data.get('id', 0),
        state_name=state_data.get('name', 'default'),
        ex_max=safe_float(state_data.get('ex_max')),
        em_max=safe_float(state_data.get('em_max')),
        ext_coeff=safe_float(state_data.get('ext_coeff')),
        qy=safe_float(state_data.get('qy')),
        brightness=safe_float(state_data.get('brightness')),
        lifetime=safe_float(state_data.get('lifetime')),
        pka=safe_float(state_data.get('pka')),
        maturation=safe_float(state_data.get('maturation')),
        is_dark=state_data.get('is_dark', False),
    )


def safe_float(value) -> Optional[float]:
    """Convertit une valeur en float de maniere sure."""
    if value is None:
        return None
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def safe_str(value) -> Optional[str]:
    """Convertit une valeur en string de maniere sure."""
    if value is None or value == '':
        return None
    return str(value).strip()


def calculate_data_quality_score(protein: Dict, states: List[SpectralState]) -> float:
    """
    Calcule un score de qualite des donnees (0-1).
    
    Criteres:
    - Presence de ex_max et em_max (+0.3)
    - Presence de QY (+0.2)
    - Presence de ext_coeff (+0.1)
    - Presence de sequence complete (+0.2)
    - Presence de structure PDB (+0.1)
    - Presence de UniProt ID (+0.1)
    """
    score = 0.0
    
    # Spectre complet
    if states:
        default = states[0]
        if default.ex_max is not None and default.em_max is not None:
            score += 0.3
        if default.qy is not None:
            score += 0.2
        if default.ext_coeff is not None:
            score += 0.1
    
    # Sequence
    seq = protein.get('seq', '')
    if seq and len(seq) >= 100:
        score += 0.2
    
    # Structures
    pdb_ids = protein.get('pdb', [])
    if pdb_ids and len(pdb_ids) > 0:
        score += 0.1
    
    # UniProt
    if protein.get('uniprot'):
        score += 0.1
    
    return min(score, 1.0)


def parse_protein(protein_data: Dict) -> FluorescentProtein:
    """
    Parse une proteine fluorescente depuis les donnees brutes de l'API.
    
    Args:
        protein_data: Dictionnaire brut de l'API FPbase
        
    Returns:
        FluorescentProtein: Objet structure
    """
    # Parser les etats spectraux
    states_data = protein_data.get('states', [])
    states = [parse_spectral_state(s) for s in states_data]
    
    # Trouver l'etat par defaut (le premier non-dark, ou le premier tout court)
    default_state = None
    for state in states:
        if not state.is_dark:
            default_state = state
            break
    if default_state is None and states:
        default_state = states[0]
    
    # Extraire les proprietes spectrales de l'etat par defaut
    ex_max = default_state.ex_max if default_state else None
    em_max = default_state.em_max if default_state else None
    ext_coeff = default_state.ext_coeff if default_state else None
    qy = default_state.qy if default_state else None
    brightness = default_state.brightness if default_state else None
    
    # Calculer le Stokes shift
    stokes_shift = None
    if ex_max is not None and em_max is not None:
        stokes_shift = em_max - ex_max
    
    # Extraire la sequence
    seq = safe_str(protein_data.get('seq')) or ''
    
    # Extraire les PDB IDs
    pdb_ids = protein_data.get('pdb', [])
    if pdb_ids is None:
        pdb_ids = []
    elif isinstance(pdb_ids, str):
        pdb_ids = [pdb_ids] if pdb_ids else []
    
    # Score de qualite
    quality_score = calculate_data_quality_score(protein_data, states)
    
    # Verifier si le spectre est complet
    has_complete = (ex_max is not None and em_max is not None)
    
    return FluorescentProtein(
        uuid=protein_data.get('uuid', ''),
        name=protein_data.get('name', 'Unknown'),
        slug=protein_data.get('slug', ''),
        
        seq=seq,
        seq_length=len(seq),
        
        states=states,
        default_state=default_state,
        
        ex_max=ex_max,
        em_max=em_max,
        ext_coeff=ext_coeff,
        qy=qy,
        brightness=brightness,
        stokes_shift=stokes_shift,
        
        chromophore=safe_str(protein_data.get('chromophore')),
        cofactor=safe_str(protein_data.get('cofactor')),
        switch_type=safe_str(protein_data.get('switch_type')),
        oligomerization=safe_str(protein_data.get('agg')),
        
        pdb_ids=pdb_ids,
        uniprot_id=safe_str(protein_data.get('uniprot')),
        genbank_id=safe_str(protein_data.get('genbank')),
        
        parent_organism=safe_str(protein_data.get('parent_organism')),
        
        has_complete_spectrum=has_complete,
        data_quality_score=quality_score,
    )


def parse_all_proteins(raw_data: List[Dict], logger: logging.Logger) -> List[FluorescentProtein]:
    """Parse toutes les proteines."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("[PARSE] PARSING DES DONNEES")
    logger.info("=" * 60)
    
    proteins = []
    errors = 0
    
    for item in tqdm(raw_data, desc="   Parsing"):
        try:
            fp = parse_protein(item)
            proteins.append(fp)
        except Exception as e:
            errors += 1
            logger.debug(f"   Erreur parsing {item.get('name', '?')}: {e}")
    
    logger.info(f"   [OK] {len(proteins)} proteines parsees")
    if errors > 0:
        logger.warning(f"   [WARN] {errors} erreurs de parsing")
    
    return proteins


# =============================================================================
# FONCTIONS DE FILTRAGE
# =============================================================================

@dataclass
class FilterStats:
    """Statistiques de filtrage."""
    initial_count: int
    after_spectrum: int
    after_sequence: int
    after_cofactor: int
    after_spectral_range: int
    after_stokes: int
    final_count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


def filter_proteins(
    proteins: List[FluorescentProtein],
    criteria: Dict,
    logger: logging.Logger
) -> Tuple[List[FluorescentProtein], FilterStats]:
    """
    Filtre les proteines selon les criteres scientifiques.
    
    Args:
        proteins: Liste des proteines parsees
        criteria: Dictionnaire des criteres de filtrage
        logger: Logger
        
    Returns:
        Tuple (proteines filtrees, statistiques)
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("[FILTER] FILTRAGE SCIENTIFIQUE")
    logger.info("=" * 60)
    
    initial_count = len(proteins)
    filtered = proteins.copy()
    
    # 1. Filtre spectral (ex_max ET em_max requis)
    if criteria.get('require_excitation') and criteria.get('require_emission'):
        filtered = [p for p in filtered if p.has_complete_spectrum]
        logger.info(f"   Apres filtre spectre complet: {len(filtered)}/{initial_count}")
    after_spectrum = len(filtered)
    
    # 2. Filtre sequence
    if criteria.get('require_sequence'):
        min_len = criteria.get('min_sequence_length', 0)
        max_len = criteria.get('max_sequence_length', 10000)
        filtered = [p for p in filtered 
                   if p.seq and min_len <= p.seq_length <= max_len]
        logger.info(f"   Apres filtre sequence ({min_len}-{max_len} AA): {len(filtered)}")
    after_sequence = len(filtered)
    
    # 3. Filtre cofacteurs
    exclude_cofactors = criteria.get('exclude_cofactors', [])
    if exclude_cofactors:
        def has_excluded_cofactor(p):
            if p.cofactor is None:
                return False
            return any(cof in p.cofactor.upper() for cof in exclude_cofactors)
        
        n_before = len(filtered)
        filtered = [p for p in filtered if not has_excluded_cofactor(p)]
        n_excluded = n_before - len(filtered)
        logger.info(f"   Apres exclusion cofacteurs ({exclude_cofactors}): {len(filtered)} (-{n_excluded})")
    after_cofactor = len(filtered)
    
    # 4. Filtre range spectral
    min_ex = criteria.get('min_excitation_nm', 0)
    max_ex = criteria.get('max_excitation_nm', 1000)
    min_em = criteria.get('min_emission_nm', 0)
    max_em = criteria.get('max_emission_nm', 1000)
    
    filtered = [p for p in filtered 
               if (p.ex_max is not None and min_ex <= p.ex_max <= max_ex) and
                  (p.em_max is not None and min_em <= p.em_max <= max_em)]
    logger.info(f"   Apres filtre range spectral: {len(filtered)}")
    logger.info(f"      Ex: {min_ex}-{max_ex} nm, Em: {min_em}-{max_em} nm")
    after_spectral_range = len(filtered)
    
    # 5. Filtre Stokes shift (physiquement realiste)
    min_stokes = criteria.get('min_stokes_shift_nm', 0)
    max_stokes = criteria.get('max_stokes_shift_nm', 500)
    
    filtered = [p for p in filtered 
               if p.stokes_shift is not None and 
                  min_stokes <= p.stokes_shift <= max_stokes]
    logger.info(f"   Apres filtre Stokes shift ({min_stokes}-{max_stokes} nm): {len(filtered)}")
    after_stokes = len(filtered)
    
    # Resume
    final_count = len(filtered)
    logger.info(f"")
    logger.info(f"   [OK] Dataset final: {final_count} proteines")
    logger.info(f"        Taux de retention: {100*final_count/initial_count:.1f}%")
    
    stats = FilterStats(
        initial_count=initial_count,
        after_spectrum=after_spectrum,
        after_sequence=after_sequence,
        after_cofactor=after_cofactor,
        after_spectral_range=after_spectral_range,
        after_stokes=after_stokes,
        final_count=final_count,
    )
    
    return filtered, stats


# =============================================================================
# FONCTIONS D'EXPORT
# =============================================================================

def proteins_to_dataframe(proteins: List[FluorescentProtein]) -> pd.DataFrame:
    """Convertit la liste de proteines en DataFrame."""
    records = []
    
    for p in proteins:
        record = {
            # Identifiants
            'uuid': p.uuid,
            'name': p.name,
            'slug': p.slug,
            
            # Sequence
            'sequence': p.seq,
            'seq_length': p.seq_length,
            
            # Proprietes spectrales
            'ex_max': p.ex_max,
            'em_max': p.em_max,
            'stokes_shift': p.stokes_shift,
            'ext_coeff': p.ext_coeff,
            'qy': p.qy,
            'brightness': p.brightness,
            
            # Metadonnees
            'chromophore': p.chromophore,
            'cofactor': p.cofactor,
            'switch_type': p.switch_type,
            'oligomerization': p.oligomerization,
            'parent_organism': p.parent_organism,
            
            # References
            'pdb_ids': ','.join(p.pdb_ids) if p.pdb_ids else None,
            'uniprot_id': p.uniprot_id,
            'genbank_id': p.genbank_id,
            
            # Qualite
            'n_states': len(p.states),
            'data_quality_score': p.data_quality_score,
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Trier par nom
    df = df.sort_values('name').reset_index(drop=True)
    
    return df


def calculate_dataset_statistics(df: pd.DataFrame) -> Dict:
    """Calcule les statistiques descriptives du dataset."""
    stats = {
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'n_proteins': len(df),
            'script_version': '1.1.0',
        },
        'spectral_properties': {
            'excitation': {
                'min': float(df['ex_max'].min()),
                'max': float(df['ex_max'].max()),
                'mean': float(df['ex_max'].mean()),
                'median': float(df['ex_max'].median()),
                'std': float(df['ex_max'].std()),
            },
            'emission': {
                'min': float(df['em_max'].min()),
                'max': float(df['em_max'].max()),
                'mean': float(df['em_max'].mean()),
                'median': float(df['em_max'].median()),
                'std': float(df['em_max'].std()),
            },
            'stokes_shift': {
                'min': float(df['stokes_shift'].min()),
                'max': float(df['stokes_shift'].max()),
                'mean': float(df['stokes_shift'].mean()),
                'median': float(df['stokes_shift'].median()),
            },
        },
        'quantum_yield': {
            'n_documented': int(df['qy'].notna().sum()),
            'percentage': float(100 * df['qy'].notna().mean()),
            'mean': float(df['qy'].mean()) if df['qy'].notna().any() else None,
            'median': float(df['qy'].median()) if df['qy'].notna().any() else None,
        },
        'extinction_coefficient': {
            'n_documented': int(df['ext_coeff'].notna().sum()),
            'percentage': float(100 * df['ext_coeff'].notna().mean()),
        },
        'structural_data': {
            'n_with_pdb': int(df['pdb_ids'].notna().sum()),
            'n_with_uniprot': int(df['uniprot_id'].notna().sum()),
            'percentage_pdb': float(100 * df['pdb_ids'].notna().mean()),
            'percentage_uniprot': float(100 * df['uniprot_id'].notna().mean()),
        },
        'sequence': {
            'mean_length': float(df['seq_length'].mean()),
            'min_length': int(df['seq_length'].min()),
            'max_length': int(df['seq_length'].max()),
        },
        'color_distribution': {},
    }
    
    # Distribution des couleurs
    def classify_color(em):
        if em < 480:
            return 'Blue'
        elif em < 520:
            return 'Cyan'
        elif em < 560:
            return 'Green'
        elif em < 600:
            return 'Yellow/Orange'
        elif em < 650:
            return 'Red'
        else:
            return 'Far-Red/NIR'
    
    df['color_class'] = df['em_max'].apply(classify_color)
    color_counts = df['color_class'].value_counts().to_dict()
    stats['color_distribution'] = color_counts
    
    return stats


def create_visualization(df: pd.DataFrame, output_path: Path, logger: logging.Logger):
    """Cree une visualisation du dataset."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend sans GUI pour Windows
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("   [WARN] matplotlib non installe, pas de visualisation")
        return
    
    logger.info("")
    logger.info("[VIZ] Creation de la visualisation...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Couleurs par classe
    color_map = {
        'Blue': '#0066CC',
        'Cyan': '#00CCCC',
        'Green': '#00CC66',
        'Yellow/Orange': '#FFAA00',
        'Red': '#CC3300',
        'Far-Red/NIR': '#990066',
    }
    
    def classify_color(em):
        if em < 480: return 'Blue'
        elif em < 520: return 'Cyan'
        elif em < 560: return 'Green'
        elif em < 600: return 'Yellow/Orange'
        elif em < 650: return 'Red'
        else: return 'Far-Red/NIR'
    
    df['color_class'] = df['em_max'].apply(classify_color)
    
    # 1. Distribution excitation
    ax1 = axes[0, 0]
    ax1.hist(df['ex_max'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Excitation (nm)', fontsize=11)
    ax1.set_ylabel('Nombre de FPs', fontsize=11)
    ax1.set_title('Distribution excitation', fontsize=12, fontweight='bold')
    ax1.axvline(df['ex_max'].median(), color='red', linestyle='--', 
                label=f'Mediane = {df["ex_max"].median():.0f} nm')
    ax1.legend()
    
    # 2. Distribution emission
    ax2 = axes[0, 1]
    ax2.hist(df['em_max'], bins=30, color='forestgreen', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Emission (nm)', fontsize=11)
    ax2.set_ylabel('Nombre de FPs', fontsize=11)
    ax2.set_title('Distribution emission', fontsize=12, fontweight='bold')
    ax2.axvline(df['em_max'].median(), color='red', linestyle='--',
                label=f'Mediane = {df["em_max"].median():.0f} nm')
    ax2.legend()
    
    # 3. Excitation vs Emission (scatter colore)
    ax3 = axes[0, 2]
    for color_class, color in color_map.items():
        mask = df['color_class'] == color_class
        if mask.any():
            ax3.scatter(df.loc[mask, 'ex_max'], df.loc[mask, 'em_max'],
                       c=color, label=color_class, alpha=0.7, s=30, edgecolor='white', linewidth=0.5)
    ax3.plot([350, 700], [350, 700], 'k--', alpha=0.3, label='y = x')
    ax3.set_xlabel('Excitation (nm)', fontsize=11)
    ax3.set_ylabel('Emission (nm)', fontsize=11)
    ax3.set_title('Excitation vs Emission', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_xlim([350, 700])
    ax3.set_ylim([400, 800])
    
    # 4. Distribution Stokes shift
    ax4 = axes[1, 0]
    ax4.hist(df['stokes_shift'], bins=30, color='purple', edgecolor='white', alpha=0.8)
    ax4.set_xlabel('Stokes shift (nm)', fontsize=11)
    ax4.set_ylabel('Nombre de FPs', fontsize=11)
    ax4.set_title('Distribution Stokes shift', fontsize=12, fontweight='bold')
    ax4.axvline(df['stokes_shift'].median(), color='red', linestyle='--',
                label=f'Mediane = {df["stokes_shift"].median():.0f} nm')
    ax4.legend()
    
    # 5. Rendement quantique
    ax5 = axes[1, 1]
    qy_valid = df['qy'].dropna()
    if len(qy_valid) > 0:
        ax5.hist(qy_valid, bins=20, color='coral', edgecolor='white', alpha=0.8)
        ax5.set_xlabel('Rendement quantique (QY)', fontsize=11)
        ax5.set_ylabel('Nombre de FPs', fontsize=11)
        ax5.set_title(f'Distribution QY (n={len(qy_valid)})', fontsize=12, fontweight='bold')
        ax5.axvline(qy_valid.median(), color='red', linestyle='--',
                    label=f'Mediane = {qy_valid.median():.2f}')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'Pas de donnees QY', ha='center', va='center', fontsize=12)
        ax5.set_title('Rendement quantique', fontsize=12, fontweight='bold')
    
    # 6. Distribution par couleur (pie chart)
    ax6 = axes[1, 2]
    color_counts = df['color_class'].value_counts()
    wedge_colors = [color_map.get(c, '#888888') for c in color_counts.index]
    wedges, texts, autotexts = ax6.pie(
        color_counts.values,
        labels=color_counts.index,
        colors=wedge_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(color_counts),
    )
    ax6.set_title('Distribution par couleur', fontsize=12, fontweight='bold')
    
    # Titre general
    fig.suptitle(f'Dataset FPbase : {len(df)} proteines fluorescentes',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"   [OK] Sauvegarde: {output_path}")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """Fonction principale."""
    # Arguments
    parser = argparse.ArgumentParser(
        description='Collecte et preparation des donnees FPbase'
    )
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Repertoire de sortie')
    parser.add_argument('--no-download', action='store_true',
                       help='Utiliser les donnees locales existantes')
    parser.add_argument('--input-file', type=str, default=None,
                       help='Fichier JSON local a utiliser')
    args = parser.parse_args()
    
    # Creer les repertoires
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / 'raw'
    processed_dir = output_dir / 'processed'
    log_dir = output_dir / 'logs'
    
    for d in [raw_dir, processed_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = setup_logging(log_dir)
    
    logger.info("=" * 70)
    logger.info("[START] PROJET MASTER : Collecte de donnees FPbase")
    logger.info("=" * 70)
    logger.info(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Output: {output_dir.absolute()}")
    
    # ==========================================================================
    # ETAPE 1 : Telechargement ou chargement
    # ==========================================================================
    
    raw_file = raw_dir / 'fpbase_raw.json'
    
    if args.input_file:
        raw_data = load_local_data(Path(args.input_file), logger)
    elif args.no_download and raw_file.exists():
        raw_data = load_local_data(raw_file, logger)
    else:
        raw_data = download_fpbase_data(logger)
        
        # Sauvegarder les donnees brutes
        if raw_data:
            with open(raw_file, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2, ensure_ascii=False)
            logger.info(f"   [SAVE] Donnees brutes sauvegardees: {raw_file}")
    
    if not raw_data:
        logger.error("[ERROR] Aucune donnee recuperee. Arret.")
        sys.exit(1)
    
    # ==========================================================================
    # ETAPE 2 : Parsing
    # ==========================================================================
    
    proteins = parse_all_proteins(raw_data, logger)
    
    # ==========================================================================
    # ETAPE 3 : Filtrage
    # ==========================================================================
    
    filtered_proteins, filter_stats = filter_proteins(
        proteins, FILTER_CRITERIA, logger
    )
    
    # ==========================================================================
    # ETAPE 4 : Export
    # ==========================================================================
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("[EXPORT] EXPORT DES DONNEES")
    logger.info("=" * 60)
    
    # DataFrame
    df = proteins_to_dataframe(filtered_proteins)
    
    # CSV (UTF-8 avec BOM pour Excel Windows)
    csv_file = processed_dir / 'fpbase_curated.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    logger.info(f"   [OK] Dataset CSV: {csv_file}")
    
    # Statistiques
    stats = calculate_dataset_statistics(df)
    stats['filter_stats'] = filter_stats.to_dict()
    
    stats_file = processed_dir / 'fpbase_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"   [OK] Statistiques: {stats_file}")
    
    # Visualisation
    viz_file = processed_dir / 'dataset_overview.png'
    create_visualization(df, viz_file, logger)
    
    # ==========================================================================
    # RESUME FINAL
    # ==========================================================================
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[SUMMARY] RESUME DU DATASET")
    logger.info("=" * 70)
    
    logger.info(f"")
    logger.info(f"   Nombre de proteines: {len(df)}")
    
    logger.info(f"")
    logger.info(f"   Proprietes spectrales:")
    logger.info(f"      Excitation: {df['ex_max'].min():.0f} - {df['ex_max'].max():.0f} nm "
               f"(mediane: {df['ex_max'].median():.0f} nm)")
    logger.info(f"      Emission:   {df['em_max'].min():.0f} - {df['em_max'].max():.0f} nm "
               f"(mediane: {df['em_max'].median():.0f} nm)")
    logger.info(f"      Stokes:     {df['stokes_shift'].min():.0f} - {df['stokes_shift'].max():.0f} nm "
               f"(mediane: {df['stokes_shift'].median():.0f} nm)")
    
    qy_count = df['qy'].notna().sum()
    logger.info(f"")
    logger.info(f"   Rendement quantique:")
    logger.info(f"      Documente: {qy_count}/{len(df)} ({100*qy_count/len(df):.1f}%)")
    
    pdb_count = df['pdb_ids'].notna().sum()
    uniprot_count = df['uniprot_id'].notna().sum()
    logger.info(f"")
    logger.info(f"   Donnees structurales:")
    logger.info(f"      Avec PDB: {pdb_count} ({100*pdb_count/len(df):.1f}%)")
    logger.info(f"      Avec UniProt: {uniprot_count} ({100*uniprot_count/len(df):.1f}%)")
    
    logger.info(f"")
    logger.info(f"   Distribution des couleurs:")
    for color, count in stats['color_distribution'].items():
        bar = "#" * int(count / len(df) * 30)
        logger.info(f"      {color:15s}: {count:3d} {bar}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("[DONE] PHASE 1 TERMINEE AVEC SUCCES")
    logger.info("=" * 70)
    
    logger.info(f"""
    Fichiers generes:
    - {raw_file}
    - {csv_file}
    - {stats_file}
    - {viz_file}
    
    Prochaine etape:
    -> Telecharger les structures AlphaFold (script 02_download_structures.py)
    -> Ou passer a l'extraction de features (script 03_extract_features.py)
    """)
    
    return 0


# =============================================================================
# POINT D'ENTREE
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())