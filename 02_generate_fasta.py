#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PROJET MASTER : Prediction des proprietes spectrales des proteines fluorescentes
PHASE 1b : Generation des fichiers FASTA pour ColabFold
=============================================================================

Ce script:
1. Lit le dataset fpbase_curated.csv
2. Extrait les proteines SANS UniProt ID (pas de structure AlphaFold disponible)
3. Genere des fichiers FASTA par lots de 50 pour ColabFold

Pourquoi 50 par fichier ?
- ColabFold a des limites de memoire GPU
- 50 sequences est un bon compromis temps/efficacite
- Permet de paralleliser sur plusieurs sessions Colab

Auteur: BANAS Yann
Version: 1.0.0 (Windows compatible)
Date: Decembre 2025

Usage:
    python 02_generate_fasta.py
    python 02_generate_fasta.py --batch-size 100
    python 02_generate_fasta.py --input data/processed/fpbase_curated.csv

=============================================================================
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import math

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT = Path("data/processed/fpbase_curated.csv")
DEFAULT_OUTPUT_DIR = Path("data/fasta")
DEFAULT_BATCH_SIZE = 50


# =============================================================================
# FONCTIONS
# =============================================================================

def load_dataset(filepath: Path) -> pd.DataFrame:
    """Charge le dataset CSV."""
    print(f"[LOAD] Chargement de {filepath}")
    
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    print(f"       Total: {len(df)} proteines")
    
    return df


def filter_without_uniprot(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les proteines sans UniProt ID."""
    
    # Les proteines sans UniProt (NaN ou vide)
    mask_no_uniprot = df['uniprot_id'].isna() | (df['uniprot_id'] == '')
    
    df_no_uniprot = df[mask_no_uniprot].copy()
    df_with_uniprot = df[~mask_no_uniprot].copy()
    
    print(f"[FILTER] Proteines SANS UniProt: {len(df_no_uniprot)}")
    print(f"         Proteines AVEC UniProt: {len(df_with_uniprot)} (structure AlphaFold disponible)")
    
    return df_no_uniprot


def validate_sequence(sequence: str, name: str) -> Tuple[bool, str]:
    """
    Valide une sequence proteique pour FASTA.
    
    Returns:
        (is_valid, cleaned_sequence)
    """
    if not sequence or pd.isna(sequence):
        return False, ""
    
    # Nettoyer la sequence
    seq = str(sequence).upper().strip()
    
    # Enlever les espaces et retours a la ligne
    seq = seq.replace(" ", "").replace("\n", "").replace("\r", "")
    
    # Verifier que c'est bien des acides amines
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX*")
    invalid_chars = set(seq) - valid_aa
    
    if invalid_chars:
        print(f"   [WARN] {name}: caracteres invalides {invalid_chars}")
        # Enlever les caracteres invalides
        seq = ''.join(c for c in seq if c in valid_aa)
    
    # Verifier la longueur minimale
    if len(seq) < 50:
        return False, ""
    
    return True, seq


def create_fasta_content(proteins: List[dict]) -> str:
    """
    Cree le contenu d'un fichier FASTA.
    
    Format:
    >protein_name
    SEQUENCE...
    """
    lines = []
    
    for p in proteins:
        # Header: >nom|slug|ex_max|em_max
        header = f">{p['name']}|{p['slug']}|ex{p['ex_max']:.0f}|em{p['em_max']:.0f}"
        lines.append(header)
        
        # Sequence (80 caracteres par ligne, standard FASTA)
        seq = p['sequence']
        for i in range(0, len(seq), 80):
            lines.append(seq[i:i+80])
    
    return '\n'.join(lines)


def generate_fasta_batches(
    df: pd.DataFrame, 
    output_dir: Path, 
    batch_size: int = 50
) -> List[Path]:
    """
    Genere des fichiers FASTA par lots.
    
    Args:
        df: DataFrame avec colonnes 'name', 'slug', 'sequence', 'ex_max', 'em_max'
        output_dir: Repertoire de sortie
        batch_size: Nombre de sequences par fichier
        
    Returns:
        Liste des fichiers generes
    """
    
    print(f"\n[FASTA] Generation des fichiers FASTA (batch_size={batch_size})")
    
    # Creer le repertoire
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Valider et preparer les sequences
    valid_proteins = []
    invalid_count = 0
    
    for _, row in df.iterrows():
        is_valid, clean_seq = validate_sequence(row['sequence'], row['name'])
        
        if is_valid:
            valid_proteins.append({
                'name': row['name'],
                'slug': row['slug'],
                'sequence': clean_seq,
                'ex_max': row['ex_max'],
                'em_max': row['em_max'],
            })
        else:
            invalid_count += 1
    
    print(f"        Sequences valides: {len(valid_proteins)}")
    if invalid_count > 0:
        print(f"        Sequences invalides: {invalid_count}")
    
    # Calculer le nombre de batches
    n_batches = math.ceil(len(valid_proteins) / batch_size)
    print(f"        Nombre de fichiers: {n_batches}")
    
    # Generer les fichiers
    generated_files = []
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(valid_proteins))
        
        batch_proteins = valid_proteins[start:end]
        
        # Nom du fichier: batch_001_050.fasta
        filename = f"batch_{start+1:03d}_{end:03d}.fasta"
        filepath = output_dir / filename
        
        # Creer le contenu
        content = create_fasta_content(batch_proteins)
        
        # Ecrire le fichier
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        generated_files.append(filepath)
        print(f"        [OK] {filename} ({len(batch_proteins)} sequences)")
    
    return generated_files


def generate_summary(
    df_no_uniprot: pd.DataFrame,
    generated_files: List[Path],
    output_dir: Path
):
    """Genere un fichier de resume."""
    
    summary_file = output_dir / "README.txt"
    
    content = f"""
================================================================================
FICHIERS FASTA POUR COLABFOLD
================================================================================

Date de generation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONTENU
-------
Ces fichiers contiennent les sequences des proteines fluorescentes qui n'ont
PAS d'ID UniProt, et donc pas de structure disponible dans AlphaFold DB.

Nombre total de sequences: {len(df_no_uniprot)}
Nombre de fichiers FASTA: {len(generated_files)}

FICHIERS GENERES
----------------
"""
    
    for f in generated_files:
        content += f"  - {f.name}\n"
    
    content += """
UTILISATION AVEC COLABFOLD
--------------------------
1. Ouvrir Google Colab: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb

2. Uploader UN fichier FASTA a la fois (50 sequences max recommande)

3. Configurer:
   - template_mode: none (plus rapide)
   - num_recycles: 3 (bon compromis)
   - model_type: auto

4. Lancer et attendre (~2-5 min par sequence)

5. Telecharger les resultats (.pdb, .json)

6. Repeter pour chaque batch

TEMPS ESTIME
------------
~3 min/sequence en moyenne sur Colab GPU
Total pour {0} sequences: ~{1:.0f} heures

STRUCTURE DES HEADERS FASTA
---------------------------
>nom_proteine|slug|ex_max|em_max

Exemple:
>mCherry-variant1|mcherry-variant1|ex587|em610

================================================================================
""".format(len(df_no_uniprot), len(df_no_uniprot) * 3 / 60)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n[OK] Resume: {summary_file}")


def generate_colabfold_script(output_dir: Path, generated_files: List[Path]):
    """Genere un script Python pour automatiser ColabFold (optionnel)."""
    
    script_content = '''#!/usr/bin/env python3
"""
Script d'aide pour traiter les resultats ColabFold.
A executer apres avoir telecharge tous les resultats.

Usage:
    python process_colabfold_results.py --results-dir colabfold_results/
"""

import os
import json
from pathlib import Path
import shutil

def organize_results(results_dir: Path, output_dir: Path):
    """Organise les resultats ColabFold."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_dir = output_dir / "pdb"
    pdb_dir.mkdir(exist_ok=True)
    
    # Trouver tous les fichiers PDB
    pdb_files = list(results_dir.glob("**/*_relaxed_rank_001*.pdb"))
    
    if not pdb_files:
        pdb_files = list(results_dir.glob("**/*_unrelaxed_rank_001*.pdb"))
    
    print(f"Trouve {len(pdb_files)} fichiers PDB")
    
    for pdb in pdb_files:
        # Extraire le nom de la proteine
        name = pdb.stem.split("_")[0]
        dest = pdb_dir / f"{name}.pdb"
        shutil.copy(pdb, dest)
        print(f"  {name}.pdb")
    
    print(f"\\nResultat: {len(pdb_files)} structures dans {pdb_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/structures/colabfold")
    args = parser.parse_args()
    
    organize_results(Path(args.results_dir), Path(args.output_dir))
'''
    
    script_file = output_dir / "process_colabfold_results.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"[OK] Script helper: {script_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Fonction principale."""
    
    # Arguments
    parser = argparse.ArgumentParser(
        description='Genere des fichiers FASTA pour ColabFold'
    )
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT),
                       help='Fichier CSV d\'entree')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help='Repertoire de sortie pour les FASTA')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Nombre de sequences par fichier FASTA')
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    batch_size = args.batch_size
    
    print("=" * 70)
    print("[START] Generation des fichiers FASTA pour ColabFold")
    print("=" * 70)
    print(f"   Input: {input_file}")
    print(f"   Output: {output_dir}")
    print(f"   Batch size: {batch_size}")
    print()
    
    # Verifier que le fichier existe
    if not input_file.exists():
        print(f"[ERROR] Fichier non trouve: {input_file}")
        sys.exit(1)
    
    # Charger les donnees
    df = load_dataset(input_file)
    
    # Filtrer sans UniProt
    df_no_uniprot = filter_without_uniprot(df)
    
    if len(df_no_uniprot) == 0:
        print("[INFO] Toutes les proteines ont un UniProt ID!")
        print("       Pas besoin de ColabFold, utiliser AlphaFold DB directement.")
        sys.exit(0)
    
    # Generer les fichiers FASTA
    generated_files = generate_fasta_batches(df_no_uniprot, output_dir, batch_size)
    
    # Generer le resume
    generate_summary(df_no_uniprot, generated_files, output_dir)
    
    # Generer le script helper
    generate_colabfold_script(output_dir, generated_files)
    
    # Resume final
    print()
    print("=" * 70)
    print("[DONE] Generation terminee!")
    print("=" * 70)
    print(f"""
    Fichiers generes dans: {output_dir}
    
    Prochaines etapes:
    1. Ouvrir ColabFold: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb
    2. Uploader batch_001_050.fasta
    3. Lancer la prediction
    4. Telecharger les resultats
    5. Repeter pour chaque batch
    
    Temps estime: ~{len(df_no_uniprot) * 3 / 60:.1f} heures pour {len(df_no_uniprot)} sequences
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
