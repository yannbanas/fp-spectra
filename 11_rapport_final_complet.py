#!/usr/bin/env python3
"""
============================================================================
11_RAPPORT_FINAL_COMPLET.PY - Rapport Final avec TOUS les Résultats
============================================================================

🎯 OBJECTIF:
   Générer un rapport HTML complet incluant:
   - Tous les modèles testés (baseline → advanced)
   - Deep Learning (Multi-Task, GNN)
   - Features avancés + ESM-2 Local (MEILLEUR RÉSULTAT)
   - Toutes les visualisations

📤 OUTPUT:
   - reports/RAPPORT_FINAL_COMPLET.html

============================================================================
"""

import os
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GÉNÉRATION DU RAPPORT
# ============================================================================

def generate_complete_report():
    """Génère le rapport HTML complet."""
    
    print("="*70)
    print("📄 GÉNÉRATION DU RAPPORT FINAL COMPLET")
    print("="*70)
    
    date_str = datetime.now().strftime("%d %B %Y à %H:%M")
    
    html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Final Complet - Prédiction des Propriétés Spectrales des Protéines Fluorescentes</title>
    <style>
        :root {{
            --primary: #3498db;
            --secondary: #9b59b6;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --dark: #2c3e50;
            --light: #ecf0f1;
            --gold: #f1c40f;
        }}
        
        * {{ box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: var(--dark);
            border-bottom: 4px solid var(--primary);
            padding-bottom: 15px;
            font-size: 2.2em;
        }}
        
        h2 {{
            color: var(--primary);
            border-left: 5px solid var(--primary);
            padding-left: 15px;
            margin-top: 40px;
        }}
        
        h3 {{
            color: var(--secondary);
            margin-top: 30px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 50px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }}
        
        .header h1 {{
            border: none;
            color: white;
            margin: 0;
            font-size: 2.8em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header .subtitle {{
            font-size: 1.4em;
            opacity: 0.95;
            margin-top: 15px;
        }}
        
        .header .meta {{
            margin-top: 25px;
            font-size: 1em;
            opacity: 0.85;
        }}
        
        .record-box {{
            background: linear-gradient(135deg, #f1c40f, #f39c12);
            color: #2c3e50;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin: 30px 0;
            box-shadow: 0 5px 20px rgba(241, 196, 15, 0.4);
        }}
        
        .record-box h2 {{
            color: #2c3e50;
            border: none;
            margin: 0;
            font-size: 2em;
        }}
        
        .record-box .value {{
            font-size: 4em;
            font-weight: bold;
            margin: 20px 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 12px 15px;
            text-align: center;
        }}
        
        th {{
            background: linear-gradient(135deg, var(--primary), #2980b9);
            color: white;
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{ background: #f8f9fa; }}
        tr:hover {{ background: #e8f4f8; }}
        
        .highlight-row {{
            background: linear-gradient(90deg, #d4edda, #c3e6cb) !important;
            font-weight: bold;
        }}
        
        .best-row {{
            background: linear-gradient(90deg, #fff3cd, #ffeaa7) !important;
            font-weight: bold;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        
        .img-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 25px;
            margin: 25px 0;
        }}
        
        .img-grid img {{ width: 100%; }}
        
        .img-single {{
            text-align: center;
            margin: 25px 0;
        }}
        
        .img-single img {{
            max-width: 900px;
        }}
        
        .box {{
            padding: 25px;
            border-radius: 12px;
            margin: 25px 0;
        }}
        
        .box-primary {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }}
        
        .box-success {{
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border-left: 5px solid var(--success);
        }}
        
        .box-warning {{
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border-left: 5px solid var(--warning);
        }}
        
        .box-info {{
            background: linear-gradient(135deg, #d1ecf1, #bee5eb);
            border-left: 5px solid var(--primary);
        }}
        
        .box-danger {{
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border-left: 5px solid var(--danger);
        }}
        
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, var(--primary), #2980b9);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 20px rgba(52, 152, 219, 0.3);
            transition: transform 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-card.success {{
            background: linear-gradient(135deg, var(--success), #1e8449);
            box-shadow: 0 5px 20px rgba(39, 174, 96, 0.3);
        }}
        
        .metric-card.gold {{
            background: linear-gradient(135deg, #f1c40f, #f39c12);
            color: #2c3e50;
            box-shadow: 0 5px 20px rgba(241, 196, 15, 0.4);
        }}
        
        .metric-card.secondary {{
            background: linear-gradient(135deg, var(--secondary), #7d3c98);
            box-shadow: 0 5px 20px rgba(155, 89, 182, 0.3);
        }}
        
        .metric-value {{
            font-size: 2.8em;
            font-weight: bold;
            margin: 15px 0;
        }}
        
        .metric-label {{
            font-size: 0.95em;
            opacity: 0.9;
        }}
        
        .timeline {{
            position: relative;
            padding: 20px 0;
        }}
        
        .timeline-item {{
            padding: 25px;
            background: white;
            border-radius: 12px;
            margin: 20px 0;
            border-left: 5px solid var(--primary);
            box-shadow: 0 3px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s;
        }}
        
        .timeline-item:hover {{
            transform: translateX(10px);
        }}
        
        .timeline-item.best {{
            border-left-color: var(--gold);
            background: linear-gradient(90deg, #fffbeb, white);
        }}
        
        .timeline-item h4 {{
            margin-top: 0;
            color: var(--primary);
            font-size: 1.2em;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 3px 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background: #2d3436;
            color: #dfe6e9;
            padding: 25px;
            border-radius: 10px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
        
        .toc {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 30px;
            border-radius: 12px;
            margin: 30px 0;
        }}
        
        .toc h3 {{
            margin-top: 0;
            color: var(--dark);
        }}
        
        .toc ul {{
            list-style: none;
            padding-left: 0;
            columns: 2;
        }}
        
        .toc li {{
            padding: 10px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .toc a {{
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }}
        
        .toc a:hover {{
            color: var(--secondary);
        }}
        
        blockquote {{
            border-left: 5px solid var(--secondary);
            padding: 20px 25px;
            margin: 25px 0;
            font-style: italic;
            color: #555;
            background: linear-gradient(90deg, #f9f9f9, white);
            border-radius: 0 10px 10px 0;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 25px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .badge-success {{ background: var(--success); color: white; }}
        .badge-warning {{ background: var(--warning); color: white; }}
        .badge-danger {{ background: var(--danger); color: white; }}
        .badge-primary {{ background: var(--primary); color: white; }}
        .badge-gold {{ background: linear-gradient(135deg, #f1c40f, #f39c12); color: #2c3e50; }}
        
        .progress-container {{
            background: #e9ecef;
            border-radius: 25px;
            padding: 5px;
            margin: 20px 0;
        }}
        
        .progress-bar {{
            background: linear-gradient(90deg, var(--success), #58d68d);
            height: 30px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s;
        }}
        
        footer {{
            text-align: center;
            padding: 40px;
            color: white;
            margin-top: 50px;
        }}
        
        footer a {{
            color: #ffeaa7;
        }}
        
        @media (max-width: 768px) {{
            .toc ul {{ columns: 1; }}
            .img-grid {{ grid-template-columns: 1fr; }}
            .metric-cards {{ grid-template-columns: 1fr; }}
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>

<!-- ============================================================ -->
<!-- HEADER -->
<!-- ============================================================ -->

<div class="header">
    <h1>🧬 Prédiction des Propriétés Spectrales des Protéines Fluorescentes</h1>
    <div class="subtitle">À partir de structures prédites par AlphaFold2</div>
    <div class="meta">
        <strong>Projet Master Bioinformatique</strong><br>
        Rapport Final Complet - {date_str}
    </div>
</div>

<!-- ============================================================ -->
<!-- RECORD BOX -->
<!-- ============================================================ -->

<div class="record-box">
    <h2>🏆 MEILLEUR RÉSULTAT OBTENU</h2>
    <div class="value">17.13 nm</div>
    <p style="font-size:1.3em;">MAE pour la prédiction de λ<sub>émission</sub></p>
    <p><strong>Modèle :</strong> Extra Trees + Features Avancés + ESM-2 Local</p>
</div>

<!-- ============================================================ -->
<!-- MÉTRIQUES CLÉS -->
<!-- ============================================================ -->

<div class="container">
    <h2>📊 Métriques Clés du Projet</h2>
    
    <div class="metric-cards">
        <div class="metric-card gold">
            <div class="metric-label">🏆 Meilleur MAE (émission)</div>
            <div class="metric-value">17.13 nm</div>
            <div class="metric-label">Advanced + ESM-2 Local</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">MAE Excitation</div>
            <div class="metric-value">21.44 nm</div>
            <div class="metric-label">Extra Trees</div>
        </div>
        <div class="metric-card secondary">
            <div class="metric-label">Coefficient R²</div>
            <div class="metric-value">0.737</div>
            <div class="metric-label">Variance expliquée</div>
        </div>
        <div class="metric-card success">
            <div class="metric-label">Amélioration Totale</div>
            <div class="metric-value">-2.15 nm</div>
            <div class="metric-label">vs Baseline RF</div>
        </div>
    </div>
    
    <h3>Progression vers l'Objectif (15 nm)</h3>
    <div class="progress-container">
        <div class="progress-bar" style="width: 87%;">
            87% - Plus que 2.13 nm !
        </div>
    </div>
</div>

<!-- ============================================================ -->
<!-- TABLE DES MATIÈRES -->
<!-- ============================================================ -->

<div class="container">
    <div class="toc">
        <h3>📑 Table des Matières</h3>
        <ul>
            <li><a href="#resume">1. Résumé Exécutif</a></li>
            <li><a href="#contexte">2. Contexte et Objectifs</a></li>
            <li><a href="#donnees">3. Données et Pipeline</a></li>
            <li><a href="#features">4. Extraction des Features</a></li>
            <li><a href="#baseline">5. Modèles Baseline</a></li>
            <li><a href="#shap">6. Analyse SHAP</a></li>
            <li><a href="#tuning">7. Hyperparameter Tuning</a></li>
            <li><a href="#ensemble">8. Modèles Ensemble</a></li>
            <li><a href="#esm2">9. ESM-2 Global</a></li>
            <li><a href="#deeplearning">10. Deep Learning (GNN)</a></li>
            <li><a href="#advanced">11. Features Avancés + ESM-2 Local</a></li>
            <li><a href="#comparaison">12. Comparaison Complète</a></li>
            <li><a href="#conclusions">13. Conclusions</a></li>
        </ul>
    </div>
</div>

<!-- ============================================================ -->
<!-- 1. RÉSUMÉ EXÉCUTIF -->
<!-- ============================================================ -->

<div class="container" id="resume">
    <h2>1. 📋 Résumé Exécutif</h2>
    
    <div class="box box-primary">
        <h3 style="color:white; margin-top:0;">Question de Recherche</h3>
        <p style="font-size:1.1em;">
            <em>"Peut-on prédire les propriétés spectrales des protéines fluorescentes 
            à partir des structures 3D prédites par AlphaFold2 ?"</em>
        </p>
    </div>
    
    <div class="box box-success">
        <h4>✅ Réponse : OUI</h4>
        <p>Les structures AlphaFold2 contiennent suffisamment d'information pour prédire 
        les propriétés spectrales avec une MAE de <strong>17.13 nm</strong> pour l'émission.</p>
        <p><strong>Découvertes clés :</strong></p>
        <ul>
            <li>L'angle dihédral ψ du chromophore est le feature le plus prédictif</li>
            <li>Les embeddings ESM-2 <strong>locaux</strong> (±15 résidus) améliorent les prédictions</li>
            <li>Les approches globales (ESM-2 complet, GNN) diluent le signal local</li>
        </ul>
    </div>
</div>

<!-- ============================================================ -->
<!-- 2. CONTEXTE -->
<!-- ============================================================ -->

<div class="container" id="contexte">
    <h2>2. 🎯 Contexte et Objectifs</h2>
    
    <h3>2.1 Problématique</h3>
    <p>Les protéines fluorescentes (FP) sont des outils essentiels en biologie. 
    Prédire leurs propriétés spectrales à partir de leur structure permettrait 
    de concevoir de nouvelles FP avec des couleurs spécifiques.</p>
    
    <h3>2.2 Objectifs</h3>
    <ul>
        <li><strong>Principal :</strong> MAE < 15 nm pour λ<sub>émission</sub></li>
        <li><strong>Secondaire :</strong> Identifier les features structuraux déterminants</li>
        <li><strong>Tertiaire :</strong> Comparer approches classiques ML vs Deep Learning</li>
    </ul>
    
    <h3>2.3 Innovation</h3>
    <p>Premier projet utilisant systématiquement les structures <strong>prédites par AlphaFold2</strong> 
    (et non expérimentales) pour cette tâche.</p>
</div>

<!-- ============================================================ -->
<!-- 3. DONNÉES -->
<!-- ============================================================ -->

<div class="container" id="donnees">
    <h2>3. 📊 Données et Pipeline</h2>
    
    <h3>3.1 Sources</h3>
    <table>
        <tr>
            <th>Source</th>
            <th>Type</th>
            <th>Quantité</th>
        </tr>
        <tr>
            <td><strong>FPbase</strong></td>
            <td>Propriétés spectrales</td>
            <td>1040 FPs</td>
        </tr>
        <tr>
            <td><strong>AlphaFold2/ColabFold</strong></td>
            <td>Structures 3D</td>
            <td>676 structures</td>
        </tr>
        <tr>
            <td><strong>Dataset final</strong></td>
            <td>Fusionné</td>
            <td>517 FPs</td>
        </tr>
    </table>
    
    <h3>3.2 Split</h3>
    <p>Train: 415 (80%) | Test: 102 (20%) - Stratifié par λ<sub>émission</sub></p>
</div>

<!-- ============================================================ -->
<!-- 4. FEATURES -->
<!-- ============================================================ -->

<div class="container" id="features">
    <h2>4. 🧬 Extraction des Features</h2>
    
    <h3>4.1 Features de Base (54)</h3>
    <table>
        <tr>
            <th>Catégorie</th>
            <th>Nombre</th>
            <th>Exemples</th>
        </tr>
        <tr>
            <td>Chromophore</td>
            <td>~10</td>
            <td>chrom_psi_tyr, chrom_chi1_tyr</td>
        </tr>
        <tr>
            <td>Environnement</td>
            <td>~10</td>
            <td>env_n_neighbors, env_ratio_hydrophobic</td>
        </tr>
        <tr>
            <td>Global</td>
            <td>~10</td>
            <td>glob_radius_gyration, glob_plddt_mean</td>
        </tr>
        <tr>
            <td>Séquence</td>
            <td>~24</td>
            <td>seq_freq_*, seq_gravy</td>
        </tr>
    </table>
    
    <h3>4.2 Features Avancés (29 nouveaux)</h3>
    <table>
        <tr>
            <th>Catégorie</th>
            <th>Features</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Angles dihédraux</td>
            <td>φ, ω, χ1, χ2</td>
            <td>Géométrie complète du chromophore</td>
        </tr>
        <tr>
            <td>Distances</td>
            <td>OH-backbone, Cα-Cα</td>
            <td>Liaisons H potentielles</td>
        </tr>
        <tr>
            <td>Planarité</td>
            <td>Ring, Conjugated</td>
            <td>RMSD vs plan idéal</td>
        </tr>
        <tr>
            <td>Cavité</td>
            <td>Volume 6Å, 8Å</td>
            <td>Espace autour du chromophore</td>
        </tr>
        <tr>
            <td>H-bonds</td>
            <td>Nombre, distances</td>
            <td>Réseau de liaisons H</td>
        </tr>
    </table>
    
    <h3>4.3 ESM-2 Local (30 features PCA)</h3>
    <p>Embeddings sur fenêtre de <strong>±15 résidus</strong> autour du chromophore, 
    réduits par PCA de 1280 → 30 dimensions.</p>
    
    <div class="box box-info">
        <h4>💡 Total : 113 features</h4>
        <p>54 (base) + 29 (avancés) + 30 (ESM-2 local PCA)</p>
    </div>
</div>

<!-- ============================================================ -->
<!-- 5. BASELINE -->
<!-- ============================================================ -->

<div class="container" id="baseline">
    <h2>5. 🌲 Modèles Baseline</h2>
    
    <h3>5.1 Résultats</h3>
    <table>
        <tr>
            <th>Modèle</th>
            <th>ex_max MAE</th>
            <th>em_max MAE</th>
            <th>R²</th>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>21.93 nm</td>
            <td>19.28 nm</td>
            <td>0.713</td>
        </tr>
        <tr>
            <td>XGBoost</td>
            <td>23.28 nm</td>
            <td>19.29 nm</td>
            <td>0.694</td>
        </tr>
    </table>
    
    <h3>5.2 Visualisations</h3>
    <div class="img-single">
        <img src="predictions_scatter.png" alt="Baseline predictions">
    </div>
    
    <div class="img-single">
        <img src="feature_importance.png" alt="Feature importance">
    </div>
</div>

<!-- ============================================================ -->
<!-- 6. SHAP -->
<!-- ============================================================ -->

<div class="container" id="shap">
    <h2>6. 🔍 Analyse SHAP</h2>
    
    <p>SHAP (SHapley Additive exPlanations) permet d'interpréter les contributions 
    de chaque feature à chaque prédiction.</p>
    
    <h3>6.1 Summary Plots</h3>
    <div class="img-grid">
        <img src="shap_summary_ex_max.png" alt="SHAP ex_max">
        <img src="shap_summary_em_max.png" alt="SHAP em_max">
    </div>
    
    <h3>6.2 Dependence Plots</h3>
    <div class="img-grid">
        <img src="shap_dependence_ex_max.png" alt="Dependence ex_max">
        <img src="shap_dependence_em_max.png" alt="Dependence em_max">
    </div>
    
    <h3>6.3 Waterfall Plots</h3>
    <div class="img-grid">
        <img src="shap_waterfall_ex_max.png" alt="Waterfall ex_max">
        <img src="shap_waterfall_em_max.png" alt="Waterfall em_max">
    </div>
    
    <div class="box box-success">
        <h4>🔬 Découverte SHAP</h4>
        <p><strong>chrom_psi_tyr</strong> (angle ψ du chromophore) est le feature #1 
        avec une importance SHAP de 14.51. Cela confirme que la géométrie locale 
        du chromophore détermine la couleur.</p>
    </div>
</div>

<!-- ============================================================ -->
<!-- 7. TUNING -->
<!-- ============================================================ -->

<div class="container" id="tuning">
    <h2>7. 🔧 Hyperparameter Tuning</h2>
    
    <h3>7.1 Méthode</h3>
    <p>RandomizedSearchCV (50 itérations) + GridSearchCV (affinement)</p>
    
    <h3>7.2 Résultats</h3>
    <div class="img-single">
        <img src="tuning_improvement.png" alt="Tuning improvement">
    </div>
    
    <div class="box box-warning">
        <h4>⚠️ Observation</h4>
        <p>Le tuning n'a <strong>pas amélioré</strong> les performances. 
        Les hyperparamètres par défaut de scikit-learn sont déjà proches de l'optimum.</p>
    </div>
    
    <div class="img-single">
        <img src="tuning_predictions.png" alt="Tuning predictions">
    </div>
</div>

<!-- ============================================================ -->
<!-- 8. ENSEMBLE -->
<!-- ============================================================ -->

<div class="container" id="ensemble">
    <h2>8. 🎭 Modèles Ensemble</h2>
    
    <h3>8.1 Modèles Testés</h3>
    <ul>
        <li>Random Forest, XGBoost, Extra Trees, Gradient Boosting</li>
        <li>Ridge, KNN</li>
        <li>Stacking, Moyenne simple/pondérée</li>
    </ul>
    
    <h3>8.2 Comparaison</h3>
    <div class="img-grid">
        <img src="ensemble_comparison_ex_max.png" alt="Ensemble ex_max">
        <img src="ensemble_comparison_em_max.png" alt="Ensemble em_max">
    </div>
    
    <h3>8.3 Meilleur : Extra Trees</h3>
    <div class="img-grid">
        <img src="ensemble_best_ex_max.png" alt="Best ex_max">
        <img src="ensemble_best_em_max.png" alt="Best em_max">
    </div>
    
    <table>
        <tr>
            <th>Cible</th>
            <th>Baseline RF</th>
            <th>Extra Trees</th>
            <th>Amélioration</th>
        </tr>
        <tr>
            <td>ex_max</td>
            <td>21.93 nm</td>
            <td>21.44 nm</td>
            <td><span class="badge badge-success">-0.49 nm</span></td>
        </tr>
        <tr class="highlight-row">
            <td>em_max</td>
            <td>19.28 nm</td>
            <td>18.47 nm</td>
            <td><span class="badge badge-success">-0.81 nm</span></td>
        </tr>
    </table>
</div>

<!-- ============================================================ -->
<!-- 9. ESM-2 GLOBAL -->
<!-- ============================================================ -->

<div class="container" id="esm2">
    <h2>9. 🧬 ESM-2 Global</h2>
    
    <p>ESM-2 est un modèle de langage protéique de Meta AI (650M paramètres).</p>
    
    <h3>9.1 Approche</h3>
    <p>Mean pooling sur <strong>toute la séquence</strong> (~230 résidus).</p>
    
    <h3>9.2 Résultats</h3>
    <div class="img-single">
        <img src="esm2_comparison.png" alt="ESM-2 comparison">
    </div>
    
    <div class="img-single">
        <img src="esm2_predictions.png" alt="ESM-2 predictions">
    </div>
    
    <div class="box box-danger">
        <h4>❌ Résultat Négatif</h4>
        <p>ESM-2 global n'a <strong>pas amélioré</strong> les prédictions (18.57 nm vs 18.47 nm). 
        Le signal local du chromophore est dilué dans les 230 résidus.</p>
    </div>
</div>

<!-- ============================================================ -->
<!-- 10. DEEP LEARNING -->
<!-- ============================================================ -->

<div class="container" id="deeplearning">
    <h2>10. 🧠 Deep Learning</h2>
    
    <h3>10.1 Approches Testées</h3>
    <ul>
        <li><strong>Multi-Task sklearn :</strong> Extra Trees multi-output</li>
        <li><strong>Multi-Task NN :</strong> MLP avec tronc partagé</li>
        <li><strong>GNN :</strong> Graph Neural Network sur la structure 3D</li>
    </ul>
    
    <h3>10.2 Résultats</h3>
    <div class="img-single">
        <img src="deep_learning_comparison.png" alt="Deep Learning comparison">
    </div>
    
    <div class="img-single">
        <img src="deep_learning_training_curves.png" alt="Training curves">
    </div>
    
    <table>
        <tr>
            <th>Modèle</th>
            <th>ex_max MAE</th>
            <th>em_max MAE</th>
        </tr>
        <tr>
            <td>Baseline (Extra Trees)</td>
            <td>21.44 nm</td>
            <td>18.47 nm</td>
        </tr>
        <tr>
            <td>Multi-Task (sklearn)</td>
            <td>21.44 nm</td>
            <td>18.47 nm</td>
        </tr>
        <tr>
            <td>Multi-Task (NN)</td>
            <td>25.46 nm</td>
            <td>19.53 nm</td>
        </tr>
        <tr>
            <td>GNN</td>
            <td>22.63 nm</td>
            <td>21.06 nm</td>
        </tr>
    </table>
    
    <div class="box box-warning">
        <h4>⚠️ Observation</h4>
        <p>Le Deep Learning n'améliore pas les résultats sur ce dataset de 415 échantillons. 
        Le GNN dilue le signal local du chromophore via le message passing global.</p>
    </div>
</div>

<!-- ============================================================ -->
<!-- 11. FEATURES AVANCÉS + ESM-2 LOCAL -->
<!-- ============================================================ -->

<div class="container" id="advanced">
    <h2>11. 🔬 Features Avancés + ESM-2 Local</h2>
    
    <div class="record-box">
        <h2>🏆 MEILLEUR RÉSULTAT DU PROJET</h2>
        <div class="value">17.13 nm</div>
        <p>MAE em_max avec 113 features</p>
    </div>
    
    <h3>11.1 Nouveautés</h3>
    <ul>
        <li><strong>29 features géométriques avancés</strong> du chromophore</li>
        <li><strong>ESM-2 LOCAL</strong> sur ±15 résidus autour du chromophore</li>
    </ul>
    
    <h3>11.2 Résultats</h3>
    <div class="img-single">
        <img src="advanced_comparison.png" alt="Advanced comparison">
    </div>
    
    <table>
        <tr>
            <th>Cible</th>
            <th>Baseline</th>
            <th>Advanced + ESM-2 Local</th>
            <th>Amélioration</th>
        </tr>
        <tr>
            <td>ex_max</td>
            <td>21.44 nm</td>
            <td>21.75 nm</td>
            <td><span class="badge badge-warning">-0.31 nm</span></td>
        </tr>
        <tr class="best-row">
            <td>em_max</td>
            <td>18.47 nm</td>
            <td><strong>17.13 nm</strong></td>
            <td><span class="badge badge-gold">+1.34 nm 🏆</span></td>
        </tr>
    </table>
    
    <h3>11.3 Feature Importance</h3>
    <div class="img-grid">
        <img src="advanced_importance_ex_max.png" alt="Importance ex_max">
        <img src="advanced_importance_em_max.png" alt="Importance em_max">
    </div>
    
    <div class="box box-success">
        <h4>💡 Pourquoi ça marche ?</h4>
        <p><strong>ESM-2 Local</strong> concentre l'information sur les 31 résidus 
        entourant le chromophore au lieu de diluer sur 230+ résidus. 
        Combiné aux features géométriques avancés, cela capture finement 
        l'environnement qui influence le spectre.</p>
    </div>
</div>

<!-- ============================================================ -->
<!-- 12. COMPARAISON COMPLÈTE -->
<!-- ============================================================ -->

<div class="container" id="comparaison">
    <h2>12. 📈 Comparaison Complète de Toutes les Approches</h2>
    
    <table>
        <tr>
            <th>Approche</th>
            <th>ex_max MAE</th>
            <th>em_max MAE</th>
            <th>R² (em)</th>
            <th>Statut</th>
        </tr>
        <tr>
            <td>Random Forest (baseline)</td>
            <td>21.93 nm</td>
            <td>19.28 nm</td>
            <td>0.713</td>
            <td>Point de départ</td>
        </tr>
        <tr>
            <td>XGBoost</td>
            <td>23.28 nm</td>
            <td>19.29 nm</td>
            <td>0.694</td>
            <td>Similaire RF</td>
        </tr>
        <tr class="highlight-row">
            <td>Extra Trees</td>
            <td>21.44 nm</td>
            <td>18.47 nm</td>
            <td>0.718</td>
            <td>✅ Meilleur simple</td>
        </tr>
        <tr>
            <td>Hyperparameter Tuning</td>
            <td>23.35 nm</td>
            <td>19.48 nm</td>
            <td>0.715</td>
            <td>❌ Pas d'amélioration</td>
        </tr>
        <tr>
            <td>Stacking</td>
            <td>31.12 nm</td>
            <td>22.22 nm</td>
            <td>0.685</td>
            <td>❌ Dataset trop petit</td>
        </tr>
        <tr>
            <td>ESM-2 Global</td>
            <td>22.35 nm</td>
            <td>18.57 nm</td>
            <td>0.731</td>
            <td>❌ Dilue le signal</td>
        </tr>
        <tr>
            <td>Multi-Task NN</td>
            <td>25.46 nm</td>
            <td>19.53 nm</td>
            <td>0.717</td>
            <td>❌ Overfitting</td>
        </tr>
        <tr>
            <td>GNN</td>
            <td>22.63 nm</td>
            <td>21.06 nm</td>
            <td>0.706</td>
            <td>❌ Dilue le signal</td>
        </tr>
        <tr class="best-row">
            <td><strong>Advanced + ESM-2 Local</strong></td>
            <td>21.75 nm</td>
            <td><strong>17.13 nm</strong></td>
            <td><strong>0.737</strong></td>
            <td><span class="badge badge-gold">🏆 MEILLEUR</span></td>
        </tr>
    </table>
    
    <h3>Évolution des Performances</h3>
    <div class="timeline">
        <div class="timeline-item">
            <h4>📌 Étape 1 : Baseline RF</h4>
            <p>MAE = 19.28 nm - Point de départ avec 54 features</p>
        </div>
        <div class="timeline-item">
            <h4>📌 Étape 2 : Extra Trees</h4>
            <p>MAE = 18.47 nm - Amélioration de 0.81 nm</p>
        </div>
        <div class="timeline-item">
            <h4>📌 Étape 3 : Tentatives infructueuses</h4>
            <p>Tuning, Stacking, ESM-2 global, GNN - Pas d'amélioration</p>
        </div>
        <div class="timeline-item best">
            <h4>🏆 Étape 4 : Advanced + ESM-2 Local</h4>
            <p>MAE = <strong>17.13 nm</strong> - Amélioration de 1.34 nm</p>
            <p><strong>Total : -2.15 nm vs baseline</strong></p>
        </div>
    </div>
</div>

<!-- ============================================================ -->
<!-- 13. CONCLUSIONS -->
<!-- ============================================================ -->

<div class="container" id="conclusions">
    <h2>13. 🎓 Conclusions</h2>
    
    <div class="box box-primary">
        <h3 style="color:white; margin-top:0;">Hypothèse Validée ✅</h3>
        <p style="font-size:1.1em;">
            <em>"Les structures prédites par AlphaFold2 contiennent suffisamment d'information 
            pour prédire les propriétés spectrales des protéines fluorescentes."</em>
        </p>
    </div>
    
    <h3>13.1 Résultats Principaux</h3>
    <ul>
        <li><strong>Meilleur MAE :</strong> 17.13 nm pour λ<sub>émission</sub></li>
        <li><strong>Amélioration totale :</strong> -2.15 nm vs baseline RF</li>
        <li><strong>Feature #1 :</strong> Angle ψ du chromophore (géométrie locale)</li>
    </ul>
    
    <h3>13.2 Découvertes Scientifiques</h3>
    <ol>
        <li><strong>La géométrie LOCALE du chromophore est déterminante</strong> - 
        L'angle ψ capture directement l'information sur le gap HOMO-LUMO.</li>
        
        <li><strong>Les approches GLOBALES échouent</strong> - 
        ESM-2 global, GNN diluent le signal du chromophore (3 résidus sur 230).</li>
        
        <li><strong>ESM-2 LOCAL fonctionne</strong> - 
        Concentrer sur ±15 résidus capture l'environnement pertinent.</li>
        
        <li><strong>Extra Trees > Random Forest</strong> - 
        La randomisation supplémentaire réduit l'overfitting.</li>
        
        <li><strong>Deep Learning n'est pas adapté ici</strong> - 
        415 échantillons insuffisants pour les réseaux profonds.</li>
    </ol>
    
    <h3>13.3 Pour le Mémoire</h3>
    <blockquote>
        "Cette étude démontre que les structures prédites par AlphaFold2 permettent 
        de prédire les propriétés spectrales des protéines fluorescentes avec une 
        MAE de 17.13 nm pour la longueur d'onde d'émission. L'analyse SHAP révèle 
        que l'angle dihédral ψ du chromophore est le descripteur le plus prédictif. 
        L'approche la plus performante combine des features géométriques avancés 
        du chromophore avec des embeddings ESM-2 locaux (±15 résidus), démontrant 
        l'importance de concentrer l'information sur la région du chromophore 
        plutôt que d'utiliser des représentations globales de la protéine."
    </blockquote>
    
    <h3>13.4 Perspectives</h3>
    <ul>
        <li>Augmenter le dataset (plus de FPs)</li>
        <li>Features de chimie quantique (TD-DFT simplifié)</li>
        <li>GNN local sur le voisinage du chromophore</li>
        <li>Application au design de nouvelles FP</li>
    </ul>
</div>

<!-- ============================================================ -->
<!-- ANNEXES -->
<!-- ============================================================ -->

<div class="container">
    <h2>📎 Annexes</h2>
    
    <h3>A. Scripts du Pipeline</h3>
    <table>
        <tr><th>Script</th><th>Description</th></tr>
        <tr><td><code>01_collect_fpbase.py</code></td><td>Collecte FPbase</td></tr>
        <tr><td><code>02_generate_fasta.py</code></td><td>Génération FASTA</td></tr>
        <tr><td><code>03_extract_features.py</code></td><td>Features de base</td></tr>
        <tr><td><code>05_train_baseline.py</code></td><td>RF & XGBoost</td></tr>
        <tr><td><code>06_shap_analysis.py</code></td><td>Analyse SHAP</td></tr>
        <tr><td><code>07_hyperparameter_tuning.py</code></td><td>Tuning</td></tr>
        <tr><td><code>08_ensemble_stacking.py</code></td><td>Ensembles</td></tr>
        <tr><td><code>09_esm2_embeddings.py</code></td><td>ESM-2 Global</td></tr>
        <tr><td><code>11_deep_learning_gnn.py</code></td><td>Deep Learning / GNN</td></tr>
        <tr><td><code>12_chromophore_advanced.py</code></td><td>Features Avancés + ESM-2 Local</td></tr>
        <tr><td><code>13_rapport_final_complet.py</code></td><td>Ce rapport</td></tr>
    </table>
    
    <h3>B. Références</h3>
    <ul>
        <li>Jumper et al. (2021). AlphaFold. <em>Nature</em>.</li>
        <li>Lin et al. (2023). ESM-2. <em>Science</em>.</li>
        <li>Lambert (2019). FPbase. <em>Nature Methods</em>.</li>
        <li>Lundberg & Lee (2017). SHAP. <em>NeurIPS</em>.</li>
    </ul>
</div>

<!-- ============================================================ -->
<!-- FOOTER -->
<!-- ============================================================ -->

<footer>
    <p style="font-size:1.2em;">
        <strong>🧬 Prédiction des Propriétés Spectrales des Protéines Fluorescentes</strong>
    </p>
    <p>
        Projet Master Bioinformatique<br>
        Rapport généré le {date_str}
    </p>
    <p style="margin-top:20px; opacity:0.8;">
        🏆 Meilleur résultat : MAE = 17.13 nm
    </p>
</footer>

</body>
</html>
"""
    
    # Sauvegarder
    report_path = REPORTS_DIR / "RAPPORT_FINAL_COMPLET.html"
    report_path.write_text(html, encoding='utf-8')
    
    print(f"\n✅ Rapport généré : {report_path}")
    print(f"   Taille : {report_path.stat().st_size / 1024:.1f} KB")
    print(f"\n📂 Ouvre ce fichier dans ton navigateur !")
    print(f"   {report_path.absolute()}")
    
    return report_path


if __name__ == "__main__":
    generate_complete_report()
