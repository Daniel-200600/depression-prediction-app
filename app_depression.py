"""
Application Streamlit — Prédiction de la Dépression Étudiante
Étudiant : Tchomtchi Daniel
Modèle   : Random Forest Regressor (depression_rf_model.pkl)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction Depression Etudiante",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CSS PERSONNALISE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #1a3c5e 0%, #2e6da4 100%);
        padding: 28px 36px;
        border-radius: 12px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .main-header h1 { margin: 0 0 6px 0; font-size: 1.9rem; font-weight: 700; }
    .main-header p  { margin: 0; font-size: 0.95rem; opacity: 0.85; }

    /* Cartes de métriques */
    .metric-card {
        background: #f8f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card h2 { font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p  { font-size: 0.85rem; color: #64748b; margin: 6px 0 0 0; }

    /* Boite de résultat */
    .result-box {
        border-radius: 12px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
        font-size: 1rem;
    }
    .result-low  { background: #dcfce7; border: 2px solid #22c55e; color: #166534; }
    .result-mid  { background: #fef9c3; border: 2px solid #eab308; color: #854d0e; }
    .result-high { background: #fee2e2; border: 2px solid #ef4444; color: #991b1b; }

    /* Séparateur section */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a3c5e;
        border-bottom: 2px solid #2e6da4;
        padding-bottom: 6px;
        margin: 24px 0 16px 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #f0f4f8; }

    /* Bouton principal */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1a3c5e, #2e6da4);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        width: 100%;
        transition: opacity 0.2s;
    }
    div.stButton > button:first-child:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODELE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    """Charge le pipeline scikit-learn sauvegardé."""
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL_PATH = "depression_rf_model.pkl"
model_loaded = os.path.exists(MODEL_PATH)

if model_loaded:
    model = load_model(MODEL_PATH)
else:
    model = None


# ─────────────────────────────────────────────────────────────────────────────
# EN-TETE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Prediction de la Depression Etudiante</h1>
    <p>
        Modele : Random Forest Regressor &nbsp;|&nbsp;
        Dataset : student_lifestyle_100k.csv &nbsp;|&nbsp;
        Etudiant : Tchomtchi Daniel
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        "Fichier modele introuvable : `depression_rf_model.pkl`\n\n"
        "Veuillez d'abord executer le notebook `Tchomtchi_Daniel.ipynb` "
        "pour generer le fichier `.pkl`, puis relancer cette application."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FORMULAIRE DE SAISIE
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Profil de l'Etudiant")
    st.markdown("Renseignez les caracteristiques de l'etudiant pour obtenir le score de depression predit.")
    st.markdown("---")

    st.markdown("**Informations personnelles**")
    age        = st.slider("Age", 18, 25, 21)
    gender     = st.selectbox("Genre", ["Male", "Female"])
    department = st.selectbox("Departement", ["Science", "Engineering", "Arts", "Medical", "Business"])

    st.markdown("---")
    st.markdown("**Indicateurs academiques et de style de vie**")
    cgpa               = st.slider("CGPA (Moyenne generale)", 1.5, 4.0, 3.0, step=0.01)
    study_hours        = st.slider("Heures d'etude par jour", 0.0, 24.0, 6.0, step=0.5)
    sleep_duration     = st.slider("Duree de sommeil (heures/nuit)", 3.0, 12.0, 7.0, step=0.25)
    social_media_hours = st.slider("Reseaux sociaux (heures/jour)", 0.0, 12.0, 2.0, step=0.25)
    physical_activity  = st.slider("Activite physique (minutes/semaine)", 0, 150, 75)
    stress_level       = st.slider("Niveau de stress (1 = faible, 10 = extreme)", 1, 10, 5)

    st.markdown("---")
    predict_btn = st.button("Calculer le Score de Depression")


# ─────────────────────────────────────────────────────────────────────────────
# ZONE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediction", "Analyse du Profil", "Informations sur le Modele"])


# ── TAB 1 : Prediction ───────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Resultat de la Prediction</div>', unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame({
            "Age":                [age],
            "Gender":             [gender],
            "Department":         [department],
            "CGPA":               [cgpa],
            "Sleep_Duration":     [sleep_duration],
            "Study_Hours":        [study_hours],
            "Social_Media_Hours": [social_media_hours],
            "Physical_Activity":  [physical_activity],
            "Stress_Level":       [stress_level],
        })

        with st.spinner("Inference en cours..."):
            time.sleep(0.6)
            score = float(np.clip(model.predict(input_df)[0], 0, 1))

        # ── Affichage du score ──
        col_score, col_gauge = st.columns([1, 1])

        with col_score:
            pct = score * 100
            if score < 0.35:
                risk_label = "Risque Faible"
                css_class  = "result-low"
                icon       = "Risque faible de depression detecte."
            elif score < 0.65:
                risk_label = "Risque Modere"
                css_class  = "result-mid"
                icon       = "Risque modere. Un suivi est conseille."
            else:
                risk_label = "Risque Eleve"
                css_class  = "result-high"
                icon       = "Risque eleve. Une prise en charge est recommandee."

            st.markdown(f"""
            <div class="result-box {css_class}">
                <h2 style="font-size:3rem; margin:0">{pct:.1f}%</h2>
                <p style="font-size:1.1rem; font-weight:600; margin:10px 0 6px 0">{risk_label}</p>
                <p style="margin:0">{icon}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            # Jauge semi-circulaire
            fig_g, ax_g = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
            ax_g.set_xlim(-1.3, 1.3)
            ax_g.set_ylim(-0.2, 1.3)
            ax_g.axis("off")

            segments = [
                (0.00, 0.35, "#22c55e"),
                (0.35, 0.65, "#eab308"),
                (0.65, 1.00, "#ef4444"),
            ]
            for lo, hi, color in segments:
                theta1 = 180 * (1 - hi)
                theta2 = 180 * (1 - lo)
                arc = mpatches.Wedge(
                    center=(0, 0), r=1.0, theta1=theta1, theta2=theta2,
                    width=0.28, color=color, alpha=0.85
                )
                ax_g.add_patch(arc)

            # Aiguille
            angle = np.pi * (1 - score)
            needle_x = 0.72 * np.cos(angle)
            needle_y = 0.72 * np.sin(angle)
            ax_g.annotate("", xy=(needle_x, needle_y), xytext=(0, 0),
                           arrowprops=dict(arrowstyle="->", color="#1a3c5e", lw=2.5))
            ax_g.add_patch(plt.Circle((0, 0), 0.06, color="#1a3c5e", zorder=5))

            ax_g.text(0, -0.12, f"{pct:.1f}%", ha="center", va="center",
                      fontsize=16, fontweight="bold", color="#1a3c5e")
            ax_g.text(-1.2, -0.05, "0%", ha="center", fontsize=8, color="gray")
            ax_g.text( 1.2, -0.05, "100%", ha="center", fontsize=8, color="gray")
            ax_g.set_title("Score de Depression", fontsize=11, pad=6, color="#1a3c5e", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

        # ── Recapitulatif des donnees saisies ──
        st.markdown('<div class="section-title">Recapitulatif du Profil Saisi</div>', unsafe_allow_html=True)
        summary = {
            "Age": age, "Genre": gender, "Departement": department,
            "CGPA": cgpa, "Heures d'etude/jour": study_hours,
            "Duree de sommeil (h)": sleep_duration,
            "Reseaux sociaux (h/j)": social_media_hours,
            "Activite physique (min/sem)": physical_activity,
            "Niveau de stress": stress_level,
        }
        summary_df = pd.DataFrame(summary.items(), columns=["Caracteristique", "Valeur"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.info("Renseignez le profil dans le panneau gauche, puis cliquez sur **Calculer le Score de Depression**.")


# ── TAB 2 : Analyse du Profil ────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Analyse Visuelle du Profil</div>', unsafe_allow_html=True)

    # Comparaison radar — profil vs moyennes dataset
    dataset_means = {
        "CGPA":               2.78,
        "Heures etude":       6.1,
        "Sommeil (h)":        6.9,
        "Reseaux sociaux":    3.2,
        "Activite physique":  74.4,
        "Stress":             4.1,
    }

    # Normalisation dans [0, 1] pour le radar
    ranges = {
        "CGPA":               (1.5, 4.0),
        "Heures etude":       (0, 24),
        "Sommeil (h)":        (3, 12),
        "Reseaux sociaux":    (0, 12),
        "Activite physique":  (0, 150),
        "Stress":             (1, 10),
    }
    user_vals = {
        "CGPA":               cgpa,
        "Heures etude":       study_hours,
        "Sommeil (h)":        sleep_duration,
        "Reseaux sociaux":    social_media_hours,
        "Activite physique":  physical_activity,
        "Stress":             stress_level,
    }

    def norm(val, lo, hi):
        return (val - lo) / (hi - lo)

    labels = list(ranges.keys())
    user_norm  = [norm(user_vals[k],     *ranges[k]) for k in labels]
    mean_norm  = [norm(dataset_means[k], *ranges[k]) for k in labels]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    user_norm  += user_norm[:1]
    mean_norm  += mean_norm[:1]
    angles     += angles[:1]

    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax_r.plot(angles, user_norm, 'o-', linewidth=2, color='#2e6da4', label="Profil etudiant")
        ax_r.fill(angles, user_norm, alpha=0.2, color='#2e6da4')
        ax_r.plot(angles, mean_norm, 's--', linewidth=1.5, color='#DD8452', label="Moyenne dataset")
        ax_r.fill(angles, mean_norm, alpha=0.1, color='#DD8452')
        ax_r.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax_r.set_ylim(0, 1)
        ax_r.set_title("Radar — Profil vs Moyenne", fontsize=11, pad=16, fontweight="bold")
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    with col_r2:
        # Barres comparatives
        fig_b, ax_b = plt.subplots(figsize=(6, 5))
        x = np.arange(N)
        width = 0.35
        bars1 = ax_b.bar(x - width/2, [user_vals[k] for k in labels],
                         width, label="Profil etudiant", color='#2e6da4', alpha=0.85)
        bars2 = ax_b.bar(x + width/2, [dataset_means[k] for k in labels],
                         width, label="Moyenne dataset", color='#DD8452', alpha=0.65)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax_b.set_title("Comparaison Profil vs Moyenne Dataset", fontweight="bold")
        ax_b.legend(fontsize=9)
        sns.despine(ax=ax_b)
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    # Indicateurs individuels
    st.markdown('<div class="section-title">Indicateurs de Risque Individuels</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    stress_color = "#ef4444" if stress_level >= 7 else "#eab308" if stress_level >= 4 else "#22c55e"
    sleep_color  = "#ef4444" if sleep_duration < 5 else "#eab308" if sleep_duration < 7 else "#22c55e"
    act_color    = "#ef4444" if physical_activity < 30 else "#eab308" if physical_activity < 75 else "#22c55e"

    col1.markdown(f"""
    <div class="metric-card">
        <h2 style="color:{stress_color}">{stress_level}/10</h2>
        <p>Niveau de Stress</p>
    </div>""", unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
        <h2 style="color:{sleep_color}">{sleep_duration}h</h2>
        <p>Duree de Sommeil</p>
    </div>""", unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
        <h2 style="color:{act_color}">{physical_activity} min</h2>
        <p>Activite Physique/Semaine</p>
    </div>""", unsafe_allow_html=True)


# ── TAB 3 : Informations modele ──────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Informations sur le Modele</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**Type de modele**")
        st.info("Random Forest Regressor — Pipeline scikit-learn")

        st.markdown("**Variables d'entree**")
        features_info = {
            "Age": "Numerique — 18 a 25 ans",
            "Gender": "Categorielle — Male / Female",
            "Department": "Categorielle — 5 departements",
            "CGPA": "Numerique — 1.5 a 4.0",
            "Sleep_Duration": "Numerique — heures de sommeil",
            "Study_Hours": "Numerique — heures d'etude/jour",
            "Social_Media_Hours": "Numerique — heures RS/jour",
            "Physical_Activity": "Numerique — min/semaine",
            "Stress_Level": "Ordinale — 1 a 10",
        }
        feat_df = pd.DataFrame(features_info.items(), columns=["Variable", "Type / Description"])
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    with col_m2:
        st.markdown("**Variable cible**")
        st.info("Depression — encodee en 0 (Non) / 1 (Oui). "
                "La sortie du modele est un score de probabilite [0, 1].")

        st.markdown("**Hyperparametres principaux**")
        params = {
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
        params_df = pd.DataFrame(params.items(), columns=["Parametre", "Valeur"])
        st.dataframe(params_df, hide_index=True, use_container_width=True)

        st.markdown("**Interpretation du score**")
        st.markdown("""
        | Plage du Score | Niveau de Risque |
        |---|---|
        | 0% – 35% | Risque Faible |
        | 35% – 65% | Risque Modere |
        | 65% – 100% | Risque Eleve |
        """)

    st.markdown("---")
    st.markdown(
        "_Cette application est developpee a des fins pedagogiques dans le cadre du cours de Machine Learning. "
        "
    )
