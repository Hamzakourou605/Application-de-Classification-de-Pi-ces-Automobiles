"""
Application Streamlit - Classification de Pieces Automobiles
Systeme d'intelligence artificielle pour la reconnaissance automatique de pieces automobiles
Version 1.0 - Janvier 2026
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from app import AutomobilePartsCNN
from utils import DatasetManager, ResultsExporter
import json
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Automobile Parts Classification System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalise pour meilleure presentation
st.markdown("""
    <style>
    /* Styles generaux */
    .main {
        padding: 2rem;
    }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 500;
    }
    
    /* En-tetes */
    h1, h2, h3 {
        color: #1f77b4;
        font-weight: 600;
    }
    
    /* Sections */
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Cartes de statistiques */
    .stat-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        margin: 20px 0;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 8px;
    }
    
    .logo-image {
        max-width: 200px;
        height: auto;
    }
    
    /* Footer */
    .footer {
        border-top: 1px solid #ddd;
        margin-top: 40px;
        padding-top: 20px;
        text-align: center;
        color: #666;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation de la session
if 'cnn_app' not in st.session_state:
    # Obtenir le répertoire courant
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Créer l'instance avec les chemins absolus
    st.session_state.cnn_app = AutomobilePartsCNN(
        model_path=os.path.join(script_dir, "mon_modele_rgb.keras"),
        label_encoder_path=os.path.join(script_dir, "label_encoder.pkl"),
        csv_path=os.path.join(script_dir, "data_set.csv")
    )
    st.session_state.model_loaded = False
    st.session_state.predictions_history = []
    st.session_state.logo_path = None

cnn_app = st.session_state.cnn_app

# Fonction pour afficher le logo
def display_logo_section():
    """Affiche la section du logo personnalise"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        
        if st.session_state.logo_path and os.path.exists(st.session_state.logo_path):
            try:
                logo = Image.open(st.session_state.logo_path)
                st.image(logo, use_column_width=False, width=180)
            except:
                st.markdown("**[LOGO]**")
        else:
            st.markdown("**Application de Classification de Pieces Automobiles**")
            st.markdown("Systeme de reconnaissance d'images par intelligence artificielle")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Sidebar - Navigation et configuration
with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("---")
    
    page = st.radio(
        "Selectionner une section",
        [
            "Accueil",
            "Analyse du Dataset",
            "Entraînement du Modele",
            "Predictions",
            "Historique",
            "Configuration",
            "Logo Personnalise"
        ]
    )
    
    st.markdown("---")
    st.markdown("### Statut du Modele")
    
    if st.session_state.model_loaded:
        st.success("Modele charge")
        st.metric("Nombre de classes", len(cnn_app.classes) if cnn_app.classes is not None else 0)
        st.metric("Parametres", "4.35M")
    else:
        st.warning("Modele non charge")

# PAGE 1: ACCUEIL
if page == "Accueil":
    display_logo_section()
    
    st.markdown("# Classification de Pieces Automobiles")
    st.markdown("Systeme d'intelligence artificielle pour la reconnaissance automatique")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Presentation de l'Application")
        st.markdown("""
        Cette application utilise la technologie des reseaux de neurones convolutifs (CNN) 
        pour classifier automatiquement les pieces automobiles a partir d'images numeriques.
        
        ### Capacites principales
        
        **Classification** - Identification automatique du type de piece a partir d'une image
        
        **Entraînement** - Possibilite d'entraîner un nouveau modele avec vos donnees
        
        **Analyse** - Visualisation des performances et resultats de predictions
        
        **Gestion Dataset** - Exploration et statistiques des donnees disponibles
        
        **Support Multi-formats** - Compatible avec JPG, PNG, BMP et autres formats courants
        """)
    
    with col2:
        st.markdown("## Informations Cles")
        st.metric("Classes supportees", "14 types")
        st.metric("Architecture", "CNN 4 couches")
        st.metric("Parametres reseau", "4.35M")
        st.metric("Precision attendue", "90-95%")
    
    st.markdown("---")
    st.markdown("## Classes Disponibles")
    
    classes_info = pd.DataFrame({
        'Code': ['bearing', 'clutch', 'filter', 'fuel-tank', 'piston', 'valve', 'wheel',
                 'shocker', 'spark-plug', 'spur-gear', 'helical_gear', 'Bevel-gear', 'cylincer', 'rack-pinion'],
        'Designation': ['Roulement', 'Embrayage', 'Filtre', 'Reservoir de carburant', 'Piston', 'Soupape', 'Roue',
                       'Amortisseur', 'Bougie d\'allumage', 'Engrenage droit', 'Engrenage helicoidale', 
                       'Engrenage conique', 'Cylindre', 'Cremaillere'],
        'Description': ['Composant de roulement', 'Systeme de transmission', 'Systeme de filtration', 
                       'Reservoir carburant', 'Composant moteur', 'Systeme de soupape',
                       'Composant chassis', 'Systeme de suspension', 'Composant allumage',
                       'Transmission de puissance', 'Transmission de puissance', 'Transmission de puissance',
                       'Bloc moteur', 'Systeme de direction']
    })
    
    st.dataframe(classes_info, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## Guide de Demarrage Rapide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Etape 1: Charger le Modele
        Acces a la section **Entraînement du Modele** 
        pour charger un modele pre-entraîne ou 
        en entraîner un nouveau.
        """)
    
    with col2:
        st.markdown("""
        ### Etape 2: Faire une Prediction
        Utilisez l'onglet **Predictions** pour 
        uploader une image et obtenir une 
        classification automatique.
        """)
    
    with col3:
        st.markdown("""
        ### Etape 3: Analyser les Resultats
        Consultez **Historique** pour voir 
        toutes vos predictions et leurs 
        performances.
        """)

# PAGE 2: ANALYSE DATASET
elif page == "Analyse du Dataset":
    st.markdown("# Analyse du Dataset")
    st.markdown("---")
    
    if os.path.exists('data_set.csv'):
        df = pd.read_csv('data_set.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total d'images", f"{df.shape[0]:,}")
        with col2:
            st.metric("Features (caracteristiques)", df.shape[1])
        with col3:
            st.metric("Nombre de classes", df['label'].nunique())
        with col4:
            st.metric("Valeurs manquantes", df.isnull().sum().sum())
        
        st.markdown("---")
        st.markdown("## Distribution des Classes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Graphique de Distribution")
            class_dist = df['label'].value_counts()
            st.bar_chart(class_dist)
        
        with col2:
            st.markdown("### Tableau de Distribution")
            dist_table = class_dist.to_frame().rename(columns={'label': 'Nombre d\'images'})
            st.dataframe(dist_table, use_container_width=True)
        
        st.markdown("---")
        st.markdown("## Analyse des Dossiers")
        
        folder_stats = DatasetManager.scan_all_folders('.')
        
        if folder_stats:
            folders_data = []
            for folder_name, stats in folder_stats.items():
                folders_data.append({
                    'Dossier': folder_name,
                    'Nombre d\'images': stats['total_images']
                })
            
            folders_df = pd.DataFrame(folders_data).sort_values('Nombre d\'images', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(folders_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.bar_chart(folders_df.set_index('Dossier'))
    else:
        st.error("Fichier data_set.csv non trouve. Veuillez verifier que le fichier existe.")

# PAGE 3: ENTRAÎNEMENT DU MODELE
elif page == "Entraînement du Modele":
    st.markdown("# Entraînement du Modele CNN")
    st.markdown("---")
    
    st.markdown("## Actions Principales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Charger un Modele Existant")
        if st.button("Charger le Modele Pre-Entraîne", key="load_model", use_container_width=True):
            with st.spinner("Chargement du modele en cours..."):
                try:
                    if cnn_app.load_model():
                        st.session_state.model_loaded = True
                        st.success("✓ Modele charge avec succès!")
                        st.info(f"📊 Nombre de classes: {len(cnn_app.classes) if cnn_app.classes else 0}")
                    else:
                        st.error("❌ Le modèle n'existe pas encore!")
                        st.warning("💡 **Solution**: Cliquez d'abord sur 'Démarrer l'Entraînement' pour entraîner le modèle")
                        st.info(f"Fichiers attendus:\n- {cnn_app.model_path}\n- {cnn_app.label_encoder_path}")
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement: {str(e)}")
                    st.info(f"Fichiers attendus:\n- {cnn_app.model_path}\n- {cnn_app.label_encoder_path}")
    
    with col2:
        st.markdown("### Entraîner un Nouveau Modele")
        if st.button("Demarrer l'Entraînement", key="train_model", use_container_width=True):
            with st.spinner("Entraînement du modele en cours..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Chargement des donnees...")
                    progress_bar.progress(20)
                    
                    if cnn_app.train_model():
                        st.session_state.model_loaded = True
                        progress_bar.progress(100)
                        status_text.empty()
                        st.success("Entraînement termine et modele sauvegarde!")
                    else:
                        st.error("Erreur lors de l'entraînement")
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    st.markdown("---")
    
    if st.session_state.model_loaded and cnn_app.model:
        st.markdown("## Informations du Modele Charge")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de couches", len(cnn_app.model.layers))
        with col2:
            st.metric("Parametres du reseau", "4.35M")
        with col3:
            st.metric("Classes", len(cnn_app.classes) if cnn_app.classes is not None else 0)
        
        st.markdown("---")
        st.markdown("## Architecture du Modele")
        st.text("""
Couche 1: Input [100 x 100 x 3 pixels RGB]
          |
          V
Couche 2: Convolution 2D (32 filtres, 3x3)
          Activation ReLU
          |
          V
Couche 3: MaxPooling (2x2)
          |
          V
Couche 4: Convolution 2D (64 filtres, 3x3)
          Activation ReLU
          |
          V
Couche 5: MaxPooling (2x2)
          |
          V
Couche 6: Flatten (Aplatissement)
          |
          V
Couche 7: Dense Fully Connected (128 neurones)
          Activation ReLU
          Dropout (50%)
          |
          V
Couche 8: Dense Classification (14 classes)
          Activation Softmax
          |
          V
Output:   Probabilites par classe [0-1]
        """)
        
        st.markdown("---")
        
        if st.button("Tester la Precision du Modele", use_container_width=True):
            with st.spinner("Evaluation du modele..."):
                try:
                    cnn_app.test_accuracy()
                    st.success("Evaluation terminee. Verifiez la console pour les details.")
                except Exception as e:
                    st.error(f"Erreur lors de l'evaluation: {str(e)}")
    else:
        st.info("Veuillez d'abord charger ou entraîner un modele pour voir ses informations.")

# PAGE 4: PREDICTIONS
elif page == "Predictions":
    st.markdown("# Systeme de Prediction")
    st.markdown("---")
    
    if not st.session_state.model_loaded:
        st.warning("Le modele n'est pas charge. Allez a la section 'Entraînement du Modele' pour charger un modele.")
    else:
        tab1, tab2 = st.tabs(["Image Unique", "Traitement par Dossier"])
        
        # TAB 1: Prediction image unique
        with tab1:
            st.markdown("## Prediction d'une Image Unique")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Option 1: Upload une image")
                uploaded_file = st.file_uploader(
                    "Selectionner une image",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    key="single_image"
                )
            
            with col2:
                st.markdown("### Option 2: Chemin du fichier")
                image_path = st.text_input(
                    "Ou entrer le chemin complet",
                    "bearing/bearing.jpg",
                    key="image_path"
                )
            
            if uploaded_file is not None:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Apercu de l'image")
                    try:
                        image = Image.open(temp_path)
                        st.image(image, use_column_width=True)
                    except:
                        st.error("Impossible d'afficher l'image")
                
                with col2:
                    st.markdown("### Resultat de la Prediction")
                    if st.button("Executer la Prediction", use_container_width=True):
                        with st.spinner("Prediction en cours..."):
                            try:
                                result = cnn_app.predict_image(temp_path, show_plot=False)
                                
                                if result:
                                    st.session_state.predictions_history.append({
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'image': uploaded_file.name,
                                        'classe': result['classe'],
                                        'confiance': result['confiance_pourcentage']
                                    })
                                    
                                    st.success(f"Classe Predite: {result['classe']}")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.metric("Confiance", f"{result['confiance_pourcentage']:.2f}%")
                                    
                                    with col_b:
                                        conf = result['confiance_pourcentage']
                                        if conf > 80:
                                            st.metric("Fiabilite", "Tres Elevee", delta="Excellent")
                                        elif conf > 60:
                                            st.metric("Fiabilite", "Bonne", delta="Bon")
                                        else:
                                            st.metric("Fiabilite", "Moyenne", delta="A Verifier")
                                    
                                    st.markdown("### Probabilites Detaillees")
                                    probs_data = []
                                    for classe, prob in sorted(result['probabilites'].items(), key=lambda x: x[1], reverse=True):
                                        probs_data.append({
                                            'Classe': classe,
                                            'Probabilite (%)': f"{prob*100:.2f}",
                                            'Valeur': prob
                                        })
                                    
                                    probs_df = pd.DataFrame(probs_data)
                                    st.dataframe(probs_df[['Classe', 'Probabilite (%)']], use_container_width=True, hide_index=True)
                                    
                                    st.bar_chart(probs_df.set_index('Classe')['Valeur'])
                            except Exception as e:
                                st.error(f"Erreur lors de la prediction: {str(e)}")
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            elif os.path.exists(image_path):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Apercu de l'image")
                    try:
                        image = Image.open(image_path)
                        st.image(image, use_column_width=True)
                    except:
                        st.error(f"Impossible d'afficher l'image depuis le chemin: {image_path}")
                
                with col2:
                    st.markdown("### Resultat de la Prediction")
                    if st.button("Executer la Prediction", use_container_width=True):
                        with st.spinner("Prediction en cours..."):
                            try:
                                result = cnn_app.predict_image(image_path, show_plot=False)
                                
                                if result:
                                    st.session_state.predictions_history.append({
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'image': os.path.basename(image_path),
                                        'classe': result['classe'],
                                        'confiance': result['confiance_pourcentage']
                                    })
                                    
                                    st.success(f"Classe Predite: {result['classe']}")
                                    st.metric("Confiance", f"{result['confiance_pourcentage']:.2f}%")
                                    
                                    st.markdown("### Top 5 Classes par Probabilite")
                                    top_5 = sorted(result['probabilites'].items(), key=lambda x: x[1], reverse=True)[:5]
                                    top_data = []
                                    for classe, prob in top_5:
                                        top_data.append({'Classe': classe, 'Probabilite (%)': f"{prob*100:.2f}"})
                                    st.dataframe(pd.DataFrame(top_data), use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Erreur lors de la prediction: {str(e)}")
        
        # TAB 2: Traitement par dossier
        with tab2:
            st.markdown("## Traitement par Dossier")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                folder_name = st.text_input(
                    "Nom du dossier contenant les images",
                    "bearing",
                    key="folder_name"
                )
            
            with col2:
                st.markdown("")
                st.markdown("")
                if st.button("Demarrer le Traitement", use_container_width=True):
                    process = True
                else:
                    process = False
            
            if process:
                if os.path.isdir(folder_name):
                    with st.spinner(f"Traitement des images du dossier {folder_name}..."):
                        try:
                            results = cnn_app.predict_folder(folder_name)
                            
                            st.success(f"Traitement termine: {len(results)} images traitees!")
                            
                            results_data = []
                            for r in results:
                                results_data.append({
                                    'Image': os.path.basename(r['image_path']),
                                    'Classe': r['classe'],
                                    'Confiance (%)': f"{r['confiance_pourcentage']:.2f}"
                                })
                            
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            st.markdown("## Statistiques du Traitement")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total traite", len(results))
                            with col2:
                                avg_conf = np.mean([r['confiance_pourcentage'] for r in results])
                                st.metric("Confiance moyenne", f"{avg_conf:.2f}%")
                            with col3:
                                top_class = results_df['Classe'].mode()
                                st.metric("Classe dominante", top_class[0] if len(top_class) > 0 else "N/A")
                            with col4:
                                st.metric("Precision min-max", f"{results_df['Confiance (%)'].min()}-{results_df['Confiance (%)'].max()}%")
                        except Exception as e:
                            st.error(f"Erreur lors du traitement: {str(e)}")
                else:
                    st.error(f"Le dossier '{folder_name}' n'existe pas ou n'est pas accessible.")

# PAGE 5: HISTORIQUE
elif page == "Historique":
    st.markdown("# Historique des Predictions")
    st.markdown("---")
    
    if len(st.session_state.predictions_history) > 0:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        st.markdown("## Tableau des Predictions")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("## Statistiques")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total predictions", len(history_df))
        with col2:
            avg_conf = history_df['confiance'].mean()
            st.metric("Confiance moyenne", f"{avg_conf:.2f}%")
        with col3:
            top_class = history_df['classe'].mode()
            st.metric("Classe la plus predite", top_class[0] if len(top_class) > 0 else "N/A")
        with col4:
            st.metric("Confiances (min-max)", f"{history_df['confiance'].min():.0f}%-{history_df['confiance'].max():.0f}%")
        
        st.markdown("---")
        st.markdown("## Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Evolution des Confiances")
            st.line_chart(history_df['confiance'])
        
        with col2:
            st.markdown("### Distribution par Classe")
            classe_counts = history_df['classe'].value_counts()
            st.bar_chart(classe_counts)
        
        st.markdown("---")
        
        if st.button("Effacer l'Historique", use_container_width=True):
            st.session_state.predictions_history = []
            st.success("Historique efface")
            st.rerun()
    else:
        st.info("Aucune prediction effectuee pour le moment. Commencez par faire une prediction dans l'onglet 'Predictions'.")

# PAGE 6: CONFIGURATION
elif page == "Configuration":
    st.markdown("# Configuration et Parametres")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## Configuration de l'Entraînement")
        
        with st.expander("Parametres du Reseau", expanded=False):
            model_type = st.selectbox(
                "Type d'architecture",
                ["simple", "deep", "lightweight", "vgg_style"],
                help="Selectionner l'architecture du modele CNN"
            )
            
            epochs = st.slider(
                "Nombre d'epochs",
                5, 200, 30,
                5,
                help="Nombre d'iterations d'entraînement"
            )
            
            batch_size = st.slider(
                "Batch size",
                8, 128, 32,
                8,
                help="Nombre d'images par batch"
            )
            
            validation_split = st.slider(
                "Ratio validation",
                0.1, 0.5, 0.2,
                0.05,
                help="Pourcentage de donnees pour validation"
            )
        
        with st.expander("Optimiseur et Regularisation", expanded=False):
            optimizer = st.selectbox(
                "Optimiseur",
                ["Adam", "SGD", "RMSprop"],
                help="Algorithme d'optimisation"
            )
            
            learning_rate = st.slider(
                "Taux d'apprentissage",
                0.00001, 0.1, 0.001,
                step=0.00001,
                format="%.5f",
                help="Taux d'apprentissage (Learning Rate)"
            )
            
            dropout = st.slider(
                "Taux de Dropout",
                0.0, 0.9, 0.5,
                0.1,
                help="Dropout pour eviter l'overfitting"
            )
            
            l2_regularization = st.slider(
                "L2 Regularization",
                0.0, 0.01, 0.0001,
                0.0001,
                format="%.5f",
                help="Penalite de regularisation L2"
            )
    
    with col2:
        st.markdown("## Configuration de l'Application")
        
        with st.expander("Donnees et Traitement", expanded=False):
            image_size = st.number_input(
                "Taille des images (pixels)",
                64, 512, 100,
                step=1,
                help="Dimensions cibles des images (100x100x3)"
            )
            
            train_test_split = st.slider(
                "Split Train/Test",
                0.5, 0.95, 0.8,
                0.05,
                format="%.2f",
                help="Pourcentage d'images pour l'entraînement"
            )
            
            augmentation_enabled = st.checkbox(
                "Augmentation de donnees",
                value=True,
                help="Appliquer l'augmentation de donnees"
            )
        
        with st.expander("Fichiers et Modeles", expanded=False):
            csv_path = st.text_input(
                "Chemin du fichier CSV",
                "data_set.csv",
                help="Chemin du fichier de donnees"
            )
            
            model_save_path = st.text_input(
                "Chemin de sauvegarde du modele",
                "mon_modele_rgb.keras",
                help="Chemin ou sauvegarder le modele"
            )
    
    st.markdown("---")
    st.markdown("## A Propos de l'Application")
    
    st.markdown("""
    ### Informations Generales
    
    **Nom**: Systeme de Classification de Pieces Automobiles
    
    **Version**: 1.0
    
    **Date de creation**: Janvier 2026
    
    **Type de modele**: Reseau de neurones convolutif (CNN)
    
    **Nombre de classes**: 14 types de pieces
    
    **Precision attendue**: 90-95%
    
    ### Technologies Utilisees
    
    **Framework d'apprentissage profond**: TensorFlow/Keras
    
    **Traitement d'images**: OpenCV
    
    **Interface web**: Streamlit
    
    **Langage de programmation**: Python 3.8+
    
    **Visualisation de donnees**: Pandas, Matplotlib
    
    ### Architecture Technologique
    
    L'application repose sur une architecture modulaire avec:
    - Module de chargement et preprocessing des donnees
    - Module de definition et entraînement du modele CNN
    - Module d'inference pour les predictions
    - Interface web interactive pour l'utilisateur
    
    ### Support et Documentation
    
    Pour plus d'informations, consultez la documentation fournie avec l'application.
    """)

# PAGE 7: LOGO PERSONNALISE
elif page == "Logo Personnalise":
    st.markdown("# Gestion du Logo Personnalise")
    st.markdown("---")
    
    st.markdown("## Ajouter votre Logo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload un Logo")
        logo_file = st.file_uploader(
            "Selectionner un fichier image pour le logo",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Format recommande: PNG avec fond transparent"
        )
        
        if logo_file is not None:
            logo_path = f"logo_{logo_file.name}"
            with open(logo_path, "wb") as f:
                f.write(logo_file.getbuffer())
            
            st.session_state.logo_path = logo_path
            st.success("Logo charge avec succes!")
    
    with col2:
        st.markdown("### Apercu du Logo")
        if st.session_state.logo_path and os.path.exists(st.session_state.logo_path):
            try:
                logo = Image.open(st.session_state.logo_path)
                st.image(logo, use_column_width=True)
            except:
                st.error("Impossible d'afficher le logo")
        else:
            st.info("Aucun logo charge pour le moment")
    
    st.markdown("---")
    
    st.markdown("## Gestion du Logo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Reinitialiser le Logo", use_container_width=True):
            st.session_state.logo_path = None
            st.success("Logo reinitialise")
            st.rerun()
    
    with col2:
        if st.session_state.logo_path and os.path.exists(st.session_state.logo_path):
            if st.button("Supprimer le Logo", use_container_width=True):
                try:
                    os.remove(st.session_state.logo_path)
                    st.session_state.logo_path = None
                    st.success("Logo supprime")
                    st.rerun()
                except:
                    st.error("Erreur lors de la suppression du logo")

# FOOTER
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Classification de Pieces Automobiles - Systeme Intelligent</p>
    <p>Application CNN pour la reconnaissance automatique d'images automobiles</p>
    <p style="color: #999;">Copyright 2026 - Tous droits reserves</p>
</div>
""", unsafe_allow_html=True)
