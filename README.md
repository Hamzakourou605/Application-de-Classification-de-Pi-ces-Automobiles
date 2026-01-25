# Systeme de Classification Automatique de Pieces Automobiles

## Vue d'Ensemble du Projet

Cette application est un systeme intelligent de reconnaissance et de classification de pieces automobiles basé sur la vision par ordinateur. Elle utilise des algorithmes d'apprentissage profond pour analyser des images de pieces automobiles et identifier automatiquement leur type et leur fonction.

Le systeme est conçu pour être simple à utiliser tout en offrant des capacites techniques avancees. Les utilisateurs peuvent charger des modeles pre-entraines, effectuer des predictions sur des images individuelles ou en masse, et analyser les caracteristiques du dataset.

## Fonctionnement Global de l'Application

### Architecture Generale

L'application est construite autour de trois composants principaux:

1. Module de Classification (app.py) - Gere le modele CNN et les predictions
2. Interface Web (streamlit_app.py) - Fournit une interface utilisateur intuitive
3. Utilitaires (utils.py) - Fournit des fonctions de soutien pour la gestion des donnees

### Flux de Travail Principal

L'utilisateur peut suivre ce flux de travail:

1. Demarrer l'application Streamlit
2. Charger un modele pre-entraine ou entraîner un nouveau modele avec des donnees personnalisees
3. Telecharger une image ou un dossier d'images
4. Executer la prediction pour obtenir la classification et les probabilites
5. Consulter l'historique des predictions et les statistiques

## Composants Techniques Detailles

### Module de Classification (app.py)

Ce module contient la classe AutomobilePartsCNN qui gere l'ensemble du processus de classification.

Caracteristiques principales:

- Chargement et sauvegarde des modeles entraines
- Preprocessing des images (redimensionnement, normalisation)
- Generation de predictions avec calcul des probabilites
- Entraînement de nouveaux modeles avec hyperparametres personnalisables

La classe utilise pickle pour sauvegarder le label encoder, ce qui permet de s'assurer que les classes utilisees lors de l'entraînement sont correctement restaurees lors du chargement.

### Interface Utilisateur (streamlit_app.py)

L'interface est organisee en plusieurs onglets:

Onglet Principal: Affiche les informations generales du modele et les metriques principales

Onglet Entraînement: Permet de charger un modele existant ou de demarrer un nouvel entraînement. Les utilisateurs peuvent suivre la progression et voir les metriques en temps reel.

Onglet Prediction: Accepte les telechargements d'images et effectue des predictions. Les resultats incluent la classe predite, le score de confiance, et les probabilites pour chaque classe.

Onglet Analyse des Dossiers: Scanne les dossiers locaux et affiche le nombre d'images par classe, utile pour comprendre la composition du dataset.

Onglet Historique: Affiche toutes les predictions effectuees, permettant l'analyse des patterns et des performances du modele au fil du temps.

### Utilitaires (utils.py)

Contient trois classes principales:

ImagePreprocessor: Gere la preparation des images pour le modele. Redimensionne les images a 100x100 pixels, normalise les valeurs de pixels, et convertit entre les formats BGR et RGB.

DatasetManager: Accede aux donnees du CSV et fournit des statistiques sur les classes. Permet aussi de scanner les dossiers locaux pour compter les images par classe.

ResultsExporter: Exporte les resultats de predictions en format JSON ou CSV pour analyse ulterieure.

## Architecture du Modele de Reseau de Neurones

### Structure du CNN

Le modele utilise une architecture de reseau de neurones convolutifs composee de:

Couche d'Entree: Images en format 100x100 pixels avec 3 canaux de couleur (RGB)

Bloc 1 de Convolution:
- Convolution 2D avec 32 filtres, noyau 3x3
- Fonction d'activation ReLU (Rectified Linear Unit)
- MaxPooling 2x2 pour reduire la dimensionalite
- Dropout 25% pour la regularisation

Bloc 2 de Convolution:
- Convolution 2D avec 64 filtres, noyau 3x3
- Fonction d'activation ReLU
- MaxPooling 2x2
- Dropout 25%

Bloc 3 de Convolution:
- Convolution 2D avec 128 filtres, noyau 3x3
- Fonction d'activation ReLU
- MaxPooling 2x2
- Dropout 25%

Couches Denses:
- Aplatissement (Flatten) des caracteristiques extraites
- Couche Dense avec 256 neurones et activation ReLU
- Dropout 50% pour la regularisation
- Couche Dense finale avec activation Softmax pour la probabilite de chaque classe

Nombre total de parametres: Approximativement 4.35 millions

### Hyperparametres d'Entraînement

Nombre d'epochs: 65 (iterations sur l'ensemble du dataset)
Taille des batches: 32 (nombre d'images traitees simultanement)
Optimiseur: Adam (adaptive learning rate)
Fonction de perte: Sparse Categorical Crossentropy (optimisee pour plusieurs classes)
Metriques: Accuracy (pourcentage de predictions correctes)

### Processus de Normalisation

Les donnees d'entree sont normalisees comme suit:

1. Les valeurs de pixels sont converties en float32
2. Chaque pixel est divise par 255.0 pour obtenir des valeurs entre 0 et 1
3. Les images sont redimensionnees a exactement 100x100 pixels
4. Le format est assure comme RGB avec 3 canaux de couleur

Cette normalisation assure une entree coherente et optimale pour le modele.

## Processus d'Entraînement Detaille

### Preparation des Donnees

1. Chargement du fichier CSV contenant les donnees
2. Separation des features (pixels) et des labels (classes)
3. Encodage des labels en nombres entiers
4. Normalisation des donnees pixel par pixel
5. Reshape des donnees en format image 4D (nombre_images, hauteur, largeur, canaux)
6. Division aleatoire en ensemble d'entraînement (80%) et ensemble de test (20%)

### Processus d'Entraînement

1. Construction du modele CNN avec la couche d'entree ajustee au nombre de classes
2. Compilation du modele avec l'optimiseur et la fonction de perte
3. Entraînement iteratif pendant 65 epochs
4. Pour chaque epoch:
   - Le modele voit tout le dataset d'entraînement
   - Les poids sont mis a jour pour minimiser l'erreur
   - La performance est evaluee sur l'ensemble de validation (test)

### Sauvegarde apres Entraînement

Les fichiers suivants sont generes et sauvegardes:

1. mon_modele_rgb.keras - Le modele entraine avec tous ses poids
2. label_encoder.pkl - Les classes et leur mapping pour les predictions futures

## Processus de Prediction

Etapes d'une prediction:

1. Chargement de l'image depuis le disque
2. Conversion de BGR (format OpenCV) en RGB
3. Redimensionnement a 100x100 pixels
4. Normalisation des valeurs de pixels
5. Ajout d'une dimension batch
6. Passage a travers le modele CNN
7. Extraction de la classe avec la probabilite maximum
8. Calcul des probabilites pour toutes les classes
9. Retour des resultats avec la classe predite et le score de confiance

Temps de prediction par image: Moins de 100 millisecondes sur GPU ou CPU moderne.

## Classes de Pieces Automobiles Supportees

L'application peut classifier les 14 types de pieces automobiles suivants:

Bearing - Roulement
Bevel-Gear - Engrenage Conique
Clutch - Embrayage
Cylinder - Cylindre
Filter - Filtre
Fuel-Tank - Reservoir de Carburant
Helical-Gear - Engrenage Helicoidale
Piston - Piston
Rack-Pinion - Cremaillere
Shocker - Amortisseur
Spark-Plug - Bougie d'Allumage
Spur-Gear - Engrenage Droit
Valve - Soupape
Wheel - Roue

## Dataset et Donnees

### Composition du Dataset

Le dataset contient des images de 14 categories differentes de pieces automobiles.

Les donnees sont organisees comme suit:

Dossiers par classe - Chaque type de piece a son propre dossier contenant les images
CSV centralisé - Un fichier data_set.csv contenant tous les pixels aplatis et les labels

### Preprocessing du Dataset

Les images sont converties en format aplatissement (flat) ou en tenseurs 4D selon le besoin.

Chaque image est redimensionnee a 100x100 pixels pour consistance.

Les valeurs de pixels sont normalisees entre 0 et 1 pour optimiser l'apprentissage du modele.

## Guide d'Utilisation

### Installation et Configuration

1. Creer un environnement Python 3.8 ou plus recent
2. Installer les dependances: pip install -r requirements.txt
3. Placer les images d'entraînement dans les dossiers correspondants
4. Preparer le fichier data_set.csv avec les donnees

### Utilisation de l'Application

Demarrer l'application:
streamlit run streamlit_app.py

Cela ouvre l'interface web dans le navigateur par defaut.

Charger le modele:
Cliquer sur "Charger le Modele Pre-Entraine" pour utiliser un modele existant.

Entraîner un nouveau modele:
Cliquer sur "Demarrer l'Entraînement" pour entraîner avec les donnees actuelles.

Effectuer une prediction:
1. Aller a l'onglet Prediction
2. Telecharger une image
3. Cliquer sur "Executer la Prediction"
4. Consulter les resultats et probabilites

### Analyse des Donnees

Utiliser l'onglet "Analyse des Dossiers" pour voir la distribution des images par classe.

Utiliser l'onglet "Historique" pour consulter toutes les predictions passees.

## Metriques de Performance

Precision Attendue: 80-95% selon la qualite des images d'entraînement

Temps d'Inference: Moins de 100ms par image

Taux de Convergence: Stable apres 30-40 epochs

Taille du Modele: Environ 18-20 MB

## Fichiers Generés

mon_modele_rgb.keras - Modele entraine avec architecture et poids

label_encoder.pkl - Mapping des classes pour decoder les predictions

results/ - Dossier contenant les exports JSON et CSV des predictions

## Considerations pour l'Utilisation

### Qualite des Images

Les meilleures predictions sont obtenues avec des images:

- Bien eclairees avec bon contraste
- Centrees sur la piece automobiles
- De resolution adequate (minimum 100x100 pixels)
- Sans flou de mouvement

### Limitations

Le modele est specialise pour les 14 categories incluses dans l'entraînement.

Les images de pieces non vues pendant l'entraînement peuvent donner des resultats imprecis.

L'ordre des classes dépend de l'entraînement et peut varier selon les donnees utilisees.

## Troubleshooting

Si le modele ne charge pas:
- Verifier que les fichiers mon_modele_rgb.keras et label_encoder.pkl existent
- Supprimer les anciens fichiers et reentraîner le modele

Si les predictions sont imprecises:
- Verifier la qualite des images d'entraînement
- Reentraîner avec plus d'epochs (65 est le defaut)
- S'assurer que le dataset est equilibre entre les classes

Si l'application est lente:
- Verifier les ressources systeme disponibles
- Fermer les autres applications
- Utiliser un dataset plus petit pour les tests
