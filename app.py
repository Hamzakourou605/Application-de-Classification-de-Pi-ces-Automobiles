"""
Module pour gérer le modèle CNN de classification de pièces automobiles
Gère le chargement, la prédiction et le réentraînement du modèle
"""
import numpy as np
import pandas as pd
import cv2
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model


class AutomobilePartsCNN:
    """Classe pour gérer le modèle CNN de classification de pièces automobiles"""
    
    def __init__(self, model_path="mon_modele_rgb.keras", label_encoder_path="label_encoder.pkl", csv_path="data_set.csv"):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.csv_path = csv_path
        self.model = None
        self.label_encoder = None
        self.classes = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle et le label encoder s'ils existent"""
        model_loaded = False
        encoder_loaded = False
        
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"[OK] Modèle chargé depuis {self.model_path}")
                model_loaded = True
            except Exception as e:
                print(f"[ERREUR] Erreur lors du chargement du modèle: {e}")
                self.model = None
        else:
            print(f"[AVERTISSEMENT] Modèle non trouvé: {self.model_path}")
        
        if os.path.exists(self.label_encoder_path):
            try:
                with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                    self.classes = list(self.label_encoder.classes_)
                print(f"[OK] Label encoder chargé depuis {self.label_encoder_path}")
                encoder_loaded = True
            except Exception as e:
                print(f"[ERREUR] Erreur lors du chargement du label encoder: {e}")
                self.label_encoder = None
                self.classes = None
        else:
            print(f"[AVERTISSEMENT] Label encoder non trouvé: {self.label_encoder_path}")
        
        return model_loaded and encoder_loaded
    
    def preprocess_image(self, image_data):
        """
        Prépare une image pour la prédiction
        image_data: image BGR (numpy array) ou pixels aplatis
        """
        # Si c'est une ligne de CSV (pixels aplatis en 1D)
        if isinstance(image_data, np.ndarray) and image_data.ndim == 1:
            # Redimensionner à 100x100x3
            image_resized = image_data.reshape(100, 100, 3)
        else:
            # Si c'est déjà une image BGR
            image_resized = cv2.resize(image_data, (100, 100))
        
        # Normaliser [0, 1]
        image_normalized = image_resized.astype("float32") / 255.0
        # Reshape pour le CNN: (1, 100, 100, 3)
        image_final = np.expand_dims(image_normalized, axis=0)
        return image_final
    
    def predict(self, image_data, confidence_threshold=0.5):
        """
        Prédit la classe d'une image
        image_data: image BGR (numpy array) ou pixels aplatis
        Returns: (name, confidence) ou (None, 0) si modèle non disponible
        """
        if self.model is None or self.label_encoder is None:
            return None, 0.0
        
        try:
            image_final = self.preprocess_image(image_data)
            prediction = self.model.predict(image_final, verbose=0)
            class_id = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            if confidence > confidence_threshold:
                name = self.label_encoder.classes_[class_id]
                return name, confidence
            else:
                return None, confidence
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            return None, 0.0
    
    def train_model(self, epochs=65, batch_size=32, verbose=1):
        """
        Entraîne ou réentraîne le modèle avec le dataset CSV
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Le fichier CSV {self.csv_path} n'existe pas")
        
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT DU MODÈLE CNN - PIÈCES AUTOMOBILES")
        print(f"{'='*60}")
        
        print(f"\n[1] Chargement du dataset depuis {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        print(f"    ✓ {len(df)} samples chargés")
        
        # Séparation X (pixels) et y (labels)
        print(f"\n[2] Séparation des features et labels...")
        X = df.drop('label', axis=1).values
        y = df['label'].values
        print(f"    ✓ Features shape: {X.shape}")
        print(f"    ✓ Labels shape: {y.shape}")
        
        # Encodage des labels
        print(f"\n[3] Encodage des labels...")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = list(self.label_encoder.classes_)
        num_classes = len(np.unique(y_encoded))
        print(f"    ✓ Nombre de classes: {num_classes}")
        print(f"    ✓ Classes: {self.classes}")
        
        # Normalisation [0, 1]
        print(f"\n[4] Normalisation des données...")
        X = X.astype('float32') / 255.0
        print(f"    ✓ Plage des données: [{X.min():.4f}, {X.max():.4f}]")
        
        # Reshape pour CNN (100x100x3)
        print(f"\n[5] Reshape des images (100x100x3)...")
        X = X.reshape(-1, 100, 100, 3)
        print(f"    ✓ Nouvelles dimensions: {X.shape}")
        
        # Division train/test
        print(f"\n[6] Division train/test (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"    ✓ Training set: {X_train.shape[0]} samples")
        print(f"    ✓ Test set: {X_test.shape[0]} samples")
        
        # Construction du modèle
        print(f"\n[7] Construction du modèle CNN...")
        if self.model is None or self.model.output_shape[-1] != num_classes:
            self.model = models.Sequential([
                # Bloc 1
                layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                            input_shape=(100, 100, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Bloc 2
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Bloc 3
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Couches denses
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])
            print(f"    ✓ Nouveau modèle créé")
        
        # Compilation
        print(f"\n[8] Compilation du modèle...")
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"    ✓ Modèle compilé")
        
        # Affichage architecture
        print(f"\n[9] Architecture du modèle:")
        self.model.summary()
        
        # Entraînement
        print(f"\n[10] Début de l'entraînement ({epochs} epochs)...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=verbose
        )
        
        # Évaluation
        print(f"\n[11] Évaluation sur l'ensemble de test...")
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"    ✓ Loss: {test_loss:.4f}")
        print(f"    ✓ Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Sauvegarde
        print(f"\n[12] Sauvegarde du modèle et du label encoder...")
        self.save_model()
        self.save_label_encoder()
        
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"{'='*60}\n")
        
        return history, test_acc
    
    def save_model(self):
        """Sauvegarde le modèle"""
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"    ✓ Modèle sauvegardé dans {self.model_path}")
    
    def save_label_encoder(self):
        """Sauvegarde le label encoder"""
        if self.label_encoder is not None:
            with open(self.label_encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"    ✓ Label encoder sauvegardé dans {self.label_encoder_path}")
    
    def get_classes(self):
        """Retourne la liste des classes (noms des pièces)"""
        if self.label_encoder is not None:
            return list(self.label_encoder.classes_)
        return []
    
    def test_accuracy(self):
        """Teste la précision du modèle sur l'ensemble de test"""
        if self.model is None or self.label_encoder is None:
            print("Erreur: Modèle non chargé.")
            return
        
        print(f"\n{'='*60}")
        print(f"TEST DE PRÉCISION DU MODÈLE")
        print(f"{'='*60}")
        
        if not os.path.exists(self.csv_path):
            print(f"Erreur: Fichier {self.csv_path} introuvable!")
            return
        
        # Charger et préparer les données
        df = pd.read_csv(self.csv_path)
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        X = X.astype('float32') / 255.0
        X = X.reshape(-1, 100, 100, 3)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Évaluation
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nPerte (Loss): {loss:.4f}")
        print(f"Précision (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    def predict_image(self, image_path, show_plot=False):
        """
        Prédit la classe d'une image
        
        Args:
            image_path: Chemin vers l'image à prédire
            show_plot: Afficher l'image avec la prédiction
            
        Returns:
            dict: Contenant la classe prédite et la confiance
        """
        if self.model is None or self.label_encoder is None:
            print("Erreur: Modèle non chargé.")
            return None
        
        # Vérifier l'existence du fichier
        if not os.path.exists(image_path):
            print(f"Erreur: Image {image_path} introuvable!")
            return None
        
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur: Impossible de lire l'image {image_path}")
            return None
        
        # Prétraitement
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_batch = self.preprocess_image(img)
        
        # Prédiction
        predictions = self.model.predict(img_batch, verbose=0)
        class_id = np.argmax(predictions)
        class_name = self.label_encoder.classes_[class_id]
        confidence = float(np.max(predictions))
        
        # Résultat
        result = {
            'classe': class_name,
            'confiance': confidence,
            'confiance_pourcentage': confidence * 100,
            'probabilites': {self.label_encoder.classes_[i]: float(predictions[0][i]) 
                            for i in range(len(self.classes))},
            'image_path': image_path
        }
        
        # Affichage
        print(f"\n{'='*60}")
        print(f"PRÉDICTION: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        print(f"Classe prédite: {class_name}")
        print(f"Confiance: {confidence*100:.2f}%")
        
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.imshow(img_rgb)
            plt.title(f"Prédiction: {class_name} ({confidence*100:.2f}%)")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return result
    
    def predict_folder(self, folder_path):
        """
        Prédit les classes pour toutes les images d'un dossier
        
        Args:
            folder_path: Chemin vers le dossier contenant les images
            
        Returns:
            list: Liste des résultats de prédictions
        """
        if self.model is None or self.label_encoder is None:
            print("Erreur: Modèle non chargé.")
            return []
        
        if not os.path.isdir(folder_path):
            print(f"Erreur: {folder_path} n'est pas un dossier valide!")
            return []
        
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        print(f"\n{'='*60}")
        print(f"PRÉDICTIONS POUR LE DOSSIER: {folder_path}")
        print(f"Nombre d'images trouvées: {len(image_files)}")
        print(f"{'='*60}\n")
        
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(folder_path, image_file)
            print(f"\n[{i}/{len(image_files)}] Traitement: {image_file}")
            result = self.predict_image(image_path, show_plot=False)
            if result:
                results.append(result)
        
        # Résumé
        print(f"\n{'='*60}")
        print("RÉSUMÉ DES PRÉDICTIONS")
        print(f"{'='*60}")
        df_results = pd.DataFrame([{
            'Image': r['image_path'].split(os.sep)[-1],
            'Classe': r['classe'],
            'Confiance': f"{r['confiance_pourcentage']:.2f}%"
        } for r in results])
        print(df_results.to_string(index=False))
        print()
        
        return results


def main():
    """Fonction principale pour utiliser l'application"""
    app = AutomobilePartsCNN()
    
    # Menu interactif
    while True:
        print("\n" + "=" * 60)
        print("APPLICATION CNN - CLASSIFICATION DE PIÈCES AUTOMOBILES")
        print("=" * 60)
        print("1. Entraîner le modèle")
        print("2. Charger le modèle existant")
        print("3. Prédire une image")
        print("4. Prédire un dossier")
        print("5. Tester la précision")
        print("6. Voir les classes disponibles")
        print("7. Quitter")
        print("=" * 60)
        
        choice = input("Sélectionnez une option (1-7): ").strip()
        
        if choice == '1':
            app.train_model()
        
        elif choice == '2':
            app.load_model()
        
        elif choice == '3':
            if app.model is None:
                app.load_model()
            image_path = input("Entrez le chemin de l'image: ").strip()
            app.predict_image(image_path, show_plot=True)
        
        elif choice == '4':
            if app.model is None:
                app.load_model()
            folder_path = input("Entrez le chemin du dossier: ").strip()
            app.predict_folder(folder_path)
        
        elif choice == '5':
            if app.model is None:
                app.load_model()
            app.test_accuracy()
        
        elif choice == '6':
            classes = app.get_classes()
            if classes:
                print(f"\n{'='*60}")
                print("CLASSES DISPONIBLES:")
                print(f"{'='*60}")
                for i, cls in enumerate(classes, 1):
                    print(f"{i}. {cls}")
            else:
                print("Aucune classe disponible. Veuillez d'abord entraîner le modèle.")
        
        elif choice == '7':
            print("\nAu revoir!")
            break
        
        else:
            print("Option invalide. Veuillez réessayer.")


if __name__ == "__main__":
    main()
