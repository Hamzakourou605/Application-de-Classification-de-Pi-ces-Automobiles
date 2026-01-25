"""
Utilitaires pour l'application de classification de pieces automobiles
Fournit des classes pour la gestion des donnees, l'exportation des resultats et l'analyse
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
from datetime import datetime
from pathlib import Path


class ImagePreprocessor:
    """Preprocessor pour les images"""
    
    def __init__(self, target_size=(100, 100)):
        self.target_size = target_size
    
    def preprocess(self, image_path):
        """Preprocesse une image pour le modele"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            img = cv2.resize(img, self.target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Erreur lors du preprocessing de {image_path}: {e}")
            return None
    
    def preprocess_batch(self, folder_path):
        """Preprocesse un dossier d'images"""
        images = []
        filenames = []
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                file_path = os.path.join(folder_path, file)
                processed = self.preprocess(file_path)
                if processed is not None:
                    images.append(processed[0])
                    filenames.append(file)
        
        if images:
            return np.array(images), filenames
        return None, []


class DatasetManager:
    """Gestionnaire pour les donnees du dataset"""
    
    def __init__(self, csv_path='data_set.csv'):
        self.csv_path = csv_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Charge les donnees du CSV"""
        try:
            self.data = pd.read_csv(self.csv_path)
            return self.data
        except Exception as e:
            print(f"Erreur lors du chargement du CSV: {e}")
            return None
    
    def get_classes(self):
        """Retourne la liste des classes"""
        if self.data is not None:
            last_column = self.data.columns[-1]
            return sorted(self.data[last_column].unique().tolist())
        return []
    
    def get_class_distribution(self):
        """Retourne la distribution des classes"""
        if self.data is not None:
            last_column = self.data.columns[-1]
            return self.data[last_column].value_counts().sort_index()
        return None
    
    def get_statistics(self):
        """Retourne les statistiques du dataset"""
        if self.data is not None:
            return {
                'total_samples': len(self.data),
                'num_features': len(self.data.columns) - 1,
                'num_classes': len(self.get_classes()),
                'classes': self.get_classes()
            }
        return {}
    
    def get_class_info(self):
        """Retourne les informations sur chaque classe"""
        classes_info = {
            'bearing': 'Roulement - Composant rotatif essentiel',
            'bevel-gear': 'Engrenage conique - Transmission de puissance',
            'clutch': 'Embrayage - Connexion moteur/transmission',
            'cylinder': 'Cylindre - Composant du moteur',
            'filter': 'Filtre - Purification des fluides',
            'fuel-tank': 'Reservoir - Stockage carburant',
            'helical-gear': 'Engrenage helicoidale - Transmission silencieuse',
            'piston': 'Piston - Composant moteur critical',
            'rack-pinion': 'Cremaillere - Systeme de direction',
            'shocker': 'Amortisseur - Suspension',
            'spark-plug': 'Bougie - Allumage moteur',
            'spur-gear': 'Engrenage droit - Transmission puissance',
            'valve': 'Soupape - Controle des fluides',
            'wheel': 'Roue - Mobilite du vehicule'
        }
        return classes_info
    
    @staticmethod
    def scan_all_folders(root_path='.'):
        """Scanne tous les dossiers et compte les images par classe"""
        stats = {}
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for folder_name in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder_name)
            
            # Ignorer les fichiers et certains dossiers
            if not os.path.isdir(folder_path):
                continue
            if folder_name in ['.', '..', '__pycache__', 'venv', '.git']:
                continue
            
            # Compter les images
            image_count = 0
            try:
                for file in os.listdir(folder_path):
                    if any(file.lower().endswith(fmt) for fmt in supported_formats):
                        image_count += 1
            except PermissionError:
                continue
            
            if image_count > 0:
                stats[folder_name] = {
                    'count': image_count,
                    'path': folder_path
                }
        
        return stats


class ResultsExporter:
    """Exportateur pour les resultats de prediction"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_json(self, results, filename=None):
        """Exporte les resultats en JSON"""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            print(f"Erreur lors de l'export JSON: {e}")
            return None
    
    def export_csv(self, results_list, filename=None):
        """Exporte les resultats en CSV"""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            df = pd.DataFrame(results_list)
            df.to_csv(filepath, index=False, encoding='utf-8')
            return filepath
        except Exception as e:
            print(f"Erreur lors de l'export CSV: {e}")
            return None
    
    def export_report(self, report_data, filename=None):
        """Exporte un rapport texte"""
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_data)
            return filepath
        except Exception as e:
            print(f"Erreur lors de l'export du rapport: {e}")
            return None


class ModelAnalyzer:
    """Analyseur pour le modele entrainé"""
    
    @staticmethod
    def get_model_info(model):
        """Retourne les informations du modele"""
        try:
            info = {
                'total_params': model.count_params(),
                'layers': len(model.layers),
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'trainable_params': sum([np.prod(w.shape) for w in model.trainable_weights]),
                'non_trainable_params': sum([np.prod(w.shape) for w in model.non_trainable_weights])
            }
            return info
        except Exception as e:
            print(f"Erreur lors de l'analyse du modele: {e}")
            return {}
    
    @staticmethod
    def get_layer_info(model):
        """Retourne les details des couches"""
        layers_info = []
        try:
            for i, layer in enumerate(model.layers):
                layer_detail = {
                    'index': i,
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'output_shape': str(layer.output_shape),
                    'params': layer.count_params()
                }
                layers_info.append(layer_detail)
            return layers_info
        except Exception as e:
            print(f"Erreur lors de l'extraction des details des couches: {e}")
            return []
    
    @staticmethod
    def format_model_summary(model):
        """Retourne un resume formaté du modele"""
        try:
            info = ModelAnalyzer.get_model_info(model)
            summary = f"""
=== RESUME DU MODELE ===
Parametres totaux: {info.get('total_params', 'N/A'):,}
Parametres entrainables: {info.get('trainable_params', 'N/A'):,}
Parametres non-entrainables: {info.get('non_trainable_params', 'N/A'):,}
Nombre de couches: {info.get('layers', 'N/A')}
Forme d'entree: {info.get('input_shape', 'N/A')}
Forme de sortie: {info.get('output_shape', 'N/A')}
"""
            return summary
        except Exception as e:
            return f"Erreur lors du formatage du resume: {e}"


class ConfigurationManager:
    """Gestionnaire de configurations"""
    
    DEFAULT_CONFIG = {
        'model_path': 'mon_modele_rgb.keras',
        'data_path': 'data_set.csv',
        'image_size': (100, 100),
        'batch_size': 32,
        'epochs': 30,
        'train_split': 0.8,
        'random_state': 42,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy']
    }
    
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """Charge la configuration depuis un fichier"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")
    
    def save_config(self, config_file=None):
        """Sauvegarde la configuration"""
        filepath = config_file or self.config_file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return None
    
    def get(self, key, default=None):
        """Obtient une valeur de configuration"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Definit une valeur de configuration"""
        self.config[key] = value
