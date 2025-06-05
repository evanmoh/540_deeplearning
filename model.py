"""
Model Training and Prediction Module
Implements three approaches: Naive, Classical ML, and Deep Learning
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
import argparse
import json
import time
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available, deep learning models will be skipped")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Complete model training pipeline for all three approaches:
    1. Naive approach (threshold-based)
    2. Classical ML (Random Forest, SVM, Logistic Regression)
    3. Deep Learning (Neural Network)
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize model trainer
        
        Args:
            data_dir: Directory containing processed data
            models_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.processed_dir = self.data_dir / "processed"
        self.outputs_dir = self.data_dir / "outputs"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Storage for models and results
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.training_history = {}
        
        # Random state for reproducibility
        self.random_state = 42
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load processed features and labels
        
        Returns:
            Features array, labels array, patient IDs
        """
        features_path = self.processed_dir / "features.csv"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        logger.info(f"📊 Loading features from {features_path}")
        features_df = pd.read_csv(features_path)
        
        # Separate features, labels, and IDs
        X = features_df.drop(['patient_id', 'label'], axis=1).values
        y = features_df['label'].values
        patient_ids = features_df['patient_id'].tolist()
        
        logger.info(f"📊 Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y, patient_ids
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features array
            y: Labels array
            
        Returns:
            Train, validation, and test splits
        """
        # First split: train/temp (60/40)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.random_state, stratify=y
        )
        
        # Second split: val/test (20/20 from original)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"📊 Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_naive_approach(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train naive threshold-based approach
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Model performance results
        """
        logger.info("🎯 Training Naive Approach (Threshold-based)...")
        
        start_time = time.time()
        
        # Use first feature (intensity_mean) as the primary decision feature
        feature_values = X_train[:, 0]
        
        # Find optimal threshold using grid search
        best_threshold = None
        best_score = 0
        best_predictions = None
        
        thresholds = np.linspace(feature_values.min(), feature_values.max(), 100)
        
        for threshold in thresholds:
            predictions = (feature_values > threshold).astype(int)
            score = accuracy_score(y_train, predictions)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_predictions = predictions
        
        # Test performance
        test_predictions = (X_test[:, 0] > best_threshold).astype(int)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(y_test, test_predictions, zero_division=0)
        test_recall = recall_score(y_test, test_predictions, zero_division=0)
        test_f1 = f1_score(y_test, test_predictions, zero_division=0)
        
        training_time = time.time() - start_time
        
        # Create model object
        naive_model = {
            'type': 'naive_threshold',
            'threshold': best_threshold,
            'feature_index': 0,
            'feature_name': 'intensity_mean'
        }
        
        # Save model
        model_path = self.models_dir / "naive_model.joblib"
        joblib.dump(naive_model, model_path)
        
        # Store results
        results = {
            'model_type': 'Naive Threshold',
            'approach': 'Naive',
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'training_time': training_time,
            'predictions': test_predictions,
            'model_path': str(model_path),
            'threshold': best_threshold
        }
        
        self.models['Naive'] = naive_model
        self.results['Naive'] = results
        
        logger.info(f"✅ Naive model trained - Accuracy: {test_accuracy:.3f}")
        
        return results
    
    def train_classical_ml(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train classical ML approaches
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            X_val, y_val: Validation data (optional)
            
        Returns:
            Dictionary of model results
        """
        logger.info("🌳 Training Classical ML Approaches...")
        
        # Define algorithms with hyperparameters
        algorithms = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        classical_results = {}
        
        for name, model in algorithms.items():
            logger.info(f"🔧 Training {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                f1 = f1_score(y_test, predictions, zero_division=0)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                training_time = time.time() - start_time
                
                # Save model
                model_filename = f"{name.lower().replace(' ', '_')}_model.joblib"
                model_path = self.models_dir / model_filename
                joblib.dump(model, model_path)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = np.abs(model.coef_[0])
                
                # Store results
                results = {
                    'model_type': name,
                    'approach': 'Classical ML',
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'training_time': training_time,
                    'predictions': predictions,
                    'model_path': str(model_path)
                }
                
                self.models[name] = model
                classical_results[name] = results
                
                logger.info(f"✅ {name} trained - Accuracy: {accuracy:.3f} (CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {name}: {e}")
                continue
        
        # Store all classical results
        self.results.update(classical_results)
        
        return classical_results
    
    def train_deep_learning(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train deep learning approach (Neural Network)
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Model performance results
        """
        if not TF_AVAILABLE:
            logger.warning("⚠️  TensorFlow not available, skipping deep learning")
            return {}
        
        logger.info("🧠 Training Deep Learning Approach (Neural Network)...")
        
        start_time = time.time()
        
        try:
            # Create model architecture
            model = keras.Sequential([
                layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.4),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Define callbacks
            model_callbacks = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train model
            logger.info("🔧 Training neural network...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=min(32, len(X_train) // 4),
                callbacks=model_callbacks,
                verbose=1
            )
            
            # Make predictions
            predictions_proba = model.predict(X_test, verbose=0)
            predictions = (predictions_proba > 0.5).astype(int).flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            # Try to calculate AUC
            try:
                auc = roc_auc_score(y_test, predictions_proba.flatten())
            except:
                auc = 0.5
            
            training_time = time.time() - start_time
            
            # Save model
            model_path = self.models_dir / "deep_learning_model.h5"
            model.save(str(model_path))
            
            # Store results
            results = {
                'model_type': 'Deep Learning',
                'approach': 'Deep Learning',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'training_time': training_time,
                'predictions': predictions,
                'model_path': str(model_path),
                'epochs_trained': len(history.history['loss'])
            }
            
            self.models['Deep Learning'] = model
            self.results['Deep Learning'] = results
            self.training_history['Deep Learning'] = history.history
            
            logger.info(f"✅ Deep Learning model trained - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to train deep learning model: {e}")
            return {}
    
    def train_all_approaches(self) -> Dict:
        """
        Train all three approaches and compare performance
        
        Returns:
            Dictionary containing all results
        """
        logger.info("🤖 Starting comprehensive model training...")
        
        # Load data
        X, y, patient_ids = self.load_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Train all approaches
        logger.info("🎯 Training all three approaches...")
        
        # 1. Naive approach
        naive_results = self.train_naive_approach(X_train, y_train, X_test, y_test)
        
        # 2. Classical ML approaches
        classical_results = self.train_classical_ml(X_train, y_train, X_test, y_test, X_val, y_val)
        
        # 3. Deep Learning approach
        dl_results = self.train_deep_learning(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Generate comprehensive comparison
        self.save_results_and_comparison()
        
        logger.info("✅ All model training completed!")
        
        return self.results
    
    def save_results_and_comparison(self):
        """Save comprehensive results and generate comparison reports"""
        logger.info("📊 Generating results and comparisons...")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for name, result in self.results.items():
            if result:  # Skip empty results
                comparison_data.append({
                    'Model': name,
                    'Approach': result.get('approach', 'Unknown'),
                    'Accuracy': result.get('accuracy', 0),
                    'Precision': result.get('precision', 0),
                    'Recall': result.get('recall', 0),
                    'F1-Score': result.get('f1_score', 0),
                    'Training_Time_sec': result.get('training_time', 0),
                    'Model_Path': result.get('model_path', '')
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
            
            # Save comparison
            comparison_path = self.outputs_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"📊 Model comparison saved to: {comparison_path}")
        
        # Save feature importance if available
        if self.feature_importance:
            importance_data = []
            
            for model_name, importances in self.feature_importance.items():
                for i, importance in enumerate(importances):
                    importance_data.append({
                        'model': model_name,
                        'feature_index': i,
                        'feature_name': f'feature_{i}',
                        'importance': importance
                    })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                
                # Calculate average importance across models
                avg_importance = importance_df.groupby(['feature_index', 'feature_name'])['importance'].mean().reset_index()
                avg_importance = avg_importance.sort_values('importance', ascending=False)
                
                importance_path = self.outputs_dir / "feature_importance.csv"
                avg_importance.to_csv(importance_path, index=False)
                logger.info(f"🔍 Feature importance saved to: {importance_path}")
        
        # Save detailed results
        results_path = self.outputs_dir / "detailed_results.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for name, result in self.results.items():
            if result:
                json_result = result.copy()
                # Convert numpy arrays to lists
                if 'predictions' in json_result:
                    json_result['predictions'] = json_result['predictions'].tolist()
                json_results[name] = json_result
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_path}")
        
        # Print summary
        if comparison_data:
            print("\n" + "="*80)
            print("📊 MODEL COMPARISON SUMMARY")
            print("="*80)
            for _, row in comparison_df.iterrows():
                print(f"{row['Model']:20} | {row['Approach']:15} | Acc: {row['Accuracy']:.3f} | "
                      f"F1: {row['F1-Score']:.3f} | Time: {row['Training_Time_sec']:.1f}s")
            print("="*80)
            
            best_model = comparison_df.iloc[0]
            print(f"🏆 BEST MODEL: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.3f})")
            print("="*80)

class ModelPredictor:
    """Handle predictions from trained models"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model predictor
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available trained models"""
        logger.info("📥 Loading trained models...")
        
        # Load naive model
        naive_path = self.models_dir / "naive_model.joblib"
        if naive_path.exists():
            self.models['Naive'] = joblib.load(naive_path)
            logger.info("✅ Loaded naive model")
        
        # Load classical ML models
        classical_models = {
            'Random Forest': 'random_forest_model.joblib',
            'SVM': 'svm_model.joblib',
            'Logistic Regression': 'logistic_regression_model.joblib'
        }
        
        for model_name, filename in classical_models.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"✅ Loaded {model_name}")
        
        # Load deep learning model
        if TF_AVAILABLE:
            dl_path = self.models_dir / "deep_learning_model.h5"
            if dl_path.exists():
                try:
                    self.models['Deep Learning'] = keras.models.load_model(str(dl_path))
                    logger.info("✅ Loaded Deep Learning model")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load deep learning model: {e}")
        
        logger.info(f"📥 Loaded {len(self.models)} models total")
    
    def predict_single_model(self, features: np.ndarray, model_name: str) -> Dict:
        """
        Make prediction using a single model
        
        Args:
            features: Feature array for single sample
            model_name: Name of the model to use
            
        Returns:
            Prediction results dictionary
        """
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.models[model_name]
        
        try:
            if model_name == 'Naive':
                # Naive threshold prediction
                threshold = model['threshold']
                feature_idx = model['feature_index']
                value = features[feature_idx]
                probability = 1.0 if value > threshold else 0.0
                
            elif model_name == 'Deep Learning':
                # Deep learning prediction
                prediction_proba = model.predict(features.reshape(1, -1), verbose=0)
                probability = float(prediction_proba[0][0])
                
            else:
                # Classical ML prediction
                probability = float(model.predict_proba(features.reshape(1, -1))[0][1])
            
            return {
                'model': model_name,
                'probability': probability,
                'prediction': 'Malignant' if probability > 0.5 else 'Benign',
                'confidence': abs(probability - 0.5) * 2
            }
            
        except Exception as e:
            return {'error': f'Prediction failed for {model_name}: {e}'}
    
    def predict_all_models(self, features: np.ndarray) -> Dict:
        """
        Make predictions using all available models
        
        Args:
            features: Feature array for single sample
            
        Returns:
            Dictionary of all model predictions
        """
        results = {}
        
        for model_name in self.models.keys():
            result = self.predict_single_model(features, model_name)
            results[model_name] = result
        
        return results

def main():
    """Command line interface for model training"""
    parser = argparse.ArgumentParser(description="Train lung nodule detection models")
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--models-dir', default='models', help='Models directory path')
    parser.add_argument('--approach', choices=['naive', 'classical', 'deep', 'all'], 
                       default='all', help='Which approach to train')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(data_dir=args.data_dir, models_dir=args.models_dir)
    
    try:
        if args.approach == 'all':
            # Train all approaches
            results = trainer.train_all_approaches()
        else:
            # Load data first
            X, y, _ = trainer.load_data()
            X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
            
            # Train specific approach
            if args.approach == 'naive':
                results = trainer.train_naive_approach(X_train, y_train, X_test, y_test)
            elif args.approach == 'classical':
                results = trainer.train_classical_ml(X_train, y_train, X_test, y_test, X_val, y_val)
            elif args.approach == 'deep':
                results = trainer.train_deep_learning(X_train, y_train, X_val, y_val, X_test, y_test)
        
        print(f"\n✅ MODEL TRAINING COMPLETED!")
        print(f"📊 Results saved to: {trainer.outputs_dir}")
        
    except Exception as e:
        print(f"\n❌ MODEL TRAINING FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
