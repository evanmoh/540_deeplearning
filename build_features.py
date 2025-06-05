"""
Feature Extraction Module for Lung Nodule Detection
Extracts comprehensive radiological features from CT volumes
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
import argparse
from typing import List, Dict, Tuple
from scipy import ndimage
from skimage import measure, feature
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Comprehensive feature extraction for lung nodule analysis
    Extracts 26 features across 4 categories: intensity, geometric, texture, morphological
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize feature extractor
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.models_dir = Path("models")
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        
        # Feature configuration
        self.feature_names = []
        self.feature_categories = {
            'intensity': 10,
            'geometric': 6,
            'texture': 7,
            'morphological': 3
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
    def _validate_volume(self, volume: np.ndarray) -> bool:
        """Validate that volume data is appropriate for feature extraction"""
        if volume.ndim != 3:
            logger.warning(f"Volume has {volume.ndim} dimensions, expected 3")
            return False
        
        if np.all(volume == 0):
            logger.warning("Volume contains only zeros")
            return False
            
        if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
            logger.warning("Volume contains NaN or infinite values")
            return False
            
        return True
    
    def extract_intensity_features(self, volume: np.ndarray) -> List[float]:
        """
        Extract intensity-based statistical features
        
        Args:
            volume: 3D CT volume array
            
        Returns:
            List of 10 intensity features
        """
        flat_volume = volume.flatten()
        
        # Remove outliers for more stable statistics
        p1, p99 = np.percentile(flat_volume, [1, 99])
        filtered_volume = flat_volume[(flat_volume >= p1) & (flat_volume <= p99)]
        
        if len(filtered_volume) == 0:
            filtered_volume = flat_volume
        
        features = [
            float(np.mean(filtered_volume)),           # Mean intensity
            float(np.std(filtered_volume)),            # Standard deviation
            float(np.var(filtered_volume)),            # Variance
            float(np.min(filtered_volume)),            # Minimum intensity
            float(np.max(filtered_volume)),            # Maximum intensity
            float(np.median(filtered_volume)),         # Median intensity
            float(np.percentile(filtered_volume, 25)), # 25th percentile
            float(np.percentile(filtered_volume, 75)), # 75th percentile
            float(np.percentile(filtered_volume, 10)), # 10th percentile
            float(np.percentile(filtered_volume, 90)), # 90th percentile
        ]
        
        return features
    
    def extract_geometric_features(self, volume: np.ndarray) -> List[float]:
        """
        Extract geometric and shape-based features
        
        Args:
            volume: 3D CT volume array
            
        Returns:
            List of 6 geometric features
        """
        try:
            # Threshold volume for shape analysis (adaptive threshold)
            threshold = np.percentile(volume, 70)
            binary_volume = volume > threshold
            
            if not np.any(binary_volume):
                # Fallback to lower threshold
                threshold = np.percentile(volume, 50)
                binary_volume = volume > threshold
                
            if not np.any(binary_volume):
                return [0.0] * 6
            
            # Find connected components
            labeled_volume = measure.label(binary_volume)
            regions = measure.regionprops(labeled_volume)
            
            if not regions:
                return [0.0] * 6
            
            # Use largest region (assumed to be the nodule)
            largest_region = max(regions, key=lambda r: r.area)
            
            # Calculate bounding box dimensions
            bbox = largest_region.bbox
            height = bbox[3] - bbox[0] if len(bbox) >= 4 else 1
            width = bbox[4] - bbox[1] if len(bbox) >= 5 else 1
            depth = bbox[5] - bbox[2] if len(bbox) >= 6 else 1
            
            # Extract geometric properties with error handling
            features = [
                float(largest_region.area),                    # Volume (number of voxels)
                float(height),                                 # Bounding box height
                float(width),                                  # Bounding box width  
                float(depth),                                  # Bounding box depth
                float(getattr(largest_region, 'extent', 0.5)), # Ratio of area to bounding box
                float(getattr(largest_region, 'solidity', 0.7)) # Ratio of area to convex hull
            ]
            
        except Exception as e:
            logger.warning(f"Error in geometric feature extraction: {e}")
            features = [100.0, 10.0, 10.0, 8.0, 0.5, 0.7]  # Default reasonable values
        
        return features
    
    def extract_texture_features(self, volume: np.ndarray) -> List[float]:
        """
        Extract texture-based features
        
        Args:
            volume: 3D CT volume array
            
        Returns:
            List of 7 texture features
        """
        flat_volume = volume.flatten()
        
        # Histogram-based features
        try:
            hist, _ = np.histogram(flat_volume, bins=32, density=True)
            hist = hist + 1e-10  # Avoid log(0)
            
            # Entropy (measure of randomness)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Energy (uniformity measure)
            energy = np.sum(hist**2)
            
        except Exception as e:
            logger.warning(f"Error in histogram features: {e}")
            entropy = 5.0
            energy = 0.05
        
        # Gradient-based features
        try:
            if volume.ndim == 3:
                grad_x = np.gradient(volume, axis=0)
                grad_y = np.gradient(volume, axis=1) 
                grad_z = np.gradient(volume, axis=2)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                
                grad_mean = float(np.mean(gradient_magnitude))
                grad_std = float(np.std(gradient_magnitude))
                grad_max = float(np.max(gradient_magnitude))
            else:
                grad_mean = grad_std = grad_max = 0.0
                
        except Exception as e:
            logger.warning(f"Error in gradient features: {e}")
            grad_mean = grad_std = grad_max = 10.0
        
        # Statistical moments
        try:
            mean_val = np.mean(flat_volume)
            std_val = np.std(flat_volume)
            
            if std_val > 0:
                normalized = (flat_volume - mean_val) / std_val
                skewness = float(np.mean(normalized**3))
                kurtosis = float(np.mean(normalized**4))
            else:
                skewness = kurtosis = 0.0
                
        except Exception as e:
            logger.warning(f"Error in moment features: {e}")
            skewness = kurtosis = 0.0
        
        features = [
            float(entropy),
            float(energy),
            grad_mean,
            grad_std,
            grad_max,
            skewness,
            kurtosis
        ]
        
        return features
    
    def extract_morphological_features(self, volume: np.ndarray) -> List[float]:
        """
        Extract morphological features
        
        Args:
            volume: 3D CT volume array
            
        Returns:
            List of 3 morphological features
        """
        try:
            # Threshold volume
            threshold = np.percentile(volume, 70)
            binary_volume = volume > threshold
            
            if not np.any(binary_volume):
                return [0.8, 1.2, 0.5]  # Default values
            
            # Morphological operations
            try:
                eroded = ndimage.binary_erosion(binary_volume)
                dilated = ndimage.binary_dilation(binary_volume)
                
                original_volume = np.sum(binary_volume)
                eroded_volume = np.sum(eroded)
                dilated_volume = np.sum(dilated)
                total_volume = binary_volume.size
                
                # Calculate ratios with safety checks
                erosion_ratio = eroded_volume / (original_volume + 1e-10)
                dilation_ratio = dilated_volume / (original_volume + 1e-10)
                fill_ratio = original_volume / total_volume
                
            except Exception as e:
                logger.warning(f"Error in morphological operations: {e}")
                erosion_ratio = 0.8
                dilation_ratio = 1.2
                fill_ratio = 0.5
            
        except Exception as e:
            logger.warning(f"Error in morphological feature extraction: {e}")
            erosion_ratio = 0.8
            dilation_ratio = 1.2
            fill_ratio = 0.5
        
        features = [
            float(erosion_ratio),   # Erosion ratio
            float(dilation_ratio),  # Dilation ratio
            float(fill_ratio),      # Fill ratio
        ]
        
        return features
    
    def extract_comprehensive_features(self, volume: np.ndarray) -> List[float]:
        """
        Extract all radiological features from a single volume
        
        Args:
            volume: 3D CT volume array
            
        Returns:
            List of 26 comprehensive features
        """
        if not self._validate_volume(volume):
            logger.warning("Invalid volume, using default features")
            return [0.0] * 26
        
        features = []
        
        try:
            # Extract features by category
            intensity_features = self.extract_intensity_features(volume)
            geometric_features = self.extract_geometric_features(volume)
            texture_features = self.extract_texture_features(volume)
            morphological_features = self.extract_morphological_features(volume)
            
            # Combine all features
            features.extend(intensity_features)
            features.extend(geometric_features)
            features.extend(texture_features)
            features.extend(morphological_features)
            
            # Set feature names if not already set
            if not self.feature_names:
                self._set_feature_names()
                
        except Exception as e:
            logger.error(f"Error in comprehensive feature extraction: {e}")
            features = [0.0] * 26  # Return default features
        
        # Validate feature count
        if len(features) != 26:
            logger.warning(f"Expected 26 features, got {len(features)}")
            features = features[:26] + [0.0] * max(0, 26 - len(features))
        
        return features
    
    def _set_feature_names(self):
        """Set descriptive names for all features"""
        self.feature_names = [
            # Intensity features (10)
            'intensity_mean', 'intensity_std', 'intensity_var', 'intensity_min', 'intensity_max',
            'intensity_median', 'intensity_p25', 'intensity_p75', 'intensity_p10', 'intensity_p90',
            
            # Geometric features (6)
            'geometric_area', 'geometric_height', 'geometric_width', 'geometric_depth',
            'geometric_extent', 'geometric_solidity',
            
            # Texture features (7)
            'texture_entropy', 'texture_energy', 'texture_grad_mean', 'texture_grad_std',
            'texture_grad_max', 'texture_skewness', 'texture_kurtosis',
            
            # Morphological features (3)
            'morphological_erosion', 'morphological_dilation', 'morphological_fill'
        ]
    
    def process_all_features(self) -> pd.DataFrame:
        """
        Process features for all volumes in the dataset
        
        Returns:
            DataFrame with extracted features and labels
        """
        logger.info("🔧 Extracting features from all volumes...")
        
        # Check required files
        volumes_dir = self.processed_dir / "volumes"
        annotations_path = self.raw_dir / "annotations.csv"
        
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        if not volumes_dir.exists():
            raise FileNotFoundError(f"Volumes directory not found: {volumes_dir}")
        
        # Load annotations
        annotations_df = pd.read_csv(annotations_path)
        logger.info(f"📄 Loaded {len(annotations_df)} annotations")
        
        # Process each volume
        all_features = []
        all_labels = []
        patient_ids = []
        processing_errors = 0
        
        volume_files = list(volumes_dir.glob("*_volume.npy"))
        logger.info(f"🔍 Found {len(volume_files)} volume files")
        
        for i, volume_file in enumerate(volume_files):
            if i % 20 == 0:
                logger.info(f"Processing volume {i+1}/{len(volume_files)}...")
            
            try:
                # Extract patient ID from filename
                patient_id = volume_file.stem.replace('_volume', '')
                
                # Load volume
                volume = np.load(volume_file)
                
                # Extract features
                features = self.extract_comprehensive_features(volume)
                
                # Get label from annotations
                matching_rows = annotations_df[annotations_df['seriesuid'] == patient_id]
                
                if len(matching_rows) > 0:
                    malignancy = matching_rows.iloc[0]['malignancy']
                    # Convert to binary classification (≥4 = malignant, ≤3 = benign)
                    label = 1 if malignancy >= 4 else 0
                    
                    all_features.append(features)
                    all_labels.append(label)
                    patient_ids.append(patient_id)
                else:
                    logger.warning(f"No annotation found for patient {patient_id}")
                    
            except Exception as e:
                logger.error(f"Error processing {volume_file}: {e}")
                processing_errors += 1
                continue
        
        if processing_errors > 0:
            logger.warning(f"⚠️  {processing_errors} volumes failed to process")
        
        if not all_features:
            raise ValueError("No features were successfully extracted")
        
        # Convert to arrays
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        logger.info(f"📊 Feature matrix shape: {features_array.shape}")
        logger.info(f"📊 Label distribution: {np.bincount(labels_array)}")
        
        # Fit scaler and transform features
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Create feature DataFrame
        if not self.feature_names:
            self._set_feature_names()
            
        features_df = pd.DataFrame(features_scaled, columns=self.feature_names)
        features_df['patient_id'] = patient_ids
        features_df['label'] = labels_array
        
        # Save processed features
        output_path = self.processed_dir / "features.csv"
        features_df.to_csv(output_path, index=False)
        
        # Save scaler
        scaler_path = self.models_dir / "feature_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature extraction metadata
        metadata = {
            'n_features': len(self.feature_names),
            'n_samples': len(features_df),
            'feature_names': self.feature_names,
            'feature_categories': self.feature_categories,
            'label_distribution': {
                'benign': int(sum(labels_array == 0)),
                'malignant': int(sum(labels_array == 1))
            },
            'scaler_path': str(scaler_path),
            'processing_errors': processing_errors
        }
        
        metadata_path = self.processed_dir / "feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print summary
        logger.info("✅ Feature extraction completed!")
        logger.info(f"📊 Extracted {len(self.feature_names)} features for {len(patient_ids)} patients")
        logger.info(f"💾 Features saved to: {output_path}")
        logger.info(f"🔧 Scaler saved to: {scaler_path}")
        logger.info(f"📄 Metadata saved to: {metadata_path}")
        
        return features_df
    
    def extract_features_single(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract features from a single volume for prediction
        
        Args:
            volume: 3D CT volume array
            
        Returns:
            Scaled feature array
        """
        features = self.extract_comprehensive_features(volume)
        
        # Load scaler if available
        scaler_path = self.models_dir / "feature_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            features_scaled = scaler.transform([features])
            return features_scaled[0]
        else:
            logger.warning("No scaler found, returning unscaled features")
            return np.array(features)

def main():
    """Command line interface for feature extraction"""
    parser = argparse.ArgumentParser(description="Extract features from lung nodule CT volumes")
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--output-dir', help='Output directory (defaults to data/processed)')
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(data_dir=args.data_dir)
    
    if args.output_dir:
        extractor.processed_dir = Path(args.output_dir)
        extractor.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features
    try:
        features_df = extractor.process_all_features()
        
        print("\n" + "="*60)
        print("✅ FEATURE EXTRACTION COMPLETED!")
        print("="*60)
        print(f"📊 Features extracted: {len(extractor.feature_names)}")
        print(f"👥 Patients processed: {len(features_df)}")
        print(f"🎯 Malignant cases: {sum(features_df['label'] == 1)}")
        print(f"💚 Benign cases: {sum(features_df['label'] == 0)}")
        print(f"💾 Output file: {extractor.processed_dir}/features.csv")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ FEATURE EXTRACTION FAILED: {e}")
        raise

if __name__ == "__main__":
    main()
