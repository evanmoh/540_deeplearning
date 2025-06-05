"""
Dataset Creation and Loading Module
Handles LIDC-IDRI data processing and mock data generation
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
import argparse
from typing import Tuple, Dict, List
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCreator:
    """
    Handles dataset creation for lung nodule detection project
    Supports both real LIDC-IDRI data and mock data generation
    """
    
    def __init__(self, use_mock: bool = True, data_dir: str = "data"):
        """
        Initialize dataset creator
        
        Args:
            use_mock: Whether to use mock data or real LIDC-IDRI data
            data_dir: Directory to store data files
        """
        self.use_mock = use_mock
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset parameters
        self.n_patients = 100 if use_mock else 1018
        self.volume_shape = (64, 64, 32)  # Reduced for demo
        
    def generate_mock_annotations(self) -> pd.DataFrame:
        """Generate realistic mock annotations similar to LIDC-IDRI format"""
        logger.info(f"🎲 Generating mock annotations for {self.n_patients} patients...")
        
        np.random.seed(42)  # For reproducible results
        
        annotations = []
        for i in range(self.n_patients):
            # Generate realistic patient data
            patient_data = {
                'seriesuid': f'LIDC-IDRI-{i+1:04d}',
                'patient_id': f'LIDC-IDRI-{i+1:04d}',
                'coordX': np.random.uniform(100, 400),  # Pixel coordinates
                'coordY': np.random.uniform(100, 400),
                'coordZ': np.random.uniform(10, 50),
                'diameter_mm': np.random.uniform(3, 30),  # 3-30mm nodules
                'malignancy': np.random.randint(1, 6),  # 1-5 scale
                'spiculation': np.random.randint(1, 6),
                'margin': np.random.randint(1, 6), 
                'lobulation': np.random.randint(1, 6),
                'sphericity': np.random.randint(1, 6),
                'calcification': np.random.randint(1, 7),  # 1-6 scale
                'internal_structure': np.random.randint(1, 5),
                'subtlety': np.random.randint(1, 6),
                'texture': np.random.randint(1, 6)
            }
            annotations.append(patient_data)
        
        df = pd.DataFrame(annotations)
        
        # Add realistic correlations
        # Larger nodules slightly more likely to be malignant
        large_nodule_mask = df['diameter_mm'] > 15
        df.loc[large_nodule_mask, 'malignancy'] = np.clip(
            df.loc[large_nodule_mask, 'malignancy'] + np.random.randint(-1, 2, sum(large_nodule_mask)),
            1, 5
        )
        
        # Spiculated nodules more likely malignant
        spiculated_mask = df['spiculation'] >= 4
        df.loc[spiculated_mask, 'malignancy'] = np.clip(
            df.loc[spiculated_mask, 'malignancy'] + 1,
            1, 5
        )
        
        logger.info(f"✅ Generated {len(df)} mock annotations")
        return df
    
    def generate_mock_volumes(self, annotations: pd.DataFrame) -> None:
        """Generate mock CT volumes for each patient"""
        logger.info(f"🔧 Generating mock CT volumes...")
        
        volumes_dir = self.processed_dir / "volumes"
        volumes_dir.mkdir(exist_ok=True)
        
        for idx, row in annotations.iterrows():
            if idx % 20 == 0:
                logger.info(f"Generated {idx}/{len(annotations)} volumes...")
            
            # Generate realistic CT volume
            volume = self._create_realistic_ct_volume(
                malignancy=row['malignancy'],
                diameter=row['diameter_mm'],
                spiculation=row['spiculation']
            )
            
            # Save volume
            volume_path = volumes_dir / f"{row['seriesuid']}_volume.npy"
            np.save(volume_path, volume)
        
        logger.info(f"✅ Generated {len(annotations)} CT volumes")
    
    def _create_realistic_ct_volume(self, malignancy: int, diameter: float, spiculation: int) -> np.ndarray:
        """Create a realistic CT volume with nodule characteristics"""
        # Base lung tissue (typical HU values: -800 to -600)
        volume = np.random.normal(-700, 100, self.volume_shape)
        
        # Add nodule in center region
        center = (self.volume_shape[0]//2, self.volume_shape[1]//2, self.volume_shape[2]//2)
        radius = int(diameter / 2)  # Simplified pixel to mm conversion
        
        # Create nodule region
        y, x, z = np.ogrid[:self.volume_shape[0], :self.volume_shape[1], :self.volume_shape[2]]
        mask = ((y - center[0])**2 + (x - center[1])**2 + (z - center[2])**2) <= radius**2
        
        # Nodule characteristics based on malignancy
        if malignancy >= 4:  # Likely malignant
            # Higher attenuation, more heterogeneous
            nodule_intensity = np.random.normal(-200, 150)
            heterogeneity = 0.3
        elif malignancy <= 2:  # Likely benign
            # Lower attenuation, more homogeneous  
            nodule_intensity = np.random.normal(-400, 100)
            heterogeneity = 0.1
        else:  # Uncertain
            nodule_intensity = np.random.normal(-300, 125)
            heterogeneity = 0.2
        
        # Apply nodule characteristics
        nodule_values = np.random.normal(nodule_intensity, heterogeneity * abs(nodule_intensity), np.sum(mask))
        volume[mask] = nodule_values
        
        # Add spiculation effects (irregular edges)
        if spiculation >= 4:
            # Add some irregular extensions
            extended_mask = ((y - center[0])**2 + (x - center[1])**2 + (z - center[2])**2) <= (radius * 1.2)**2
            spicule_mask = extended_mask & ~mask
            spicule_prob = np.random.random(np.sum(spicule_mask)) < 0.3
            volume[spicule_mask] = np.where(
                spicule_prob, 
                np.random.normal(nodule_intensity * 0.7, 50, np.sum(spicule_mask)),
                volume[spicule_mask]
            )
        
        # Normalize to typical CT range
        volume = np.clip(volume, -1000, 500)
        
        return volume.astype(np.float32)
    
    def load_real_lidc_data(self, lidc_path: str) -> pd.DataFrame:
        """Load real LIDC-IDRI annotations (placeholder for real implementation)"""
        logger.info("📥 Loading real LIDC-IDRI data...")
        
        # This would contain actual LIDC-IDRI data loading logic
        # For now, we'll use mock data with a warning
        logger.warning("⚠️  Real LIDC-IDRI data loading not implemented")
        logger.info("🎲 Falling back to mock data generation")
        
        return self.generate_mock_annotations()
    
    def create_dataset_metadata(self, annotations: pd.DataFrame) -> Dict:
        """Create metadata about the dataset"""
        metadata = {
            'dataset_type': 'mock' if self.use_mock else 'real_lidc',
            'n_patients': len(annotations),
            'n_malignant': sum(annotations['malignancy'] >= 4),
            'n_benign': sum(annotations['malignancy'] <= 2),
            'n_uncertain': sum(annotations['malignancy'] == 3),
            'malignancy_distribution': annotations['malignancy'].value_counts().to_dict(),
            'diameter_stats': {
                'mean': float(annotations['diameter_mm'].mean()),
                'std': float(annotations['diameter_mm'].std()),
                'min': float(annotations['diameter_mm'].min()),
                'max': float(annotations['diameter_mm'].max())
            },
            'volume_shape': self.volume_shape,
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return metadata
    
    def save_annotations(self, annotations: pd.DataFrame) -> None:
        """Save annotations to CSV file"""
        annotations_path = self.raw_dir / "annotations.csv"
        annotations.to_csv(annotations_path, index=False)
        logger.info(f"💾 Saved annotations to {annotations_path}")
    
    def save_metadata(self, metadata: Dict) -> None:
        """Save dataset metadata to JSON file"""
        metadata_path = self.raw_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata to {metadata_path}")
    
    def create_complete_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """Create complete dataset with annotations and volumes"""
        logger.info("🚀 Creating complete dataset...")
        
        # Generate or load annotations
        if self.use_mock:
            annotations = self.generate_mock_annotations()
        else:
            # This would load real LIDC-IDRI data
            annotations = self.load_real_lidc_data("")
        
        # Create metadata
        metadata = self.create_dataset_metadata(annotations)
        
        # Save annotations and metadata
        self.save_annotations(annotations)
        self.save_metadata(metadata)
        
        # Generate volumes (for mock data)
        if self.use_mock:
            self.generate_mock_volumes(annotations)
        
        # Print summary
        logger.info("📊 Dataset Summary:")
        logger.info(f"   Total patients: {len(annotations)}")
        logger.info(f"   Malignant (score 4-5): {metadata['n_malignant']}")
        logger.info(f"   Benign (score 1-2): {metadata['n_benign']}")
        logger.info(f"   Uncertain (score 3): {metadata['n_uncertain']}")
        logger.info(f"   Average diameter: {metadata['diameter_stats']['mean']:.1f}mm")
        
        logger.info("✅ Dataset creation completed successfully!")
        return annotations, metadata

def main():
    """Command line interface for dataset creation"""
    parser = argparse.ArgumentParser(description="Create lung nodule dataset")
    parser.add_argument('--mock', action='store_true', default=True,
                       help='Use mock data (default)')
    parser.add_argument('--real', action='store_true',
                       help='Use real LIDC-IDRI data')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory path')
    parser.add_argument('--n-patients', type=int, default=100,
                       help='Number of patients for mock data')
    
    args = parser.parse_args()
    
    # Create dataset
    creator = DatasetCreator(
        use_mock=not args.real,
        data_dir=args.data_dir
    )
    
    if args.n_patients != 100:
        creator.n_patients = args.n_patients
    
    annotations, metadata = creator.create_complete_dataset()
    
    print("\n" + "="*50)
    print("✅ DATASET CREATION COMPLETED!")
    print("="*50)
    print(f"📁 Data saved to: {creator.data_dir}")
    print(f"📄 Annotations: {creator.raw_dir}/annotations.csv")
    print(f"📊 Metadata: {creator.raw_dir}/metadata.json")
    if creator.use_mock:
        print(f"🔧 Volumes: {creator.processed_dir}/volumes/")
    print("="*50)

if __name__ == "__main__":
    main()
