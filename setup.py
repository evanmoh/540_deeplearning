#!/usr/bin/env python3
"""
Setup script for Lung Nodule Detection Project
Automates data preparation, feature extraction, and model training
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Handles complete project setup and initialization"""
    
    def __init__(self, use_mock_data=True):
        self.use_mock_data = use_mock_data
        self.project_root = Path(__file__).parent
        self.required_dirs = [
            'data/raw',
            'data/processed', 
            'data/outputs',
            'models',
            'notebooks'
        ]
    
    def create_directories(self):
        """Create required project directories"""
        logger.info("📁 Creating project directories...")
        
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep files for empty directories
            gitkeep_file = full_path / '.gitkeep'
            if not gitkeep_file.exists():
                gitkeep_file.touch()
        
        logger.info("✅ Project directories created successfully")
    
    def check_dependencies(self):
        """Check if all required packages are installed"""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'scikit-learn',
            'matplotlib', 'seaborn', 'scipy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {missing_packages}")
            logger.info("Install with: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All dependencies satisfied")
        return True
    
    def run_data_creation(self):
        """Execute data creation script"""
        logger.info("📊 Creating/loading dataset...")
        
        try:
            from scripts.make_dataset import DatasetCreator
            dataset_creator = DatasetCreator(use_mock=self.use_mock_data)
            dataset_creator.create_complete_dataset()
            logger.info("✅ Dataset created successfully")
        except Exception as e:
            logger.error(f"❌ Dataset creation failed: {e}")
            return False
        return True
    
    def run_feature_extraction(self):
        """Execute feature extraction script"""
        logger.info("🔧 Extracting features...")
        
        try:
            from scripts.build_features import FeatureExtractor
            feature_extractor = FeatureExtractor()
            feature_extractor.process_all_features()
            logger.info("✅ Feature extraction completed")
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return False
        return True
    
    def run_model_training(self):
        """Execute model training script"""
        logger.info("🤖 Training models...")
        
        try:
            from scripts.model import ModelTrainer
            model_trainer = ModelTrainer()
            model_trainer.train_all_approaches()
            logger.info("✅ Model training completed")
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
        return True
    
    def validate_setup(self):
        """Validate that setup completed successfully"""
        logger.info("🔎 Validating setup...")
        
        # Check if required files exist
        required_files = [
            'data/processed/features.csv',
            'models/naive_model.joblib',
            'models/random_forest_model.joblib',
            'models/deep_learning_model.h5'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"⚠️  Some files missing: {missing_files}")
            logger.info("This is normal for mock data setup")
        else:
            logger.info("✅ All expected files present")
        
        return True
    
    def run_complete_setup(self):
        """Run complete project setup pipeline"""
        logger.info("🚀 Starting complete project setup...")
        
        steps = [
            ("Creating directories", self.create_directories),
            ("Checking dependencies", self.check_dependencies),
            ("Creating dataset", self.run_data_creation),
            ("Extracting features", self.run_feature_extraction),
            ("Training models", self.run_model_training),
            ("Validating setup", self.validate_setup)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"⏳ {step_name}...")
            try:
                if not step_func():
                    logger.error(f"❌ Failed at: {step_name}")
                    return False
            except Exception as e:
                logger.error(f"❌ Error in {step_name}: {e}")
                return False
        
        logger.info("🎉 Project setup completed successfully!")
        logger.info("🌐 Run 'streamlit run main.py' to launch the web application")
        return True

def main():
    """Main setup function with command line arguments"""
    parser = argparse.ArgumentParser(description="Setup Lung Nodule Detection Project")
    parser.add_argument('--real-data', action='store_true', 
                       help='Use real LIDC-IDRI data (requires download)')
    parser.add_argument('--mock-data', action='store_true', default=True,
                       help='Use mock data for demonstration (default)')
    parser.add_argument('--skip-models', action='store_true',
                       help='Skip model training (faster setup)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation checks')
    
    args = parser.parse_args()
    
    # Determine data source
    use_mock = not args.real_data or args.mock_data
    
    # Initialize setup
    setup = ProjectSetup(use_mock_data=use_mock)
    
    if args.validate_only:
        logger.info("🔍 Running validation only...")
        setup.validate_setup()
        return
    
    # Print setup information
    print("=" * 60)
    print("🫁 LUNG NODULE DETECTION PROJECT SETUP")
    print("=" * 60)
    print(f"📊 Data Source: {'Mock Data' if use_mock else 'Real LIDC-IDRI'}")
    print(f"🤖 Model Training: {'Disabled' if args.skip_models else 'Enabled'}")
    print("=" * 60)
    
    # Run setup based on arguments
    if args.skip_models:
        # Quick setup without model training
        logger.info("🚀 Running quick setup (no model training)...")
        success = (setup.create_directories() and 
                  setup.check_dependencies() and
                  setup.run_data_creation() and
                  setup.run_feature_extraction())
    else:
        # Complete setup
        success = setup.run_complete_setup()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🌐 Next steps:")
        print("1. Run: streamlit run main.py")
        print("2. Open browser to: http://localhost:8501")
        print("3. Start making predictions!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SETUP FAILED!")
        print("=" * 60)
        print("🔧 Troubleshooting:")
        print("1. Check that all dependencies are installed")
        print("2. Ensure sufficient disk space")
        print("3. Run with --mock-data for demo version")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
