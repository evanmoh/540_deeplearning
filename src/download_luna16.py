#!/usr/bin/env python3
"""
🏆 GOOGLE COLAB LUNA16 DOWNLOADER
Download LUNA16 CT scan subsets 0-6 directly in Colab to Google Drive
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import time

def download_file_with_progress(url, filepath):
    """Download file with progress bar"""
    print(f"📥 Downloading {filepath.name}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✅ Downloaded {filepath.name} ({total_size / (1024**3):.1f} GB)")

def extract_with_progress(zip_path, extract_path):
    """Extract zip file with progress"""
    print(f"📦 Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_path)
                pbar.update(1)
    
    print(f"✅ Extracted {zip_path.name}")

def download_luna16_subsets():
    """Download LUNA16 subsets 0-6 to Google Drive"""
    print("🏆 LUNA16 CT SCAN DOWNLOADER FOR GOOGLE COLAB")
    print("="*60)
    print("📁 Target: /content/drive/MyDrive/data/raw/")
    print("📊 Downloading subsets 0-6 (~42GB total)")
    print("="*60)
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except Exception as e:
        print(f"❌ Error mounting drive: {e}")
        return
    
    # Setup paths
    base_path = Path('/content/drive/MyDrive/data/raw')
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Download directory: {base_path}")
    
    # LUNA16 subset URLs
    base_url = "https://zenodo.org/records/3723295/files"
    subsets = range(0, 7)  # 0-6
    
    total_start_time = time.time()
    
    for subset_num in subsets:
        subset_start_time = time.time()
        
        print(f"\n🔄 Processing subset{subset_num}")
        print("-" * 40)
        
        # Paths
        zip_filename = f"subset{subset_num}.zip"
        zip_url = f"{base_url}/{zip_filename}?download=1"
        zip_path = base_path / zip_filename
        extract_path = base_path
        subset_folder = base_path / f"subset{subset_num}"
        
        # Check if already downloaded and extracted
        if subset_folder.exists() and any(subset_folder.iterdir()):
            print(f"✅ subset{subset_num} already exists, skipping...")
            continue
        
        try:
            # Download
            download_file_with_progress(zip_url, zip_path)
            
            # Extract
            extract_with_progress(zip_path, extract_path)
            
            # Cleanup zip file to save space
            zip_path.unlink()
            print(f"🗑️  Removed {zip_filename} to save space")
            
            # Verify extraction
            if subset_folder.exists():
                mhd_files = list(subset_folder.glob("*.mhd"))
                raw_files = list(subset_folder.glob("*.raw"))
                print(f"📊 subset{subset_num}: {len(mhd_files)} scans ({len(raw_files)} raw files)")
            
            subset_time = time.time() - subset_start_time
            print(f"⏱️  subset{subset_num} completed in {subset_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"❌ Error processing subset{subset_num}: {e}")
            # Cleanup on error
            if zip_path.exists():
                zip_path.unlink()
            continue
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n🏆 DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    # Check final structure
    downloaded_subsets = []
    total_scans = 0
    
    for subset_num in subsets:
        subset_folder = base_path / f"subset{subset_num}"
        if subset_folder.exists():
            mhd_files = list(subset_folder.glob("*.mhd"))
            downloaded_subsets.append(subset_num)
            total_scans += len(mhd_files)
            print(f"✅ subset{subset_num}: {len(mhd_files)} CT scans")
    
    print(f"\n📊 FINAL STATS:")
    print(f"   📁 Downloaded subsets: {downloaded_subsets}")
    print(f"   🔬 Total CT scans: {total_scans}")
    print(f"   💾 Location: {base_path}")
    
    # Check disk usage
    total_size = sum(f.stat().st_size for f in base_path.rglob('*') if f.is_file())
    print(f"   📏 Total size: {total_size / (1024**3):.1f} GB")
    
    if len(downloaded_subsets) >= 3:
        print(f"\n🎉 SUCCESS! You have enough CT scan data for competition-level training!")
        print(f"🎯 Ready to achieve FROC > 0.951 with real medical imaging!")
    elif len(downloaded_subsets) >= 1:
        print(f"\n✅ Good start! You can begin training with this data.")
        print(f"💡 More subsets will improve performance.")
    else:
        print(f"\n⚠️  No subsets downloaded successfully.")
    
    return downloaded_subsets

def verify_luna16_data():
    """Verify downloaded LUNA16 data structure"""
    print("🔍 VERIFYING LUNA16 DATA STRUCTURE")
    print("-" * 40)
    
    base_path = Path('/content/drive/MyDrive/data/raw')
    
    if not base_path.exists():
        print("❌ Data directory not found")
        return False
    
    subsets_found = []
    for subset_num in range(0, 7):
        subset_folder = base_path / f"subset{subset_num}"
        if subset_folder.exists():
            mhd_files = list(subset_folder.glob("*.mhd"))
            raw_files = list(subset_folder.glob("*.raw"))
            
            if len(mhd_files) > 0 and len(raw_files) > 0:
                subsets_found.append(subset_num)
                print(f"✅ subset{subset_num}: {len(mhd_files)} scans")
                
                # Check a sample file
                sample_mhd = mhd_files[0]
                sample_raw = sample_mhd.with_suffix('.raw')
                if sample_raw.exists():
                    mhd_size = sample_mhd.stat().st_size
                    raw_size = sample_raw.stat().st_size
                    print(f"   📄 Sample: {sample_mhd.name}")
                    print(f"   📏 Sizes: .mhd={mhd_size} bytes, .raw={raw_size/1024/1024:.1f}MB")
    
    if subsets_found:
        print(f"\n✅ Found valid subsets: {subsets_found}")
        print(f"🔬 Ready for 3D CNN training!")
        return True
    else:
        print(f"\n❌ No valid LUNA16 subsets found")
        return False

def show_download_progress():
    """Show current download status"""
    base_path = Path('/content/drive/MyDrive/data/raw')
    
    print("📊 CURRENT DOWNLOAD STATUS")
    print("-" * 30)
    
    if not base_path.exists():
        print("❌ Download directory not found")
        return
    
    for subset_num in range(0, 7):
        subset_folder = base_path / f"subset{subset_num}"
        zip_file = base_path / f"subset{subset_num}.zip"
        
        if subset_folder.exists():
            mhd_files = list(subset_folder.glob("*.mhd"))
            if len(mhd_files) > 0:
                print(f"✅ subset{subset_num}: COMPLETE ({len(mhd_files)} scans)")
            else:
                print(f"📦 subset{subset_num}: EXTRACTED (checking...)")
        elif zip_file.exists():
            size_mb = zip_file.stat().st_size / (1024**2)
            print(f"📥 subset{subset_num}: DOWNLOADING ({size_mb:.1f}MB so far)")
        else:
            print(f"⏳ subset{subset_num}: PENDING")

if __name__ == "__main__":
    print("🏆 LUNA16 CT Scan Downloader for Google Colab")
    print("📁 Downloads to: /content/drive/MyDrive/data/raw/")
    print("📊 Subsets: 0-6 (~42GB total)")
    print("\n" + "="*60)
    
    # Run the download
    downloaded_subsets = download_luna16_subsets()
    
    if downloaded_subsets:
        print(f"\n🎉 Download completed!")
        print(f"🚀 Ready to run competition-grade 3D CNN!")
        
        # Verify data
        verify_luna16_data()
    else:
        print(f"\n⚠️  Download incomplete")
        print(f"💡 Check internet connection and try again")

#!/usr/bin/env python3
"""
🏆 LUNA16 SUBSETS 7-9 DOWNLOADER
Download remaining LUNA16 CT scan subsets 7-9 from Zenodo record 4121926
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import time

def download_file_with_progress(url, filepath):
    """Download file with progress bar and retry capability"""
    print(f"📥 Downloading {filepath.name}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Downloaded {filepath.name} ({total_size / (1024**3):.1f} GB)")
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"❌ Failed to download {filepath.name} after {max_retries} attempts")
                return False

def extract_with_progress(zip_path, extract_path):
    """Extract zip file with progress"""
    print(f"📦 Extracting {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    zip_ref.extract(member, extract_path)
                    pbar.update(1)
        
        print(f"✅ Extracted {zip_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting {zip_path.name}: {e}")
        return False

def download_luna16_subsets_7_9():
    """Download LUNA16 subsets 7-9 to Google Drive"""
    print("🏆 LUNA16 SUBSETS 7-9 DOWNLOADER")
    print("="*60)
    print("📁 Target: /content/drive/MyDrive/data/raw/")
    print("📊 Downloading subsets 7-9 from Zenodo record 4121926")
    print("="*60)
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except Exception as e:
        print(f"❌ Error mounting drive: {e}")
        return []
    
    # Setup paths
    base_path = Path('/content/drive/MyDrive/data/raw')
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Download directory: {base_path}")
    
    # LUNA16 subset 7-9 URLs (from second Zenodo record)
    base_url = "https://zenodo.org/records/4121926/files"
    subsets = [7, 8, 9]  # Only 7-9
    
    total_start_time = time.time()
    downloaded_subsets = []
    
    for subset_num in subsets:
        subset_start_time = time.time()
        
        print(f"\n🔄 Processing subset{subset_num}")
        print("-" * 40)
        
        # Paths
        zip_filename = f"subset{subset_num}.zip"
        zip_url = f"{base_url}/{zip_filename}?download=1"
        zip_path = base_path / zip_filename
        extract_path = base_path
        subset_folder = base_path / f"subset{subset_num}"
        
        # Check if already downloaded and extracted
        if subset_folder.exists() and any(subset_folder.iterdir()):
            mhd_files = list(subset_folder.glob("*.mhd"))
            print(f"✅ subset{subset_num} already exists ({len(mhd_files)} scans), skipping...")
            downloaded_subsets.append(subset_num)
            continue
        
        # Download
        if download_file_with_progress(zip_url, zip_path):
            # Extract
            if extract_with_progress(zip_path, extract_path):
                # Cleanup zip file to save space
                zip_path.unlink()
                print(f"🗑️  Removed {zip_filename} to save space")
                
                # Verify extraction
                if subset_folder.exists():
                    mhd_files = list(subset_folder.glob("*.mhd"))
                    raw_files = list(subset_folder.glob("*.raw"))
                    print(f"📊 subset{subset_num}: {len(mhd_files)} scans ({len(raw_files)} raw files)")
                    downloaded_subsets.append(subset_num)
                
                subset_time = time.time() - subset_start_time
                print(f"⏱️  subset{subset_num} completed in {subset_time/60:.1f} minutes")
            else:
                # Cleanup on extraction error
                if zip_path.exists():
                    zip_path.unlink()
        else:
            # Cleanup on download error
            if zip_path.exists():
                zip_path.unlink()
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print(f"\n🏆 DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    # Check final structure
    total_scans_new = 0
    for subset_num in subsets:
        subset_folder = base_path / f"subset{subset_num}"
        if subset_folder.exists():
            mhd_files = list(subset_folder.glob("*.mhd"))
            total_scans_new += len(mhd_files)
            print(f"✅ subset{subset_num}: {len(mhd_files)} CT scans")
        else:
            print(f"❌ subset{subset_num}: NOT FOUND")
    
    # Check ALL subsets (0-9)
    print(f"\n📊 COMPLETE LUNA16 DATASET STATUS:")
    all_subsets = []
    total_all_scans = 0
    
    for subset_num in range(0, 10):
        subset_folder = base_path / f"subset{subset_num}"
        if subset_folder.exists():
            mhd_files = list(subset_folder.glob("*.mhd"))
            if len(mhd_files) > 0:
                all_subsets.append(subset_num)
                total_all_scans += len(mhd_files)
                status = "✅"
            else:
                status = "📦"
        else:
            status = "❌"
        
        print(f"   {status} subset{subset_num}: {len(mhd_files) if subset_folder.exists() else 0} scans")
    
    print(f"\n📊 FINAL STATS:")
    print(f"   📁 Available subsets: {all_subsets}")
    print(f"   📁 Just downloaded: {downloaded_subsets}")
    print(f"   🔬 Total CT scans: {total_all_scans}")
    print(f"   💾 Location: {base_path}")
    
    # Check disk usage
    total_size = sum(f.stat().st_size for f in base_path.rglob('*') if f.is_file())
    print(f"   📏 Total size: {total_size / (1024**3):.1f} GB")
    
    if len(all_subsets) == 10:
        print(f"\n🎉 COMPLETE SUCCESS! You now have ALL LUNA16 subsets (0-9)!")
        print(f"🏆 Ready for competition-level training with full dataset!")
        print(f"🎯 Target FROC > 0.951 is achievable!")
    elif len(all_subsets) >= 7:
        print(f"\n🎉 Excellent! You have {len(all_subsets)}/10 subsets!")
        print(f"🚀 Ready for high-performance 3D CNN training!")
    elif len(downloaded_subsets) > 0:
        print(f"\n✅ Successfully downloaded {len(downloaded_subsets)} new subsets!")
        print(f"💡 Total available: {len(all_subsets)}/10 subsets")
    else:
        print(f"\n⚠️  No new subsets downloaded successfully.")
    
    return downloaded_subsets, all_subsets

def verify_complete_luna16():
    """Verify complete LUNA16 dataset (subsets 0-9)"""
    print("🔍 VERIFYING COMPLETE LUNA16 DATASET")
    print("-" * 50)
    
    base_path = Path('/content/drive/MyDrive/data/raw')
    
    if not base_path.exists():
        print("❌ Data directory not found")
        return False
    
    expected_counts = {
        0: 143, 1: 97, 2: 89, 3: 129, 4: 117,
        5: 133, 6: 137, 7: 138, 8: 137, 9: 133
    }  # Approximate expected counts per subset
    
    all_valid = True
    total_scans = 0
    
    for subset_num in range(0, 10):
        subset_folder = base_path / f"subset{subset_num}"
        if subset_folder.exists():
            mhd_files = list(subset_folder.glob("*.mhd"))
            raw_files = list(subset_folder.glob("*.raw"))
            
            if len(mhd_files) > 0 and len(raw_files) > 0:
                expected = expected_counts.get(subset_num, 100)
                status = "✅" if len(mhd_files) >= expected * 0.9 else "⚠️"
                print(f"{status} subset{subset_num}: {len(mhd_files)} scans (expected ~{expected})")
                total_scans += len(mhd_files)
                
                # Check sample file integrity
                sample_mhd = mhd_files[0]
                sample_raw = sample_mhd.with_suffix('.raw')
                if sample_raw.exists():
                    raw_size_mb = sample_raw.stat().st_size / (1024**2)
                    print(f"   📄 Sample: {sample_mhd.name} ({raw_size_mb:.1f}MB)")
            else:
                print(f"❌ subset{subset_num}: INCOMPLETE or EMPTY")
                all_valid = False
        else:
            print(f"❌ subset{subset_num}: NOT FOUND")
            all_valid = False
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   🔬 Total scans: {total_scans} (expected ~1,353)")
    print(f"   ✅ Complete: {all_valid}")
    
    if total_scans >= 1200:
        print(f"\n🏆 EXCELLENT! Dataset is ready for competition-level training!")
    elif total_scans >= 800:
        print(f"\n✅ Good dataset size for serious training!")
    else:
        print(f"\n⚠️  Consider downloading more subsets for better performance")
    
    return all_valid

def show_storage_usage():
    """Show detailed storage usage breakdown"""
    base_path = Path('/content/drive/MyDrive/data/raw')
    
    print("💾 STORAGE USAGE BREAKDOWN")
    print("-" * 40)
    
    if not base_path.exists():
        print("❌ Data directory not found")
        return
    
    total_size = 0
    for subset_num in range(0, 10):
        subset_folder = base_path / f"subset{subset_num}"
        if subset_folder.exists():
            subset_size = sum(f.stat().st_size for f in subset_folder.rglob('*') if f.is_file())
            total_size += subset_size
            mhd_count = len(list(subset_folder.glob("*.mhd")))
            print(f"📁 subset{subset_num}: {subset_size / (1024**3):.2f} GB ({mhd_count} scans)")
        else:
            print(f"📁 subset{subset_num}: NOT DOWNLOADED")
    
    print(f"\n💾 Total LUNA16 size: {total_size / (1024**3):.1f} GB")
    
    # Check remaining Google Drive space
    try:
        import shutil
        drive_usage = shutil.disk_usage('/content/drive/MyDrive')
        free_gb = drive_usage.free / (1024**3)
        total_gb = drive_usage.total / (1024**3)
        used_gb = total_gb - free_gb
        
        print(f"💿 Google Drive usage: {used_gb:.1f}GB / {total_gb:.1f}GB")
        print(f"🆓 Free space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print(f"⚠️  WARNING: Low disk space!")
        elif free_gb < 15:
            print(f"💡 Consider cleaning up old files")
        else:
            print(f"✅ Sufficient space available")
            
    except Exception as e:
        print(f"ℹ️  Could not check drive space: {e}")

if __name__ == "__main__":
    print("🏆 LUNA16 Subsets 7-9 Downloader")
    print("📁 Downloads to: /content/drive/MyDrive/data/raw/")
    print("📊 Source: https://zenodo.org/records/4121926")
    print("\n" + "="*60)
    
    # Show current storage
    show_storage_usage()
    
    print(f"\n🚀 Starting download of subsets 7-9...")
    
    # Run the download
    downloaded_subsets, all_subsets = download_luna16_subsets_7_9()
    
    if downloaded_subsets:
        print(f"\n🎉 Download completed!")
        print(f"📦 Downloaded: {downloaded_subsets}")
        
        # Verify complete dataset
        verify_complete_luna16()
        
        # Final storage check
        print(f"\n" + "="*50)
        show_storage_usage()
        
        if len(all_subsets) == 10:
            print(f"\n🏆 MISSION ACCOMPLISHED!")
            print(f"🎯 Full LUNA16 dataset ready for training!")
    else:
        print(f"\n⚠️  Download incomplete")
        print(f"💡 Check internet connection and try again")
        
        # Show what's available
        verify_complete_luna16()
