#!/usr/bin/env python3
"""
Setup script for Chemical Analysis Platform on Snapdragon X Elite
Handles ARM64-specific dependencies and optimizations
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path

def detect_architecture():
    """Detect system architecture and compatibility"""
    machine = platform.machine().lower()
    system = platform.system().lower()
    
    print(f"Detected system: {system}")
    print(f"Detected architecture: {machine}")
    
    is_arm64 = machine in ['arm64', 'aarch64', 'armv8']
    is_windows = system == 'windows'
    
    if is_windows and is_arm64:
        print("✓ Snapdragon X Elite / Windows ARM64 detected")
        return "windows_arm64"
    elif is_arm64:
        print("✓ ARM64 architecture detected")
        return "arm64"
    else:
        print("✓ x64/x86 architecture detected")
        return "x64"

def install_python_dependencies():
    """Install Python dependencies with ARM64 optimizations"""
    print("\n=== Installing Python Dependencies ===")
    
    # Upgrade pip first
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install dependencies in optimal order for ARM64
    dependencies = [
        # Core scientific computing (install first)
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        
        # Data processing
        "pandas>=1.5.0",
        
        # Visualization (matplotlib before seaborn)
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pillow>=8.0.0",
        
        # Machine learning
        "scikit-learn>=1.1.0",
        "joblib>=1.1.0",
        
        # Chemistry - RDKit (may require special handling on ARM64)
        "rdkit-pypi>=2022.9.1",
        
        # API and networking
        "requests>=2.28.0",
        "groq>=0.4.0",
        
        # Build tools
        "setuptools>=60.0.0",
        "wheel>=0.37.0",
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          check=True, capture_output=True, text=True)
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Failed to install {dep}")
            print(f"Error: {e.stderr}")
            
            # Special handling for RDKit on ARM64
            if "rdkit" in dep.lower():
                print("Attempting alternative RDKit installation...")
                try_alternative_rdkit()

def try_alternative_rdkit():
    """Alternative RDKit installation methods for ARM64"""
    print("Trying conda-forge RDKit...")
    try:
        # Try installing via conda if available
        subprocess.run(["conda", "install", "-c", "conda-forge", "rdkit", "-y"], 
                      check=True, capture_output=True)
        print("✓ RDKit installed via conda")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Conda not available or failed")
    
    # Try pre-compiled wheel
    print("Trying pre-compiled wheel...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "--find-links", "https://download.pytorch.org/whl/torch_stable.html",
                       "rdkit-pypi"], check=True)
        print("✓ RDKit installed via pre-compiled wheel")
        return True
    except subprocess.CalledProcessError:
        print("⚠ Pre-compiled wheel failed")
    
    return False

def setup_model_directory():
    """Setup directory structure for models"""
    print("\n=== Setting up Model Directory ===")
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (model_dir / "toxicity").mkdir(exist_ok=True)
    (model_dir / "bioactivity").mkdir(exist_ok=True)
    
    print(f"✓ Model directories created at {model_dir.absolute()}")
    
    # Create model info file
    model_info = model_dir / "README.md"
    with open(model_info, "w") as f:
        f.write("""# Model Directory

## Toxicity Models
Place your TOX21 endpoint models (.pkl files) in the `toxicity/` subdirectory:
- NR-AR.pkl
- NR-AR-LBD.pkl
- NR-AhR.pkl
- ... (other endpoint models)

## Bioactivity Models
Place your IC50 prediction models in the `bioactivity/` subdirectory:
- bioactivity_model.joblib
- scaler_X.joblib (optional)

## Model Sources
- TOX21 models can be downloaded from: [Your model repository URL]
- IC50 models can be loaded from GitHub URLs or local files
""")
    
    return model_dir

def create_launcher_script():
    """Create launcher script optimized for Snapdragon X Elite"""
    print("\n=== Creating Launcher Script ===")
    
    # Windows batch file for Snapdragon X Elite
    if platform.system().lower() == "windows":
        launcher_content = """@echo off
REM Chemical Analysis Platform Launcher for Snapdragon X Elite
REM Optimized for Windows ARM64

echo Starting Chemical Analysis Platform...
echo Optimized for Snapdragon X Elite

REM Set environment variables for ARM64 optimization
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
set NUMPY_NUM_THREADS=8
set OMP_NUM_THREADS=8

REM Set matplotlib backend
set MPLBACKEND=TkAgg

REM Launch the application
python chemical_analysis_tkinter.py

pause
"""
        
        with open("launch_chemical_analysis.bat", "w") as f:
            f.write(launcher_content)
        
        print("✓ Windows launcher created: launch_chemical_analysis.bat")
    
    # Unix shell script for other ARM64 systems
    launcher_content_unix = """#!/bin/bash
# Chemical Analysis Platform Launcher for ARM64
# Optimized for ARM64 processors including Snapdragon X Elite

echo "Starting Chemical Analysis Platform..."
echo "Optimized for ARM64 architecture"

# Set environment variables for ARM64 optimization
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMPY_NUM_THREADS=8
export OMP_NUM_THREADS=8

# Set matplotlib backend
export MPLBACKEND=TkAgg

# Launch the application
python3 chemical_analysis_tkinter.py
"""
    
    with open("launch_chemical_analysis.sh", "w") as f:
        f.write(launcher_content_unix)
    
    # Make executable on Unix systems
    if platform.system().lower() != "windows":
        os.chmod("launch_chemical_analysis.sh", 0o755)
        print("✓ Unix launcher created: launch_chemical_analysis.sh")

def create_config_file():
    """Create configuration file for the application"""
    print("\n=== Creating Configuration File ===")
    
    config_content = """# Chemical Analysis Platform Configuration
# Optimized for Snapdragon X Elite

[PERFORMANCE]
# Thread count optimized for Snapdragon X Elite (8-12 cores typical)
max_threads = 8
use_multiprocessing = true

# Memory optimization for ARM64
batch_size = 100
chunk_size = 1000

[MODELS]
# Model directories
toxicity_model_dir = models/toxicity/
bioactivity_model_dir = models/bioactivity/

# Model file patterns
toxicity_pattern = *.pkl
bioactivity_pattern = *.joblib

[UI]
# UI optimizations for high-DPI displays common on premium ARM64 devices
scale_factor = 1.0
window_width = 1200
window_height = 800

[API]
# Groq API settings
default_model = llama3-8b-8192
max_tokens = 1500
temperature = 0.7

[LOGGING]
log_level = INFO
log_file = chemical_analysis.log
"""
    
    with open("config.ini", "w") as f:
        f.write(config_content)
    
    print("✓ Configuration file created: config.ini")

def verify_installation():
    """Verify that all components are properly installed"""
    print("\n=== Verifying Installation ===")
    
    required_modules = [
        'tkinter', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'rdkit', 'sklearn', 'joblib', 'requests', 'groq'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            if module == 'sklearn':
                __import__('sklearn')
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠ Warning: {len(failed_imports)} modules failed to import")
        print("You may need to install these manually or use alternative methods")
        return False
    else:
        print(f"\n✓ All {len(required_modules)} required modules imported successfully")
        return True

def main():
    """Main setup function"""
    print("="*60)
    print("Chemical Analysis Platform Setup")
    print("Snapdragon X Elite / ARM64 Optimization")
    print("="*60)
    
    # Detect architecture
    arch = detect_architecture()
    
    # Install dependencies
    try:
        install_python_dependencies()
    except Exception as e:
        print(f"⚠ Some dependencies may have failed to install: {e}")
    
    # Setup directories
    setup_model_directory()
    
    # Create launcher and config
    create_launcher_script()
    create_config_file()
    
    # Verify installation
    success = verify_installation()
    
    print("\n" + "="*60)
    if success:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your model files in the models/ directory")
        print("2. Set your Groq API key in the application settings")
        print("3. Run the application:")
        
        if platform.system().lower() == "windows":
            print("   - Double-click launch_chemical_analysis.bat")
            print("   - Or run: python chemical_analysis_tkinter.py")
        else:
            print("   - Run: ./launch_chemical_analysis.sh")
            print("   - Or run: python3 chemical_analysis_tkinter.py")
    else:
        print("⚠ Setup completed with warnings")
        print("Some dependencies may need manual installation")
        print("Check the error messages above for details")
    
    print("="*60)

if __name__ == "__main__":
    main()
