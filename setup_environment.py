#!/usr/bin/env python3
"""
Environment Setup Script for Quantum Fraud Detection System
Fixes NumPy compatibility issues and ensures proper installation
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Quantum Fraud Detection System - Environment Setup")
    print("=" * 60)
    
    # Step 1: Uninstall problematic packages
    print("\n📦 Step 1: Cleaning up incompatible packages...")
    packages_to_remove = [
        "numpy",
        "torch", 
        "qiskit",
        "qiskit-machine-learning"
    ]
    
    for package in packages_to_remove:
        run_command(f"pip uninstall -y {package}", f"Uninstalling {package}")
    
    # Step 2: Install NumPy 1.24.3 first (compatible with PyTorch 2.0.1)
    print("\n📦 Step 2: Installing compatible NumPy...")
    if not run_command("pip install numpy==1.24.3", "Installing NumPy 1.24.3"):
        print("❌ Failed to install NumPy. Please check your internet connection.")
        return False
    
    # Step 3: Install PyTorch with CPU support (more stable)
    print("\n📦 Step 3: Installing PyTorch...")
    if not run_command("pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu -f https://download.pytorch.org/whl/torch_stable.html", 
                      "Installing PyTorch 2.0.1 (CPU version)"):
        print("❌ Failed to install PyTorch. Trying alternative installation...")
        if not run_command("pip install torch==2.0.1", "Installing PyTorch 2.0.1 (standard)"):
            print("❌ Failed to install PyTorch. Please check your internet connection.")
            return False
    
    # Step 4: Install Qiskit and quantum libraries
    print("\n📦 Step 4: Installing quantum libraries...")
    if not run_command("pip install qiskit==0.44.1", "Installing Qiskit"):
        print("❌ Failed to install Qiskit. Continuing without quantum support...")
    
    if not run_command("pip install qiskit-machine-learning==0.6.1", "Installing Qiskit Machine Learning"):
        print("❌ Failed to install Qiskit Machine Learning. Continuing without quantum support...")
    
    # Step 5: Install remaining requirements
    print("\n📦 Step 5: Installing remaining dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing all requirements"):
        print("❌ Failed to install some requirements. Please check the error messages above.")
    
    # Step 6: Test the installation
    print("\n🧪 Step 6: Testing installation...")
    test_script = """
import sys
import numpy as np
import torch
print(f"✅ NumPy version: {np.__version__}")
print(f"✅ PyTorch version: {torch.__version__}")

try:
    from qiskit import QuantumCircuit, Aer
    from qiskit_machine_learning import ZZFeatureMap, RealAmplitudes, EstimatorQNN, TorchConnector
    print("✅ Quantum libraries imported successfully")
except ImportError as e:
    print(f"⚠️ Quantum libraries not available: {e}")
    print("⚠️ System will run in classical mode only")

print("✅ Environment setup completed successfully!")
"""
    
    with open("test_installation.py", "w") as f:
        f.write(test_script)
    
    run_command("python test_installation.py", "Testing installation")
    
    # Clean up test file
    if os.path.exists("test_installation.py"):
        os.remove("test_installation.py")
    
    print("\n🎉 Environment setup completed!")
    print("\n📋 Next steps:")
    print("1. Run: python quantum_neural_network_fraud_detector.py demo_graph --json")
    print("2. Or run: python quantum_neural_network_fraud_detector.py demo_advanced")
    print("3. For real-time demo: python quantum_neural_network_fraud_detector.py demo_realtime")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")

