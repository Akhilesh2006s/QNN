#!/usr/bin/env python3
"""
Quantum Fraud Detection Frontend Startup Script
Provides easy options to start the frontend
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    print("🎯 Quantum Fraud Detection Frontend")
    print("=" * 50)
    print("Choose an option:")
    print("1. Start with Flask server (full functionality)")
    print("2. Open test page (frontend only)")
    print("3. Open main page (frontend only)")
    print("4. Check dependencies")
    print("5. Exit")
    print("=" * 50)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("❌ Flask is not installed")
        print("   Run: pip install flask flask-cors")
        return False
    
    try:
        import flask_cors
        print("✅ Flask-CORS is installed")
    except ImportError:
        print("❌ Flask-CORS is not installed")
        print("   Run: pip install flask-cors")
        return False
    
    # Check if backend exists
    backend_path = Path("../quantum_neural_network_fraud_detector.py")
    if backend_path.exists():
        print("✅ Backend found")
    else:
        print("⚠️ Backend not found in parent directory")
        print("   Make sure quantum_neural_network_fraud_detector.py is in the parent directory")
    
    return True

def start_flask_server():
    """Start the Flask server"""
    print("🚀 Starting Flask server...")
    
    if not check_dependencies():
        print("❌ Dependencies not met. Please install required packages.")
        return
    
    try:
        # Start the server
        subprocess.run([sys.executable, "server.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
    except FileNotFoundError:
        print("❌ server.py not found")

def open_test_page():
    """Open the test page in browser"""
    test_file = Path("test.html")
    if test_file.exists():
        print("🌐 Opening test page...")
        webbrowser.open(f"file://{test_file.absolute()}")
    else:
        print("❌ test.html not found")

def open_main_page():
    """Open the main page in browser"""
    main_file = Path("index.html")
    if main_file.exists():
        print("🌐 Opening main page...")
        webbrowser.open(f"file://{main_file.absolute()}")
    else:
        print("❌ index.html not found")

def main():
    while True:
        print_banner()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                start_flask_server()
                break
            elif choice == "2":
                open_test_page()
            elif choice == "3":
                open_main_page()
            elif choice == "4":
                check_dependencies()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

