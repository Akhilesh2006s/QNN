#!/usr/bin/env python3
"""
Quantum Fraud Detection Frontend Server
Serves the frontend and provides API endpoints for backend communication
"""

import os
import sys
import json
import subprocess
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

class QuantumFraudDetectionServer:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Configuration
        self.frontend_path = Path(__file__).parent
        self.backend_path = Path(__file__).parent.parent / "quantum_only_demo.py"
        
        # Ensure backend path is absolute and correct
        if not self.backend_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("quantum_only_demo.py"),
                Path(__file__).parent.parent.parent / "quantum_only_demo.py"
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    self.backend_path = alt_path
                    break
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        # Serve static files
        @self.app.route('/')
        def index():
            return send_from_directory(self.frontend_path, 'index.html')
            
        @self.app.route('/<path:filename>')
        def static_files(filename):
            return send_from_directory(self.frontend_path, filename)
        
        # API endpoints
        @self.app.route('/api/detect', methods=['POST'])
        def detect_fraud():
            return self.handle_detection(request.json)
            
        @self.app.route('/api/demo', methods=['POST'])
        def run_demo():
            return self.handle_demo(request.json)
            
        @self.app.route('/api/analyze_csv', methods=['POST'])
        def analyze_csv():
            return self.handle_csv_analysis(request.json)
            
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return self.check_backend_health()
    
    def handle_detection(self, data):
        """Handle individual transaction fraud detection"""
        try:
            # Prepare transaction data for quantum-only API
            api_data = {
                'command': 'detect',
                'transaction': data
            }
            transaction_data = json.dumps(api_data)
            
            # Call quantum-only backend
            cmd = ['python', str(self.backend_path), 'api']
            result = subprocess.run(
                cmd,
                input=transaction_data,
                text=True,
                capture_output=True,
                timeout=300  # 5 minutes timeout for quantum training
            )
            
            if result.returncode == 0:
                try:
                    response_data = json.loads(result.stdout)
                    return jsonify(response_data)
                except json.JSONDecodeError:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid JSON response from backend'
                    })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Backend error: {result.stderr}'
                })
                
        except subprocess.TimeoutExpired:
            return jsonify({
                'success': False,
                'error': 'Backend timeout'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Server error: {str(e)}'
            })
    
    def handle_demo(self, data):
        """Handle demo execution"""
        try:
            demo_type = data.get('demo_type', 'demo')
            options = data.get('options', {})
            
            # Prepare API data for quantum-only demo
            api_data = {
                'command': 'demo',
                'options': options
            }
            demo_data = json.dumps(api_data)
            
            # Call quantum-only backend
            cmd = ['python', str(self.backend_path), 'api']
            
            print(f"Running quantum-only demo...")
            
            result = subprocess.run(
                cmd,
                input=demo_data,
                text=True,
                capture_output=True,
                timeout=300  # 5 minutes timeout for quantum training
            )
            
            if result.returncode == 0:
                try:
                    response_data = json.loads(result.stdout)
                    return jsonify(response_data)
                except json.JSONDecodeError:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid JSON response from backend'
                    })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Backend error: {result.stderr}'
                })
                
        except subprocess.TimeoutExpired:
            return jsonify({
                'success': False,
                'error': 'Backend timeout'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Server error: {str(e)}'
            })
    
    def handle_csv_analysis(self, data):
        """Handle CSV analysis"""
        try:
            transactions = data.get('transactions', [])
            
            if not transactions:
                return jsonify({
                    'success': False,
                    'error': 'No transactions provided'
                })
            
            # Create temporary CSV file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                # Write header
                if transactions:
                    headers = list(transactions[0].keys())
                    f.write(','.join(headers) + '\n')
                    
                    # Write data
                    for transaction in transactions:
                        row = [str(transaction.get(header, '')) for header in headers]
                        f.write(','.join(row) + '\n')
                    
                    temp_csv_path = f.name
            
            try:
                # Call backend with CSV file
                cmd = ['python', str(self.backend_path), 'analyze_csv', '--input', temp_csv_path, '--json']
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout for quantum operations
                )
                
                if result.returncode == 0:
                    try:
                        response_data = json.loads(result.stdout)
                        return jsonify(response_data)
                    except json.JSONDecodeError:
                        return jsonify({
                            'success': False,
                            'error': 'Invalid JSON response from backend'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Backend error: {result.stderr}'
                    })
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_csv_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return jsonify({
                'success': False,
                'error': 'Backend timeout'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Server error: {str(e)}'
            })
    
    def check_backend_health(self):
        """Check if backend is accessible"""
        try:
            # Simple test with a quick command
            test_data = json.dumps({"command": "detect", "transaction": {"amount": 1000, "user_balance": 50000, "location": "Mumbai", "merchant": "Test"}})
            cmd = ['python', str(self.backend_path), 'api']
            result = subprocess.run(
                cmd,
                input=test_data,
                text=True,
                capture_output=True,
                timeout=60  # Longer timeout for quantum operations
            )
            
            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'status': 'Backend is accessible'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.stderr
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    def run(self, host='0.0.0.0', port=5000, debug=True):
        """Run the Flask server"""
        print("🚀 Starting Quantum Fraud Detection Frontend Server")
        print("=" * 50)
        print(f"📁 Serving files from: {self.frontend_path}")
        print(f"🔗 Backend path: {self.backend_path}")
        print(f"🌐 Server will be available at: http://localhost:{port}")
        print("=" * 50)
        
        # Test backend connectivity with longer timeout
        print("🔍 Testing backend connectivity...")
        try:
            # Simple test with a quick command
            test_data = json.dumps({"command": "detect", "transaction": {"amount": 1000, "user_balance": 50000, "location": "Mumbai", "merchant": "Test"}})
            cmd = ['python', str(self.backend_path), 'api']
            result = subprocess.run(
                cmd,
                input=test_data,
                text=True,
                capture_output=True,
                timeout=60  # Longer timeout for quantum operations
            )
            
            if result.returncode == 0:
                print("✅ Backend is accessible")
            else:
                print("⚠️ Backend may not be accessible:", result.stderr)
        except subprocess.TimeoutExpired:
            print("⚠️ Backend startup timeout (60s) - will use fallback mode")
        except Exception as e:
            print("⚠️ Backend may not be accessible:", str(e))
        
        # Start server
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    server = QuantumFraudDetectionServer()
    server.run()
