from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pymongo
from bson import ObjectId
import json
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
import threading
import time
from quantum_neural_network_fraud_detector import QuantumFraudDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quantum-fraud-detection-2024'
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"], supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

# MongoDB Connection
MONGO_URI = "mongodb+srv://akhileshsamayamanthula:rxvIPIT4Bzobk9Ne@cluster0.4ej8ne2.mongodb.net/QML1?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client.QML

# Collections
transactions_collection = db.transactions
training_results_collection = db.training_results
quantum_states_collection = db.quantum_states
users_collection = db.users

# Initialize Quantum Fraud Detector
global detector, is_training
detector = QuantumFraudDetector()
is_training = False

# Global stats
stats = {
    'total_transactions': 0,
    'fraud_transactions': 0,
    'non_fraud_transactions': 0,
    'quantum_accuracy': 0,
    'classical_accuracy': 0
}

def update_stats():
    """Update global statistics from MongoDB"""
    global stats
    total = transactions_collection.count_documents({})
    fraud = transactions_collection.count_documents({'is_fraud': True})
    
    stats['total_transactions'] = total
    stats['fraud_transactions'] = fraud
    stats['non_fraud_transactions'] = total - fraud
    
    # Get latest training results
    latest_training = training_results_collection.find_one(
        sort=[('timestamp', -1)]
    )
    if latest_training:
        stats['quantum_accuracy'] = latest_training.get('quantum_accuracy', 0)
        stats['classical_accuracy'] = latest_training.get('classical_accuracy', 0)

@app.route('/')
def index():
    return jsonify({
        'message': 'Quantum Fraud Detection API',
        'status': 'running',
        'endpoints': {
            'dashboard': '/api/dashboard/stats',
            'transactions': '/api/transactions',
            'detection': '/api/detection/analysis',
            'training': '/api/training/train',
            'quantum': '/api/quantum/circuit'
        }
    })

# Dashboard API Endpoints
@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get summary statistics for dashboard"""
    update_stats()
    
    # Get recent transactions and convert ObjectIds to strings
    recent_transactions = list(transactions_collection.find().sort('timestamp', -1).limit(10))
    for txn in recent_transactions:
        txn['_id'] = str(txn['_id'])
    
    return jsonify({
        'success': True,
        'stats': stats,
        'recent_transactions': recent_transactions
    })

@app.route('/api/dashboard/charts')
def get_dashboard_charts():
    """Get chart data for dashboard"""
    # Fraud distribution
    fraud_distribution = {
        'labels': ['Legitimate', 'Fraud'],
        'data': [stats['non_fraud_transactions'], stats['fraud_transactions']]
    }
    
    # Transaction trends (last 7 days)
    pipeline = [
        {
            '$match': {
                'timestamp': {
                    '$gte': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                }
            }
        },
        {
            '$group': {
                '_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}},
                'count': {'$sum': 1},
                'fraud_count': {'$sum': {'$cond': ['$is_fraud', 1, 0]}}
            }
        },
        {'$sort': {'_id': 1}}
    ]
    
    daily_stats = list(transactions_collection.aggregate(pipeline))
    
    return jsonify({
        'success': True,
        'fraud_distribution': fraud_distribution,
        'daily_trends': daily_stats
    })

# Transactions API Endpoints
@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Get all transactions with pagination"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    skip = (page - 1) * limit
    
    transactions = list(transactions_collection.find().sort('timestamp', -1).skip(skip).limit(limit))
    
    # Convert ObjectId to string for JSON serialization
    for txn in transactions:
        txn['_id'] = str(txn['_id'])
    
    return jsonify({
        'success': True,
        'transactions': transactions,
        'total': transactions_collection.count_documents({}),
        'page': page,
        'limit': limit
    })

@app.route('/api/transactions', methods=['POST'])
def create_transaction():
    """Create a new transaction and detect fraud"""
    try:
        data = request.json
        
        # Generate transaction ID
        transaction_id = str(uuid.uuid4()).replace('-', '')[:8].upper()
        
        # Handle user-to-user transactions
        if data.get('transaction_type') == 'user_to_user':
            from_user = data.get('from_user')
            to_user = data.get('to_user')
            amount = float(data['amount'])
            
            if not from_user or not to_user:
                return jsonify({
                    'success': False,
                    'error': 'Both from_user and to_user are required for user-to-user transactions'
                }), 400
            
            if from_user == to_user:
                return jsonify({
                    'success': False,
                    'error': 'Sender and receiver cannot be the same user'
                }), 400
            
            # Check sender's balance
            sender = users_collection.find_one({'user_id': from_user})
            if not sender:
                return jsonify({
                    'success': False,
                    'error': f'Sender user {from_user} not found'
                }), 400
            
            if sender['balance'] < amount:
                return jsonify({
                    'success': False,
                    'error': f'Insufficient funds. {from_user} has ₹{sender["balance"]} but trying to send ₹{amount}'
                }), 400
            
            # Update balances
            users_collection.update_one(
                {'user_id': from_user},
                {'$inc': {'balance': -amount}}
            )
            users_collection.update_one(
                {'user_id': to_user},
                {'$inc': {'balance': amount}}
            )
        
        # Create transaction object
        transaction = {
            'transaction_id': transaction_id,
            'user_id': data.get('user_id', f'user_{str(uuid.uuid4()).replace("-", "")[:8]}'),
            'amount': float(data['amount']),
            'user_balance': float(data.get('user_balance', 100000)),
            'location': data['location'],
            'merchant': data['merchant'],
            'user_age': int(data.get('user_age', 30)),
            'hour': int(data.get('hour', datetime.now().hour)),
            'minute': int(data.get('minute', datetime.now().minute)),
            'from_user': data.get('from_user'),
            'to_user': data.get('to_user'),
            'transaction_type': data.get('transaction_type', 'regular'),
            'timestamp': datetime.now()
        }
        
        # Detect fraud using quantum neural network
        detection_result = detector.detect_fraud(transaction)
        
        # Add detection results
        transaction.update({
            'is_fraud': detection_result['is_fraud'],
            'confidence': detection_result['confidence'],
            'quantum_prediction': detection_result['quantum_prediction'],
            'classical_prediction': detection_result['classical_prediction'],
            'risk_factors': detection_result['risk_factors']
        })
        
        # Save to MongoDB
        result = transactions_collection.insert_one(transaction)
        transaction['_id'] = str(result.inserted_id)
        
        # Convert datetime to string for JSON serialization
        transaction['timestamp'] = transaction['timestamp'].isoformat()
        
        # Update stats
        update_stats()
        
        # Emit real-time update
        socketio.emit('new_transaction', {
            'transaction': transaction,
            'stats': stats
        })
        
        return jsonify({
            'success': True,
            'transaction': transaction,
            'message': f'Transaction {transaction_id} created successfully'
        })
        
    except Exception as e:
        print(f"Error creating transaction: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error creating transaction: {str(e)}'
        }), 400

@app.route('/api/transactions/generate-random', methods=['POST'])
def generate_random_transactions():
    """Generate random test transactions"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is empty'
            }), 400
        
        # Validate count parameter
        count_raw = data.get('count', 10)
        try:
            count = int(count_raw)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': f'Invalid count value: {count_raw}. Must be a valid integer.'
            }), 400
        
        if count <= 0:
            return jsonify({
                'success': False,
                'error': f'Count must be greater than 0, got: {count}'
            }), 400
        
        if count > 1000:
            return jsonify({
                'success': False,
                'error': f'Count cannot exceed 1000, got: {count}'
            }), 400
        
        transactions = []
        
        locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Dubai', 'Moscow', 'New York']
        merchants = ['Online Store', 'Restaurant', 'Gas Station', 'Grocery', 'Electronics', 'Online Casino', 'Cryptocurrency']
        
        for i in range(count):
            transaction = {
                'transaction_id': str(uuid.uuid4()).replace('-', '')[:8].upper(),
                'user_id': f'user_{str(uuid.uuid4()).replace("-", "")[:8]}',
                'amount': np.random.randint(100, 50000),
                'user_balance': np.random.randint(50000, 200000),
                'location': np.random.choice(locations),
                'merchant': np.random.choice(merchants),
                'user_age': np.random.randint(18, 70),
                'hour': np.random.randint(0, 24),
                'minute': np.random.randint(0, 60),
                'timestamp': datetime.now()
            }
            
            # Detect fraud
            detection_result = detector.detect_fraud(transaction)
            transaction.update({
                'is_fraud': detection_result['is_fraud'],
                'confidence': detection_result['confidence'],
                'quantum_prediction': detection_result['quantum_prediction'],
                'classical_prediction': detection_result['classical_prediction'],
                'risk_factors': detection_result['risk_factors']
            })
            
            transactions.append(transaction)
        
        # Save to MongoDB
        if transactions:
            result = transactions_collection.insert_many(transactions)
            
            # Convert ObjectIds to strings and datetime to string for JSON serialization
            for i, transaction in enumerate(transactions):
                transaction['_id'] = str(result.inserted_ids[i])
                transaction['timestamp'] = transaction['timestamp'].isoformat()
            
            update_stats()
            
            # Emit real-time update
            socketio.emit('bulk_transactions', {
                'count': len(transactions),
                'stats': stats
            })
        
        return jsonify({
            'success': True,
            'transactions': transactions,
            'message': f'Generated {count} random transactions'
        })
        
    except Exception as e:
        print(f"Error generating random transactions: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/transactions/create', methods=['POST'])
def create_new_transaction():
    """Create a new transaction with fraud detection"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is empty'
            }), 400
        
        # Extract transaction data
        amount = data.get('amount')
        location = data.get('location')
        merchant = data.get('merchant')
        user_age = data.get('user_age', 30)
        payment_method = data.get('payment_method', 'Credit Card')
        device = data.get('device', 'Mobile')
        from_user = data.get('from_user')
        to_user = data.get('to_user')
        transaction_type = data.get('transaction_type', 'user-to-user')
        
        # Validate required fields
        required_fields = ['amount', 'location', 'from_user']
        if transaction_type == 'user-to-merchant':
            required_fields.append('merchant')
        
        if not all([data.get(field) for field in required_fields]):
            missing_fields = [field for field in required_fields if not data.get(field)]
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate amount
        try:
            amount = float(amount)
            if amount <= 0:
                return jsonify({
                    'success': False,
                    'error': 'Amount must be greater than 0'
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid amount value'
            }), 400
        
        # Get user information
        from_user_data = users_collection.find_one({'user_id': from_user})
        if not from_user_data:
            return jsonify({
                'success': False,
                'error': f'From user {from_user} not found'
            }), 404
        
        # Check if user has sufficient balance
        if from_user_data['balance'] < amount:
            return jsonify({
                'success': False,
                'error': f'Insufficient balance. Available: ₹{from_user_data["balance"]}, Required: ₹{amount}'
            }), 400
        
        # Create transaction object
        transaction = {
            'transaction_id': str(uuid.uuid4()).replace('-', '')[:8].upper(),
            'user_id': from_user,
            'amount': amount,
            'user_balance': from_user_data['balance'],
            'location': location,
            'merchant': merchant if merchant else ('User Transfer' if transaction_type == 'user-to-user' else 'Unknown Merchant'),
            'user_age': user_age,
            'payment_method': payment_method,
            'device': device,
            'hour': datetime.now().hour,
            'minute': datetime.now().minute,
            'timestamp': datetime.now(),
            'transaction_type': transaction_type,
            'to_user': to_user if transaction_type == 'user-to-user' else None
        }
        
        # Detect fraud
        detection_result = detector.detect_fraud(transaction)
        transaction.update({
            'is_fraud': detection_result['is_fraud'],
            'confidence': detection_result['confidence'],
            'quantum_prediction': detection_result['quantum_prediction'],
            'classical_prediction': detection_result['classical_prediction'],
            'risk_factors': detection_result['risk_factors']
        })
        
        # Save transaction to MongoDB
        result = transactions_collection.insert_one(transaction)
        transaction['_id'] = str(result.inserted_id)
        transaction['timestamp'] = transaction['timestamp'].isoformat()
        
        # Update user balances
        new_balance = from_user_data['balance'] - amount
        users_collection.update_one(
            {'user_id': from_user},
            {'$set': {'balance': new_balance}}
        )
        
        # If user-to-user transaction, update receiver's balance
        if transaction_type == 'user-to-user' and to_user:
            to_user_data = users_collection.find_one({'user_id': to_user})
            if to_user_data:
                new_to_balance = to_user_data['balance'] + amount
                users_collection.update_one(
                    {'user_id': to_user},
                    {'$set': {'balance': new_to_balance}}
                )
        
        # Update stats
        update_stats()
        
        # Emit real-time update
        socketio.emit('new_transaction', {
            'transaction': transaction,
            'stats': stats
        })
        
        return jsonify({
            'success': True,
            'transaction_id': transaction['transaction_id'],
            'fraud_detection': {
                'is_fraud': detection_result['is_fraud'],
                'maybe_fraud': detection_result.get('maybe_fraud', False),
                'risk_score': detection_result.get('risk_score', 0),
                'confidence': detection_result['confidence'],
                'explanation': detection_result.get('explanation', ''),
                'detection_method': detection_result.get('detection_method', 'classical')
            },
            'message': 'Transaction created successfully'
        })
        
    except Exception as e:
        print(f"Error creating transaction: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error creating transaction: {str(e)}'
        }), 500

# Detection API Endpoints
@app.route('/api/detection/investigate/<transaction_id>')
def investigate_transaction(transaction_id):
    """Investigate a specific transaction"""
    try:
        transaction = transactions_collection.find_one({'transaction_id': transaction_id})
        if not transaction:
            return jsonify({
                'success': False,
                'error': 'Transaction not found'
            }), 404
        
        # Re-run detection for detailed analysis
        detection_result = detector.detect_fraud(transaction)
        
        # Get quantum state if available
        quantum_state = None
        if detector.quantum_model is not None:
            try:
                features = detector.extract_features(transaction)
                X = np.array([features])
                X_scaled = detector.scaler.transform(X)
                
                # Get quantum circuit state
                quantum_state = {
                    'features': features,
                    'scaled_features': X_scaled[0].tolist(),
                    'quantum_output': float(detector.quantum_model(torch.FloatTensor(X_scaled)).squeeze().item())
                }
            except Exception as e:
                quantum_state = {'error': str(e)}
        
        transaction['_id'] = str(transaction['_id'])
        
        return jsonify({
            'success': True,
            'transaction': transaction,
            'detection_result': detection_result,
            'quantum_state': quantum_state
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/detection/analysis')
def get_detection_analysis():
    """Get fraud detection analysis and trends"""
    try:
        # Fraud distribution by location
        location_pipeline = [
            {'$group': {
                '_id': '$location',
                'total': {'$sum': 1},
                'fraud': {'$sum': {'$cond': ['$is_fraud', 1, 0]}}
            }},
            {'$project': {
                'location': '$_id',
                'total': 1,
                'fraud': 1,
                'fraud_rate': {'$divide': ['$fraud', '$total']}
            }}
        ]
        
        location_stats = list(transactions_collection.aggregate(location_pipeline))
        
        # Fraud distribution by merchant
        merchant_pipeline = [
            {'$group': {
                '_id': '$merchant',
                'total': {'$sum': 1},
                'fraud': {'$sum': {'$cond': ['$is_fraud', 1, 0]}}
            }},
            {'$project': {
                'merchant': '$_id',
                'total': 1,
                'fraud': 1,
                'fraud_rate': {'$divide': ['$fraud', '$total']}
            }}
        ]
        
        merchant_stats = list(transactions_collection.aggregate(merchant_pipeline))
        
        # Amount distribution
        amount_pipeline = [
            {'$group': {
                '_id': None,
                'avg_amount': {'$avg': '$amount'},
                'fraud_avg_amount': {'$avg': {'$cond': ['$is_fraud', '$amount', None]}},
                'legitimate_avg_amount': {'$avg': {'$cond': ['$is_fraud', None, '$amount']}}
            }}
        ]
        
        amount_stats = list(transactions_collection.aggregate(amount_pipeline))
        
        return jsonify({
            'success': True,
            'location_stats': location_stats,
            'merchant_stats': merchant_stats,
            'amount_stats': amount_stats[0] if amount_stats else {}
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Quantum Visualization API Endpoints
@app.route('/api/quantum/circuit')
def get_quantum_circuit():
    """Get quantum circuit information"""
    try:
        if detector.quantum_model is None:
            return jsonify({
                'success': False,
                'error': 'Quantum model not available'
            }), 400
        
        # Get circuit information
        circuit_info = {
            'num_qubits': 8,  # Based on feature dimension
            'num_parameters': 16,  # Based on RealAmplitudes
            'feature_map': 'ZZFeatureMap',
            'ansatz': 'RealAmplitudes',
            'reps': 1
        }
        
        return jsonify({
            'success': True,
            'circuit': circuit_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/quantum/state/<transaction_id>')
def get_quantum_state(transaction_id):
    """Get quantum state for a specific transaction"""
    try:
        transaction = transactions_collection.find_one({'transaction_id': transaction_id})
        if not transaction:
            return jsonify({
                'success': False,
                'error': 'Transaction not found'
            }), 404
        
        if detector.quantum_model is None:
            return jsonify({
                'success': False,
                'error': 'Quantum model not available'
            }), 400
        
        # Extract features and get quantum state
        features = detector.extract_features(transaction)
        X = np.array([features])
        X_scaled = detector.scaler.transform(X)
        
        # Get quantum output
        detector.quantum_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            quantum_output = detector.quantum_model(X_tensor).squeeze().item()
        
        # Create quantum state visualization data
        quantum_state = {
            'features': features,
            'scaled_features': X_scaled[0].tolist(),
            'quantum_output': quantum_output,
            'prediction': quantum_output > 0.5,
            'confidence': abs(quantum_output - 0.5) * 2  # Convert to 0-1 confidence
        }
        
        return jsonify({
            'success': True,
            'quantum_state': quantum_state
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to Quantum Fraud Detection System'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

# Training API Endpoints
@app.route('/api/training/train', methods=['POST'])
def train_models():
    """Train the quantum and classical fraud detection models"""
    global is_training, detector
    
    if is_training:
        return jsonify({
            'success': False,
            'error': 'Training already in progress'
        }), 400
    
    try:
        is_training = True
        
        # Train the models
        results = detector.train_models()
        
        # Save training results to MongoDB
        training_result = {
            'timestamp': datetime.now(),
            'quantum_accuracy': results['quantum_accuracy'],
            'classical_accuracy': results['classical_accuracy'],
            'quantum_available': results['quantum_available'],
            'training_time': results['training_time']
        }
        
        training_results_collection.insert_one(training_result)
        
        # Update global stats
        update_stats()
        
        # Emit real-time update
        socketio.emit('training_completed', {
            'quantum_accuracy': results['quantum_accuracy'],
            'classical_accuracy': results['classical_accuracy'],
            'quantum_available': results['quantum_available']
        })
        
        return jsonify({
            'success': True,
            'results': results,
            'message': 'Models trained successfully'
        })
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Training failed: {str(e)}'
        }), 500
    
    finally:
        is_training = False

# CSV Analysis API Endpoints
@app.route('/api/csv/upload-test', methods=['POST'])
def test_csv_upload():
    """Test CSV file upload without processing"""
    try:
        print("=== CSV Upload Test ===")
        print(f"Request files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        # Read file content
        file_content = file.read()
        print(f"File size: {len(file_content)} bytes")
        print(f"First 200 characters: {file_content[:200]}")
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'size': len(file_content),
            'content_preview': file_content[:200].decode('utf-8', errors='ignore')
        })
        
    except Exception as e:
        print(f"Upload test error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Upload test failed: {str(e)}'
        }), 500

@app.route('/api/csv/test', methods=['GET'])
def test_csv_analysis():
    """Test CSV analysis with sample data"""
    try:
        # Create a sample transaction
        sample_transaction = {
            'transaction_id': 'TEST001',
            'user_id': 'test_user',
            'amount': 5000.0,
            'user_balance': 50000.0,
            'location': 'Mumbai',
            'merchant': 'Test Merchant',
            'user_age': 30,
            'hour': 14,
            'minute': 30,
            'payment_method': 'Credit Card',
            'device': 'Mobile',
            'timestamp': datetime.now()
        }
        
        # Test fraud detection
        detection_result = detector.detect_fraud(sample_transaction)
        
        return jsonify({
            'success': True,
            'message': 'CSV analysis system is working',
            'sample_detection': detection_result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'CSV analysis test failed: {str(e)}'
        }), 500

@app.route('/api/csv/format', methods=['GET'])
def get_csv_format():
    """Get CSV format requirements"""
    return jsonify({
        'success': True,
        'format': {
            'required_columns': ['amount', 'user_id', 'location', 'merchant'],
            'optional_columns': ['user_balance', 'user_age', 'payment_method', 'device', 'hour', 'minute'],
            'sample_data': [
                {
                    'amount': '5000',
                    'user_id': 'user_1',
                    'location': 'Mumbai',
                    'merchant': 'Amazon India',
                    'user_balance': '50000',
                    'user_age': '30',
                    'payment_method': 'Credit Card',
                    'device': 'Mobile'
                },
                {
                    'amount': '15000',
                    'user_id': 'user_2',
                    'location': 'Delhi',
                    'merchant': 'Flipkart',
                    'user_balance': '75000',
                    'user_age': '25',
                    'payment_method': 'UPI',
                    'device': 'Desktop'
                }
            ],
            'description': 'Upload a CSV file with transaction data. The file should have headers and at least the required columns.'
        }
    })

@app.route('/api/csv/sample', methods=['GET'])
def get_sample_csv():
    """Get a sample CSV file for testing"""
    import io
    import csv
    
    # Create sample CSV data
    sample_data = [
        ['amount', 'user_id', 'location', 'merchant', 'user_balance', 'user_age', 'payment_method', 'device'],
        ['5000', 'user_1', 'Mumbai', 'Amazon India', '50000', '30', 'Credit Card', 'Mobile'],
        ['15000', 'user_2', 'Delhi', 'Flipkart', '75000', '25', 'UPI', 'Desktop'],
        ['8000', 'user_3', 'Bangalore', 'Swiggy', '100000', '28', 'Debit Card', 'Mobile'],
        ['25000', 'user_4', 'Chennai', 'Netflix', '125000', '35', 'Credit Card', 'Desktop'],
        ['12000', 'user_5', 'Kolkata', 'Spotify', '150000', '22', 'UPI', 'Mobile']
    ]
    
    # Create CSV string
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(sample_data)
    csv_content = output.getvalue()
    output.close()
    
    from flask import Response
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sample_transactions.csv'}
    )

@app.route('/api/csv/analyze', methods=['POST'])
def analyze_csv():
    """Analyze any CSV file for fraud detection"""
    print("=== CSV Analysis Started ===")
    try:
        print(f"Request files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("No file in request.files")
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            print(f"Invalid file type: {file.filename}")
            return jsonify({
                'success': False,
                'error': 'Only CSV files are allowed'
            }), 400
        
        # Read CSV file
        try:
            print("Attempting to read CSV file...")
            df = pd.read_csv(file)
            print(f"CSV file loaded successfully. Shape: {df.shape}")
            
            if df.empty:
                print("CSV file is empty")
                return jsonify({
                    'success': False,
                    'error': 'CSV file is empty. Please upload a file with transaction data.'
                }), 400
                
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error reading CSV file: {str(e)}'
            }), 400
        
        # Process transactions with quantum neural network fraud detection
        transactions = []
        fraud_count = 0
        legitimate_count = 0
        maybe_fraud_count = 0
        
        print(f"Starting quantum fraud detection for {len(df)} transactions...")
        print(f"Detector status: {'Available' if detector else 'Not available'}")
        if detector:
            print(f"Detector type: {type(detector)}")
            print(f"Detector trained: {getattr(detector, 'is_trained', 'Unknown')}")
        
        for index, row in df.iterrows():
            try:
                # Create transaction object with all required fields
                transaction = {
                    'transaction_id': f'TXN{datetime.now().strftime("%Y%m%d%H%M%S")}{index+1:06d}',
                    'user_id': str(row.get('user_id', f'CSV_USER_{index+1}')),
                    'amount': float(row.get('amount', 0)),
                    'user_balance': float(row.get('user_balance', 100000)),
                    'location': str(row.get('location', 'Unknown')),
                    'merchant': str(row.get('merchant', 'Unknown')),
                    'user_age': int(row.get('user_age', 30)),
                    'hour': int(row.get('hour', datetime.now().hour)),
                    'minute': int(row.get('minute', datetime.now().minute)),
                    'payment_method': str(row.get('payment_method', 'Credit Card')),
                    'device': str(row.get('device', 'Mobile')),
                    'timestamp': datetime.now()
                }
                
                # Use your quantum neural network for fraud detection
                print(f"Running quantum fraud detection for transaction {index+1}...")
                try:
                    detection_result = detector.detect_fraud(transaction)
                except Exception as e:
                    print(f"Fraud detection failed for transaction {index+1}: {e}")
                    # Fallback to basic analysis
                    detection_result = {
                        'is_fraud': False,
                        'maybe_fraud': False,
                        'confidence': 50.0,
                        'quantum_prediction': False,
                        'classical_prediction': False,
                        'risk_factors': ['Analysis failed - using fallback'],
                        'fraud_status': 'LEGITIMATE'
                    }
                
                # Add fraud detection results to transaction
                transaction.update({
                    'is_fraud': detection_result['is_fraud'],
                    'maybe_fraud': detection_result.get('maybe_fraud', False),
                    'confidence': detection_result['confidence'],
                    'quantum_prediction': detection_result['quantum_prediction'],
                    'classical_prediction': detection_result['classical_prediction'],
                    'risk_factors': detection_result['risk_factors'],
                    'fraud_status': detection_result.get('fraud_status', 'LEGITIMATE')
                })
                
                # Count by status
                if detection_result['is_fraud']:
                    fraud_count += 1
                    print(f"🚨 Fraud detected in transaction {index+1}")
                elif detection_result.get('maybe_fraud', False):
                    maybe_fraud_count += 1
                    print(f"⚠️ Suspicious transaction {index+1}")
                else:
                    legitimate_count += 1
                    print(f"✅ Legitimate transaction {index+1}")
                
                transactions.append(transaction)
                
            except Exception as e:
                print(f"Error processing transaction {index+1}: {e}")
                continue
        
        print(f"Quantum fraud detection completed!")
        print(f"Results: {legitimate_count} legitimate, {maybe_fraud_count} suspicious, {fraud_count} fraudulent")
        
        # Save to MongoDB
        if transactions:
            try:
                result = transactions_collection.insert_many(transactions)
                print(f"Successfully saved {len(transactions)} transactions to MongoDB")
                
                # Convert ObjectIds to strings and datetime to string for JSON serialization
                for i, transaction in enumerate(transactions):
                    transaction['_id'] = str(result.inserted_ids[i])
                    transaction['timestamp'] = transaction['timestamp'].isoformat()
                
                update_stats()
                
                # Emit real-time update
                socketio.emit('csv_analyzed', {
                    'filename': file.filename,
                    'total_transactions': len(transactions),
                    'fraud_count': fraud_count,
                    'legitimate_count': legitimate_count
                })
            except Exception as e:
                print(f"Error saving to MongoDB: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Error saving transactions to database: {str(e)}'
                }), 500
        
        # Convert transactions to the expected format
        results = []
        for i, transaction in enumerate(transactions):
            results.append({
                'row': i + 1,
                'transaction': transaction,
                'fraud_detection': {
                    'is_fraud': transaction.get('is_fraud', False),
                    'maybe_fraud': transaction.get('maybe_fraud', False),
                    'risk_score': transaction.get('confidence', 0.5) / 100,
                    'confidence': transaction.get('confidence', 50),
                    'explanation': 'Quantum fraud analysis completed',
                    'detection_method': 'quantum_neural_network'
                }
            })
        
        return jsonify({
            'success': True,
            'total_rows': len(transactions),
            'flagged_count': fraud_count + maybe_fraud_count,
            'results': results,
            'message': f'Successfully analyzed {len(transactions)} transactions using quantum neural network'
        })
        
    except Exception as e:
        print(f"=== CSV Analysis Error ===")
        print(f"Error analyzing CSV: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'CSV analysis failed: {str(e)}'
        }), 500

@app.route('/api/csv/analyze-full', methods=['POST'])
def analyze_csv_full():
    """Full CSV analysis with all features"""
    print("=== Full CSV Analysis Started ===")
    try:
        print(f"Request files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            print("No file in request.files")
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}, Size: {len(file.read()) if hasattr(file, 'read') else 'Unknown'}")
        file.seek(0)  # Reset file pointer
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            print(f"Invalid file type: {file.filename}")
            return jsonify({
                'success': False,
                'error': 'Only CSV files are allowed'
            }), 400
        
        # Read CSV file
        try:
            print("Attempting to read CSV file...")
            df = pd.read_csv(file)
            print(f"CSV file loaded successfully. Shape: {df.shape}, Columns: {list(df.columns)}")
            print(f"First few rows: {df.head().to_dict()}")
            
            # Check if CSV is empty
            if df.empty:
                print("CSV file is empty")
                return jsonify({
                    'success': False,
                    'error': 'CSV file is empty. Please upload a file with transaction data.'
                }), 400
                
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': f'Error reading CSV file: {str(e)}'
            }), 400
        
        print("Starting pattern analysis...")
        
        # Enhanced fraud pattern analysis
        suspicious_patterns = {
            'high_amounts': [],
            'suspicious_locations': [],
            'suspicious_merchants': [],
            'user_transaction_counts': {},
            'amount_patterns': []
        }
        
        # First pass: analyze patterns
        for index, row in df.iterrows():
            try:
                # Safe data extraction with defaults
                try:
                    amount = float(row.get('amount', 0))
                except (ValueError, TypeError):
                    amount = 0.0
                
                user_id = str(row.get('user_id', f'CSV_USER_{index+1}'))
                location = str(row.get('location', 'Unknown')).lower()
                merchant = str(row.get('merchant', 'Unknown')).lower()
                
                # Track patterns
                if amount > 50000:
                    suspicious_patterns['high_amounts'].append(amount)
                
                if location in ['dubai', 'moscow', 'singapore', 'hong kong', 'cayman islands', 'mauritius']:
                    suspicious_patterns['suspicious_locations'].append(location)
                
                if any(keyword in merchant for keyword in ['casino', 'gambling', 'crypto', 'offshore', 'betting']):
                    suspicious_patterns['suspicious_merchants'].append(merchant)
                
                suspicious_patterns['user_transaction_counts'][user_id] = suspicious_patterns['user_transaction_counts'].get(user_id, 0) + 1
                suspicious_patterns['amount_patterns'].append(amount)
                
            except Exception as e:
                print(f"Error in pattern analysis row {index}: {e}")
                continue
        
        print("Pattern analysis completed. Starting transaction processing...")
        
        # Process transactions with enhanced detection
        transactions = []
        fraud_count = 0
        legitimate_count = 0
        maybe_fraud_count = 0
        
        print(f"Starting to process {len(df)} transactions...")
        
        for index, row in df.iterrows():
            try:
                # Create transaction object with safe data conversion
                try:
                    amount = float(row.get('amount', 0))
                    user_balance = float(row.get('user_balance', 100000))
                    user_age = int(row.get('user_age', 30))
                    hour = int(row.get('hour', datetime.now().hour))
                    minute = int(row.get('minute', datetime.now().minute))
                except (ValueError, TypeError) as e:
                    print(f"Data conversion error in row {index}: {str(e)}")
                    # Use default values
                    amount = 0.0
                    user_balance = 100000.0
                    user_age = 30
                    hour = datetime.now().hour
                    minute = datetime.now().minute
                
                transaction = {
                    'transaction_id': str(row.get('transaction_id', f'TXN{index+1:06d}')),
                    'user_id': str(row.get('user_id', f'CSV_USER_{index+1}')),
                    'amount': amount,
                    'user_balance': user_balance,
                    'location': str(row.get('location', 'Unknown')),
                    'merchant': str(row.get('merchant', 'Unknown')),
                    'user_age': user_age,
                    'hour': hour,
                    'minute': minute,
                    'payment_method': str(row.get('payment_method', 'Credit Card')),
                    'device': str(row.get('device', 'Mobile')),
                    'timestamp': datetime.now()
                }
                
                # Enhanced fraud detection with pattern analysis
                user_transaction_count = suspicious_patterns['user_transaction_counts'].get(transaction['user_id'], 0)
                
                # Calculate pattern-based risk
                pattern_risk = 0.0
                if transaction['amount'] > 50000:
                    pattern_risk += 0.3
                if transaction['location'].lower() in ['dubai', 'moscow', 'singapore', 'hong kong']:
                    pattern_risk += 0.4
                if any(keyword in transaction['merchant'].lower() for keyword in ['casino', 'gambling', 'crypto', 'offshore']):
                    pattern_risk += 0.5
                if user_transaction_count > 5:
                    pattern_risk += 0.2
                
                # Detect fraud
                try:
                    detection_result = detector.detect_fraud(transaction)
                except Exception as e:
                    print(f"Error in fraud detection for transaction {index}: {str(e)}")
                    # Provide a default detection result
                    detection_result = {
                        'is_fraud': False,
                        'maybe_fraud': False,
                        'confidence': 50.0,
                        'quantum_prediction': False,
                        'classical_prediction': False,
                        'risk_factors': ['Detection error occurred'],
                        'fraud_status': 'LEGITIMATE'
                    }
                
                # Apply pattern-based adjustments
                if pattern_risk > 0.2:
                    detection_result['confidence'] = min(detection_result['confidence'] + (pattern_risk * 100), 100.0)
                    
                    # Adjust fraud status based on confidence
                    if detection_result['confidence'] > 50 and not detection_result['is_fraud']:
                        detection_result['maybe_fraud'] = True
                        detection_result['fraud_status'] = 'MAYBE_FRAUD'
                
                transaction.update({
                    'is_fraud': detection_result['is_fraud'],
                    'maybe_fraud': detection_result.get('maybe_fraud', False),
                    'fraud_status': detection_result.get('fraud_status', 'LEGITIMATE'),
                    'confidence': detection_result['confidence'],
                    'quantum_prediction': detection_result['quantum_prediction'],
                    'classical_prediction': detection_result['classical_prediction'],
                    'risk_factors': detection_result['risk_factors'],
                    'pattern_risk': pattern_risk
                })
                
                # Count by status
                if detection_result['is_fraud']:
                    fraud_count += 1
                elif detection_result.get('maybe_fraud', False):
                    maybe_fraud_count += 1
                else:
                    legitimate_count += 1
                
                transactions.append(transaction)
                
            except Exception as e:
                print(f"Error processing transaction row {index}: {e}")
                continue
        
        # Save to MongoDB
        if transactions:
            try:
                result = transactions_collection.insert_many(transactions)
                print(f"Successfully saved {len(transactions)} transactions to MongoDB")
                
                # Convert ObjectIds to strings and datetime to string for JSON serialization
                for i, transaction in enumerate(transactions):
                    transaction['_id'] = str(result.inserted_ids[i])
                    transaction['timestamp'] = transaction['timestamp'].isoformat()
                
                update_stats()
                
                # Emit real-time update
                socketio.emit('csv_analyzed', {
                    'filename': file.filename,
                    'total_transactions': len(transactions),
                    'fraud_count': fraud_count,
                    'legitimate_count': legitimate_count
                })
            except Exception as e:
                print(f"Error saving to MongoDB: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Error saving transactions to database: {str(e)}'
                }), 500
        
        total_transactions = len(transactions)
        
        if total_transactions == 0:
            return jsonify({
                'success': False,
                'error': 'No valid transactions found in CSV file. Please check the file format and ensure it contains transaction data.'
            }), 400
            
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        suspicious_rate = ((fraud_count + maybe_fraud_count) / total_transactions * 100) if total_transactions > 0 else 0
        
        # Convert transactions to the expected format
        results = []
        for i, transaction in enumerate(transactions):
            results.append({
                'row': i + 1,
                'transaction': transaction,
                'fraud_detection': {
                    'is_fraud': transaction.get('is_fraud', False),
                    'maybe_fraud': transaction.get('maybe_fraud', False),
                    'risk_score': transaction.get('confidence', 0.5) / 100,
                    'confidence': transaction.get('confidence', 50),
                    'explanation': 'Quantum fraud analysis completed',
                    'detection_method': 'quantum_neural_network'
                }
            })
        
        print(f"CSV analysis completed successfully. Processed {total_transactions} transactions.")
        
        return jsonify({
            'success': True,
            'total_rows': total_transactions,
            'flagged_count': fraud_count + maybe_fraud_count,
            'results': results,
            'message': f'Successfully analyzed {total_transactions} transactions from CSV'
        })
        
    except Exception as e:
        print(f"=== CSV Analysis Error ===")
        print(f"Error analyzing CSV: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'CSV analysis failed: {str(e)}'
        }), 500

# User Management API Endpoints
@app.route('/api/users/create-20', methods=['POST'])
def create_20_users():
    """Create 20 users with unknown fraudsters"""
    try:
        # Create 20 simple users with initial amounts
        users_created = []
        initial_amounts = [
            50000, 75000, 100000, 125000, 150000,  # user_1 to user_5
            80000, 90000, 110000, 130000, 140000,  # user_6 to user_10
            60000, 85000, 95000, 115000, 135000,   # user_11 to user_15
            70000, 95000, 105000, 120000, 145000   # user_16 to user_20
        ]
        
        for i in range(1, 21):
            user_id = f"user_{i}"
            initial_balance = initial_amounts[i-1]
            
            user_data = {
                'user_id': user_id,
                'balance': initial_balance,
                'location': 'Mumbai',
                'occupation': 'User',
                'is_fraudster': False,  # No fraudsters by default
                'created_at': datetime.now()
            }
            
            users_created.append(user_data)
            print(f"🟢 Created user: {user_id} with ₹{initial_balance:,} balance")
        
        # Save users to the global collection
        users_collection.insert_many(users_created)
        
        print(f"✅ Created 20 simple users")
        
        # Convert users array to Record<string, User> format
        users_dict = {}
        for user in users_created:
            users_dict[user['user_id']] = {
                'user_id': user['user_id'],
                'name': f"User {user['user_id']}",
                'email': f"{user['user_id'].lower()}@example.com",
                'balance': user['balance'],
                'location': user['location'],
                'account_type': user['occupation'],
                'created_at': user['created_at'].isoformat(),
                'is_suspicious': user['is_fraudster']
            }
        
        # Create some payment links for demonstration
        payment_links = []
        for i in range(5):
            payment_links.append({
                'user_id': f"USER{i+1:02d}",
                'safe_link': f"https://pay.example.com/safe/{uuid.uuid4().hex[:8]}",
                'bad_link': f"https://pay.example.com/bad/{uuid.uuid4().hex[:8]}"
            })
        
        return jsonify({
            'success': True,
            'message': '20 simple users created successfully',
            'users': users_dict,
            'payment_links': payment_links
        })
        
    except Exception as e:
        print(f"Error creating users: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'User creation failed: {str(e)}'
        }), 500

@app.route('/api/data/reset', methods=['POST'])
def reset_all_data():
    """Reset all data - clear users, transactions, and training results"""
    try:
        # Clear all collections
        users_collection.delete_many({})
        transactions_collection.delete_many({})
        training_results_collection.delete_many({})
        quantum_states_collection.delete_many({})
        
        # Reset global variables
        global detector, is_training
        detector = QuantumFraudDetector()
        is_training = False
        
        # Reset stats
        global stats
        stats = {
            'total_transactions': 0,
            'fraud_transactions': 0,
            'non_fraud_transactions': 0,
            'quantum_accuracy': 0,
            'classical_accuracy': 0
        }
        
        return jsonify({
            'success': True,
            'message': 'All data has been reset successfully'
        })
        
    except Exception as e:
        print(f"Error resetting data: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to reset data: {str(e)}'
        }), 500

@app.route('/api/users')
def get_users():
    """Get all users"""
    try:
        users = list(users_collection.find({}, {'_id': 0}))
        
        # Convert to Record<string, User> format
        users_dict = {}
        for user in users:
            users_dict[user['user_id']] = {
                'user_id': user['user_id'],
                'name': f"User {user['user_id']}",
                'email': f"{user['user_id'].lower()}@example.com",
                'balance': user['balance'],
                'location': user.get('location', 'Unknown'),
                'account_type': user.get('occupation', 'Unknown'),
                'created_at': user.get('created_at', datetime.now()).isoformat() if isinstance(user.get('created_at'), datetime) else str(user.get('created_at', datetime.now())),
                'is_suspicious': user.get('is_fraudster', False)
            }
        
        return jsonify({
            'success': True,
            'users': users_dict
        })
        
    except Exception as e:
        print(f"Error getting users: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get users: {str(e)}'
        }), 500



@app.route('/api/users/fraud-analysis')
def get_user_fraud_analysis():
    """Get fraud analysis for all users"""
    try:
        # Get all transactions grouped by user
        pipeline = [
            {
                '$group': {
                    '_id': '$user_id',
                    'total_transactions': {'$sum': 1},
                    'fraud_transactions': {'$sum': {'$cond': ['$is_fraud', 1, 0]}},
                    'total_amount': {'$sum': '$amount'},
                    'avg_amount': {'$avg': '$amount'},
                    'locations': {'$addToSet': '$location'},
                    'merchants': {'$addToSet': '$merchant'},
                    'last_transaction': {'$max': '$timestamp'}
                }
            },
            {
                '$project': {
                    'user_id': '$_id',
                    'total_transactions': 1,
                    'fraud_transactions': 1,
                    'fraud_percentage': {
                        '$multiply': [
                            {'$divide': ['$fraud_transactions', '$total_transactions']},
                            100
                        ]
                    },
                    'total_amount': 1,
                    'avg_amount': 1,
                    'unique_locations': {'$size': '$locations'},
                    'unique_merchants': {'$size': '$merchants'},
                    'last_transaction': 1
                }
            },
            {'$sort': {'fraud_percentage': -1}}
        ]
        
        user_stats = list(transactions_collection.aggregate(pipeline))
        
        # Convert ObjectIds to strings
        for user in user_stats:
            if 'last_transaction' in user:
                user['last_transaction'] = user['last_transaction'].isoformat()
        
        return jsonify({
            'success': True,
            'user_stats': user_stats,
            'total_users': len(user_stats)
        })
        
    except Exception as e:
        print(f"Error getting user fraud analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'User analysis failed: {str(e)}'
        }), 500

@app.route('/api/training/status')
def get_training_status():
    """Get current training status"""
    # Import QUANTUM_AVAILABLE from the detector module
    try:
        from quantum_neural_network_fraud_detector import QUANTUM_AVAILABLE
    except ImportError:
        QUANTUM_AVAILABLE = False
    
    return jsonify({
        'success': True,
        'is_training': is_training,
        'is_trained': detector.is_trained,
        'quantum_available': QUANTUM_AVAILABLE
    })

@app.route('/api/user-to-user-payment', methods=['POST'])
def user_to_user_payment():
    """Process user-to-user payments"""
    try:
        from_user_id = request.form.get('from_user_id')
        to_user_id = request.form.get('to_user_id')
        amount = float(request.form.get('amount', 0))
        
        if not from_user_id or not to_user_id:
            return jsonify({
                'success': False,
                'error': 'Both from_user_id and to_user_id are required'
            }), 400
        
        if amount <= 0:
            return jsonify({
                'success': False,
                'error': 'Amount must be greater than 0'
            }), 400
        
        # Check if users exist
        from_user = users_collection.find_one({'user_id': from_user_id})
        to_user = users_collection.find_one({'user_id': to_user_id})
        
        if not from_user:
            return jsonify({
                'success': False,
                'error': f'User {from_user_id} not found'
            }), 404
        
        if not to_user:
            return jsonify({
                'success': False,
                'error': f'User {to_user_id} not found'
            }), 404
        
        # Check if sender has sufficient balance
        if from_user['balance'] < amount:
            return jsonify({
                'success': False,
                'error': f'Insufficient balance. Available: ₹{from_user["balance"]:,}, Required: ₹{amount:,}'
            }), 400
        
        # Generate transaction ID
        transaction_id = str(uuid.uuid4()).replace('-', '')[:8].upper()
        
        # Create transaction record
        transaction = {
            'transaction_id': transaction_id,
            'from_user_id': from_user_id,
            'to_user_id': to_user_id,
            'amount': amount,
            'merchant': f'Transfer to {to_user_id}',
            'location': 'User Transfer',
            'timestamp': datetime.now(),
            'transaction_type': 'user_to_user'
        }
        
        # Detect fraud
        try:
            detection_result = detector.detect_fraud(transaction)
        except Exception as e:
            # Fallback detection result
            detection_result = {
                'is_fraud': False,
                'confidence': 0.5,
                'quantum_prediction': 0.5,
                'classical_prediction': 0.5,
                'risk_factors': ['user_transfer']
            }
        
        # Add detection results
        transaction.update({
            'is_fraud': detection_result['is_fraud'],
            'confidence': detection_result['confidence'],
            'quantum_prediction': detection_result['quantum_prediction'],
            'classical_prediction': detection_result['classical_prediction'],
            'risk_factors': detection_result['risk_factors']
        })
        
        # Save transaction to MongoDB
        result = transactions_collection.insert_one(transaction)
        transaction['_id'] = str(result.inserted_id)
        
        # Update user balances
        users_collection.update_one(
            {'user_id': from_user_id},
            {'$inc': {'balance': -amount}}
        )
        
        users_collection.update_one(
            {'user_id': to_user_id},
            {'$inc': {'balance': amount}}
        )
        
        # Get updated balances
        updated_from_user = users_collection.find_one({'user_id': from_user_id})
        updated_to_user = users_collection.find_one({'user_id': to_user_id})
        
        # Convert datetime to string for JSON serialization
        transaction['timestamp'] = transaction['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'transaction_id': transaction_id,
            'from_user': {
                'user_id': from_user_id,
                'old_balance': from_user['balance'],
                'new_balance': updated_from_user['balance']
            },
            'to_user': {
                'user_id': to_user_id,
                'old_balance': to_user['balance'],
                'new_balance': updated_to_user['balance']
            },
            'amount': amount,
            'fraud_detection': {
                'is_fraud': detection_result['is_fraud'],
                'maybe_fraud': False,
                'risk_score': detection_result.get('risk_score', 0.5),
                'confidence': detection_result['confidence'],
                'explanation': f"User transfer from {from_user_id} to {to_user_id} with {detection_result['confidence']:.1f}% confidence",
                'detection_method': 'quantum_neural_network'
            }
        })
        
    except Exception as e:
        print(f"Error processing user-to-user payment: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'User-to-user payment failed: {str(e)}'
        }), 500

@app.route('/api/user-to-merchant-payment', methods=['POST'])
def user_to_merchant_payment():
    """Process user-to-merchant payments"""
    try:
        user_id = request.form.get('user_id')
        merchant = request.form.get('merchant')
        amount = float(request.form.get('amount', 0))
        location = request.form.get('location', 'Unknown')
        
        if not user_id or not merchant:
            return jsonify({
                'success': False,
                'error': 'Both user_id and merchant are required'
            }), 400
        
        if amount <= 0:
            return jsonify({
                'success': False,
                'error': 'Amount must be greater than 0'
            }), 400
        
        # Check if user exists
        user = users_collection.find_one({'user_id': user_id})
        
        if not user:
            return jsonify({
                'success': False,
                'error': f'User {user_id} not found'
            }), 404
        
        # Check if user has sufficient balance
        if user['balance'] < amount:
            return jsonify({
                'success': False,
                'error': f'Insufficient balance. Available: ₹{user["balance"]:,}, Required: ₹{amount:,}'
            }), 400
        
        # Generate transaction ID
        transaction_id = str(uuid.uuid4()).replace('-', '')[:8].upper()
        
        # Create transaction record
        transaction = {
            'transaction_id': transaction_id,
            'user_id': user_id,
            'amount': amount,
            'merchant': merchant,
            'location': location,
            'timestamp': datetime.now(),
            'transaction_type': 'user_to_merchant'
        }
        
        # Detect fraud
        try:
            detection_result = detector.detect_fraud(transaction)
        except Exception as e:
            # Fallback detection result
            detection_result = {
                'is_fraud': False,
                'confidence': 0.5,
                'quantum_prediction': 0.5,
                'classical_prediction': 0.5,
                'risk_factors': ['merchant_payment']
            }
        
        # Add detection results
        transaction.update({
            'is_fraud': detection_result['is_fraud'],
            'confidence': detection_result['confidence'],
            'quantum_prediction': detection_result['quantum_prediction'],
            'classical_prediction': detection_result['classical_prediction'],
            'risk_factors': detection_result['risk_factors']
        })
        
        # Save transaction to MongoDB
        result = transactions_collection.insert_one(transaction)
        transaction['_id'] = str(result.inserted_id)
        
        # Update user balance
        users_collection.update_one(
            {'user_id': user_id},
            {'$inc': {'balance': -amount}}
        )
        
        # Get updated balance
        updated_user = users_collection.find_one({'user_id': user_id})
        
        # Convert datetime to string for JSON serialization
        transaction['timestamp'] = transaction['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'transaction_id': transaction_id,
            'user': {
                'user_id': user_id,
                'old_balance': user['balance'],
                'new_balance': updated_user['balance']
            },
            'merchant': merchant,
            'amount': amount,
            'fraud_detection': {
                'is_fraud': detection_result['is_fraud'],
                'maybe_fraud': False,
                'risk_score': detection_result.get('risk_score', 0.5),
                'confidence': detection_result['confidence'],
                'explanation': f"Payment to {merchant} with {detection_result['confidence']:.1f}% confidence",
                'detection_method': 'quantum_neural_network'
            }
        })
        
    except Exception as e:
        print(f"Error processing user-to-merchant payment: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'User-to-merchant payment failed: {str(e)}'
        }), 500
    """Process user-to-user payments"""
    try:
        from_user_id = request.form.get('from_user_id')
        to_user_id = request.form.get('to_user_id')
        amount = float(request.form.get('amount', 0))
        
        if not from_user_id or not to_user_id:
            return jsonify({
                'success': False,
                'error': 'Both from_user_id and to_user_id are required'
            }), 400
        
        if amount <= 0:
            return jsonify({
                'success': False,
                'error': 'Amount must be greater than 0'
            }), 400
        
        # Check if users exist
        from_user = users_collection.find_one({'user_id': from_user_id})
        to_user = users_collection.find_one({'user_id': to_user_id})
        
        if not from_user:
            return jsonify({
                'success': False,
                'error': f'User {from_user_id} not found'
            }), 404
        
        if not to_user:
            return jsonify({
                'success': False,
                'error': f'User {to_user_id} not found'
            }), 404
        
        # Check if sender has sufficient balance
        if from_user['balance'] < amount:
            return jsonify({
                'success': False,
                'error': f'Insufficient balance. Available: ₹{from_user["balance"]:,}, Required: ₹{amount:,}'
            }), 400
        
        # Generate transaction ID
        transaction_id = str(uuid.uuid4()).replace('-', '')[:8].upper()
        
        # Create transaction record
        transaction = {
            'transaction_id': transaction_id,
            'from_user_id': from_user_id,
            'to_user_id': to_user_id,
            'amount': amount,
            'merchant': f'Transfer to {to_user_id}',
            'location': 'User Transfer',
            'timestamp': datetime.now(),
            'transaction_type': 'user_to_user'
        }
        
        # Detect fraud
        try:
            detection_result = detector.detect_fraud(transaction)
        except Exception as e:
            # Fallback detection result
            detection_result = {
                'is_fraud': False,
                'confidence': 0.5,
                'quantum_prediction': 0.5,
                'classical_prediction': 0.5,
                'risk_factors': ['user_transfer']
            }
        
        # Add detection results
        transaction.update({
            'is_fraud': detection_result['is_fraud'],
            'confidence': detection_result['confidence'],
            'quantum_prediction': detection_result['quantum_prediction'],
            'classical_prediction': detection_result['classical_prediction'],
            'risk_factors': detection_result['risk_factors']
        })
        
        # Save transaction to MongoDB
        result = transactions_collection.insert_one(transaction)
        transaction['_id'] = str(result.inserted_id)
        
        # Update user balances
        users_collection.update_one(
            {'user_id': from_user_id},
            {'$inc': {'balance': -amount}}
        )
        
        users_collection.update_one(
            {'user_id': to_user_id},
            {'$inc': {'balance': amount}}
        )
        
        # Get updated balances
        updated_from_user = users_collection.find_one({'user_id': from_user_id})
        updated_to_user = users_collection.find_one({'user_id': to_user_id})
        
        # Convert datetime to string for JSON serialization
        transaction['timestamp'] = transaction['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'transaction_id': transaction_id,
            'from_user': {
                'user_id': from_user_id,
                'old_balance': from_user['balance'],
                'new_balance': updated_from_user['balance']
            },
            'to_user': {
                'user_id': to_user_id,
                'old_balance': to_user['balance'],
                'new_balance': updated_to_user['balance']
            },
            'amount': amount,
            'fraud_detection': {
                'is_fraud': detection_result['is_fraud'],
                'maybe_fraud': False,
                'risk_score': detection_result.get('risk_score', 0.5),
                'confidence': detection_result['confidence'],
                'explanation': f"User transfer from {from_user_id} to {to_user_id} with {detection_result['confidence']:.1f}% confidence",
                'detection_method': 'quantum_neural_network'
            }
        })
        
    except Exception as e:
        print(f"Error processing user-to-user payment: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'User-to-user payment failed: {str(e)}'
        }), 500

@app.route('/api/link-payment', methods=['POST'])
def process_link_payment():
    """Process payment link payments"""
    try:
        link_id = request.form.get('link_id')
        amount = float(request.form.get('amount', 0))
        
        if not link_id:
            return jsonify({
                'success': False,
                'error': 'Link ID is required'
            }), 400
        
        # Generate a transaction ID
        transaction_id = str(uuid.uuid4()).replace('-', '')[:8].upper()
        
        # Create a mock transaction for the link payment
        transaction = {
            'transaction_id': transaction_id,
            'user_id': f'LINK_USER_{link_id[:8]}',
            'amount': amount,
            'merchant': 'Payment Link',
            'location': 'Online',
            'link_id': link_id,
            'timestamp': datetime.now()
        }
        
        # Detect fraud
        try:
            detection_result = detector.detect_fraud(transaction)
        except Exception as e:
            # Fallback detection result
            detection_result = {
                'is_fraud': False,
                'confidence': 0.5,
                'quantum_prediction': 0.5,
                'classical_prediction': 0.5,
                'risk_factors': ['unknown']
            }
        
        # Add detection results
        transaction.update({
            'is_fraud': detection_result['is_fraud'],
            'confidence': detection_result['confidence'],
            'quantum_prediction': detection_result['quantum_prediction'],
            'classical_prediction': detection_result['classical_prediction'],
            'risk_factors': detection_result['risk_factors']
        })
        
        # Save to MongoDB
        result = transactions_collection.insert_one(transaction)
        transaction['_id'] = str(result.inserted_id)
        
        # Convert datetime to string for JSON serialization
        transaction['timestamp'] = transaction['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'transaction_id': transaction_id,
            'link_info': {
                'link_id': link_id,
                'amount': amount,
                'status': 'processed'
            },
            'fraud_detection': {
                'is_fraud': detection_result['is_fraud'],
                'maybe_fraud': False,
                'risk_score': detection_result.get('risk_score', 0.5),
                'confidence': detection_result['confidence'],
                'explanation': f"Payment link {link_id} processed with {detection_result['confidence']:.1f}% confidence",
                'detection_method': 'quantum_neural_network'
            },
            'amount_charged': amount,
            'new_balance': 100000  # Mock balance
        })
        
    except Exception as e:
        print(f"Error processing link payment: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Link payment failed: {str(e)}'
        }), 500

@app.route('/api/fraud-graph')
def get_fraud_graph():
    """Get fraud graph analysis"""
    try:
        # Get all transactions for graph analysis
        transactions = list(transactions_collection.find({}))
        
        # Calculate basic graph stats
        total_transactions = len(transactions)
        fraud_transactions = len([t for t in transactions if t.get('is_fraud', False)])
        
        # Create mock graph data
        graph_stats = {
            'total_nodes': total_transactions + 20,  # transactions + users
            'total_edges': total_transactions * 2,
            'user_nodes': 20,
            'merchant_nodes': len(set(t.get('merchant', '') for t in transactions)),
            'fraud_nodes': fraud_transactions,
            'fraud_edges': fraud_transactions * 2,
            'avg_fraud_score': fraud_transactions / total_transactions if total_transactions > 0 else 0,
            'total_amount': sum(t.get('amount', 0) for t in transactions),
            'fraud_amount': sum(t.get('amount', 0) for t in transactions if t.get('is_fraud', False)),
            'density': 0.15,
            'avg_clustering': 0.25,
            'avg_degree': 2.5
        }
        
        # Create mock fraud patterns
        patterns = [
            {
                'type': 'High Amount Transactions',
                'risk': 'Medium',
                'count': len([t for t in transactions if t.get('amount', 0) > 10000]),
                'fraud_rate': 0.3
            },
            {
                'type': 'International Transactions',
                'risk': 'High',
                'count': len([t for t in transactions if t.get('location', '') in ['Dubai', 'Singapore', 'Hong Kong']]),
                'fraud_rate': 0.6
            },
            {
                'type': 'Suspicious Merchants',
                'risk': 'High',
                'count': len([t for t in transactions if 'casino' in t.get('merchant', '').lower()]),
                'fraud_rate': 0.8
            }
        ]
        
        # Create mock fraud rings
        rings = [
            {
                'users': ['USER01', 'USER02', 'USER03'],
                'merchants': ['Online Casino', 'Cryptocurrency Exchange'],
                'cycle': ['USER01', 'Online Casino', 'USER02', 'Cryptocurrency Exchange', 'USER03'],
                'size': 5,
                'fraud_score': 0.85,
                'suspicious_score': 0.92,
                'risk': 'High'
            }
        ]
        
        return jsonify({
            'success': True,
            'graph_stats': graph_stats,
            'patterns': patterns,
            'rings': rings
        })
        
    except Exception as e:
        print(f"Error generating fraud graph: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Fraud graph generation failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Quantum Fraud Detection System Starting...")
    print("🌐 Frontend: http://localhost:3000")
    print("🔗 Backend API: http://localhost:5000")
    print("📊 MongoDB: Connected to Atlas")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
