"""
Simple Quantum-Only Fraud Detector
Uses quantum neural networks for fraud detection
"""

import sys
import json
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import quantum libraries
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    QUANTUM_AVAILABLE = True
    print("✅ QUANTUM NEURAL NETWORK libraries loaded successfully!")
except ImportError as e:
    print(f"❌ Quantum libraries not available: {e}")
    QUANTUM_AVAILABLE = False

# Configuration for quantum-only fraud detection
QUANTUM_CONFIG = {
    'fraud_threshold': 0.75,
    'maybe_fraud_threshold': 0.55,
    'quantum_circuit_reps': 1,  # Simplified for compatibility
    'quantum_feature_map_reps': 1,
    'quantum_ansatz_reps': 1,
    'quantum_epochs': 5,  # Reduced for faster training
    'quantum_learning_rate': 0.01,
    'quantum_batch_size': 8,
    'quantum_backend': 'qasm_simulator',
    'quantum_shots': 512,  # Reduced for faster execution
    'risk_weights': {
        'quantum_risk': 0.9,  # Very high weight for quantum predictions
        'traditional_risk': 0.1  # Very low weight for traditional rules
    }
}

class SimpleQuantumFraudDetector:
    def __init__(self, config=None):
        self.config = config or QUANTUM_CONFIG.copy()
        self.quantum_model = None
        self.is_trained = False
        self.quantum_backend = None
        
        if not QUANTUM_AVAILABLE:
            print("🚨 QUANTUM-ONLY MODE REQUIRED - Cannot proceed without quantum libraries!")
            sys.exit(1)
        
        # Initialize quantum backend
        self._initialize_quantum_backend()
        
        print("🚀 Simple Quantum-Only Fraud Detector initialized!")
        print(f"   Quantum Backend: {self.config['quantum_backend']}")
        print(f"   Circuit Repetitions: {self.config['quantum_circuit_reps']}")
        print(f"   Training Epochs: {self.config['quantum_epochs']}")
        print(f"   Quantum Shots: {self.config['quantum_shots']}")
    
    def _initialize_quantum_backend(self):
        """Initialize quantum backend"""
        try:
            self.quantum_backend = Aer.get_backend(self.config['quantum_backend'])
            print(f"✅ Quantum backend '{self.config['quantum_backend']}' initialized")
        except Exception as e:
            print(f"❌ Error initializing quantum backend: {e}")
            # Try alternative backends
            alternative_backends = ['qasm_simulator', 'aer_simulator', 'statevector_simulator']
            for backend_name in alternative_backends:
                try:
                    self.quantum_backend = Aer.get_backend(backend_name)
                    print(f"✅ Using alternative quantum backend: {backend_name}")
                    self.config['quantum_backend'] = backend_name
                    break
                except:
                    continue
            else:
                raise Exception("No quantum backend available!")
    
    def extract_quantum_features(self, transaction):
        """Extract features optimized for quantum neural networks"""
        amount = transaction.get('amount', 0)
        user_balance = transaction.get('user_balance', 100000)
        location = transaction.get('location', 'Unknown')
        merchant = transaction.get('merchant', 'Unknown')
        hour = transaction.get('hour', 12)
        user_age = transaction.get('user_age', 30)
        
        # Quantum-optimized features (simplified to 4 features for compatibility)
        features = []
        
        # Feature 1: Normalized amount ratio (0-1)
        amount_ratio = min(amount / user_balance if user_balance > 0 else 0, 1.0)
        features.append(amount_ratio)
        
        # Feature 2: Time-based risk (0-1)
        time_risk = 0.0
        if hour in [0, 1, 2, 3, 4]:  # Very late night
            time_risk = 1.0
        elif hour in [5, 6, 7, 8]:  # Early morning
            time_risk = 0.7
        elif hour in [22, 23]:  # Late evening
            time_risk = 0.5
        features.append(time_risk)
        
        # Feature 3: Location risk (0-1)
        high_risk_locations = ['Cayman Islands', 'Mauritius', 'Panama', 'Seychelles']
        medium_risk_locations = ['Dubai', 'Singapore', 'Hong Kong']
        if location in high_risk_locations:
            location_risk = 1.0
        elif location in medium_risk_locations:
            location_risk = 0.6
        else:
            location_risk = 0.1
        features.append(location_risk)
        
        # Feature 4: Merchant risk (0-1)
        high_risk_merchants = ['Online Casino', 'Cryptocurrency Exchange', 'Offshore Bank']
        medium_risk_merchants = ['Gambling', 'Adult Content', 'Peer Transfer']
        if merchant in high_risk_merchants:
            merchant_risk = 1.0
        elif merchant in medium_risk_merchants:
            merchant_risk = 0.7
        else:
            merchant_risk = 0.1
        features.append(merchant_risk)
        
        return np.array(features, dtype=np.float32)
    
    def create_quantum_circuit(self, num_features):
        """Create quantum circuit for fraud detection"""
        print(f"🔧 Creating quantum circuit with {num_features} features...")
        
        # Create feature map
        feature_map = ZZFeatureMap(
            feature_dimension=num_features,
            reps=self.config['quantum_feature_map_reps']
        )
        
        # Create ansatz
        ansatz = RealAmplitudes(
            num_qubits=num_features,
            reps=self.config['quantum_ansatz_reps']
        )
        
        # Combine feature map and ansatz
        quantum_circuit = feature_map.compose(ansatz)
        
        print(f"✅ Quantum circuit created:")
        print(f"   - Feature map: {feature_map.num_parameters} parameters")
        print(f"   - Ansatz: {ansatz.num_parameters} parameters")
        print(f"   - Total circuit: {quantum_circuit.num_parameters} parameters")
        
        return quantum_circuit, feature_map, ansatz
    
    def create_quantum_neural_network(self, num_features):
        """Create quantum neural network"""
        print("🧠 Creating Quantum Neural Network...")
        
        # Create quantum circuit
        quantum_circuit, feature_map, ansatz = self.create_quantum_circuit(num_features)
        
        # Create QNN
        qnn = EstimatorQNN(
            circuit=quantum_circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.quantum_backend
        )
        
        # Create PyTorch connector
        self.quantum_model = TorchConnector(qnn)
        
        print("✅ Quantum Neural Network created successfully!")
        return self.quantum_model
    
    def generate_quantum_training_data(self, num_samples=100):
        """Generate training data specifically for quantum neural networks"""
        print(f"🎲 Generating {num_samples} quantum training samples...")
        
        data = []
        
        for i in range(num_samples):
            # Generate realistic transaction data
            user_balance = random.randint(10000, 500000)
            amount = random.randint(100, min(user_balance, 100000))
            
            # Locations with quantum-optimized risk levels
            locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 
                        'Dubai', 'Singapore', 'Hong Kong', 'Cayman Islands', 'Mauritius']
            location = random.choice(locations)
            
            # Merchants with quantum-optimized risk levels
            merchants = ['Amazon India', 'Flipkart', 'Swiggy', 'Netflix', 'Spotify',
                        'Online Casino', 'Cryptocurrency Exchange', 'Offshore Bank', 
                        'Peer Transfer', 'Money Transfer']
            merchant = random.choice(merchants)
            
            hour = random.randint(0, 23)
            user_age = random.randint(18, 65)
            
            # Determine fraud based on quantum-optimized patterns
            is_fraud = False
            
            # High-risk quantum patterns
            if (location in ['Cayman Islands', 'Mauritius'] and 
                merchant in ['Online Casino', 'Cryptocurrency Exchange', 'Offshore Bank']):
                is_fraud = random.random() < 0.9  # 90% chance
            
            elif (amount > user_balance * 0.8 and hour in [0, 1, 2, 3, 4]):
                is_fraud = random.random() < 0.8  # 80% chance
            
            elif (merchant in ['Online Casino', 'Cryptocurrency Exchange'] and amount > 50000):
                is_fraud = random.random() < 0.7  # 70% chance
            
            # Low-risk quantum patterns
            elif (location in ['Mumbai', 'Delhi', 'Bangalore'] and 
                  merchant in ['Amazon India', 'Flipkart', 'Swiggy'] and
                  amount < user_balance * 0.3 and
                  hour in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
                is_fraud = random.random() < 0.02  # 2% chance
            
            # Add quantum noise
            if random.random() < 0.05:  # 5% random quantum noise
                is_fraud = not is_fraud
            
            transaction = {
                'transaction_id': f'QTN{i+1:06d}',
                'user_id': f'quantum_user_{random.randint(1, 100)}',
                'amount': amount,
                'user_balance': user_balance,
                'location': location,
                'merchant': merchant,
                'user_age': user_age,
                'hour': hour,
                'is_fraud': is_fraud
            }
            
            data.append(transaction)
        
        print(f"✅ Generated {len(data)} quantum training samples")
        fraud_count = sum(1 for t in data if t['is_fraud'])
        print(f"   - Fraudulent: {fraud_count} ({fraud_count/len(data)*100:.1f}%)")
        print(f"   - Legitimate: {len(data)-fraud_count} ({(len(data)-fraud_count)/len(data)*100:.1f}%)")
        
        return data
    
    def prepare_quantum_training_data(self, transactions=None):
        """Prepare data for quantum neural network training"""
        if transactions is None:
            transactions = self.generate_quantum_training_data()
        
        print("🔧 Preparing quantum training data...")
        
        # Extract features and labels
        features = []
        labels = []
        
        for transaction in transactions:
            feature_vector = self.extract_quantum_features(transaction)
            features.append(feature_vector)
            labels.append(transaction['is_fraud'])
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"✅ Prepared {len(X)} samples with {X.shape[1]} quantum features")
        print(f"   - Feature range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   - Fraud rate: {np.mean(y)*100:.1f}%")
        
        return X, y
    
    def train_quantum_model(self):
        """Train the quantum neural network"""
        print("🚀 Starting Quantum Neural Network Training...")
        
        # Prepare data
        X, y = self.prepare_quantum_training_data()
        
        # Create quantum neural network
        self.create_quantum_neural_network(X.shape[1])
        
        # Convert to PyTorch tensors
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Training setup
        optimizer = optim.Adam(self.quantum_model.parameters(), lr=self.config['quantum_learning_rate'])
        criterion = nn.BCELoss()
        
        # Create data loader
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config['quantum_batch_size'], shuffle=True)
        
        print(f"🎯 Training quantum model for {self.config['quantum_epochs']} epochs...")
        
        # Training loop
        for epoch in range(self.config['quantum_epochs']):
            self.quantum_model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass through quantum model
                output = self.quantum_model(batch_X).squeeze()
                
                # Calculate loss
                loss = criterion(output, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (output > 0.5).float()
                correct_predictions += (predictions == batch_y).sum().item()
                total_predictions += batch_y.size(0)
            
            # Calculate epoch statistics
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_predictions
            
            print(f"   Epoch {epoch+1}/{self.config['quantum_epochs']}: "
                  f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.3f}")
        
        print(f"✅ Quantum training completed!")
        print(f"   - Final Accuracy: {epoch_accuracy:.3f}")
        print(f"   - Quantum Backend: {self.config['quantum_backend']}")
        print(f"   - Circuit Parameters: {self.quantum_model.qnn.circuit.num_parameters}")
        
        self.is_trained = True
        return {
            'quantum_accuracy': epoch_accuracy,
            'quantum_backend': self.config['quantum_backend'],
            'circuit_parameters': self.quantum_model.qnn.circuit.num_parameters,
            'training_epochs': self.config['quantum_epochs']
        }
    
    def detect_fraud_quantum(self, transaction):
        """Detect fraud using quantum neural network only"""
        if not self.is_trained:
            raise Exception("Quantum model not trained! Please train the model first.")
        
        # Extract quantum features
        features = self.extract_quantum_features(transaction)
        
        # Convert to tensor
        import torch
        features_tensor = torch.FloatTensor([features])
        
        # Get quantum prediction
        self.quantum_model.eval()
        with torch.no_grad():
            quantum_output = self.quantum_model(features_tensor).squeeze().item()
        
        # Calculate traditional risk factors (very low weight)
        traditional_risk = self._calculate_traditional_risk(transaction)
        
        # Combine quantum and traditional risk (heavily weighted toward quantum)
        quantum_weight = self.config['risk_weights']['quantum_risk']
        traditional_weight = self.config['risk_weights']['traditional_risk']
        
        combined_risk = (quantum_output * quantum_weight + 
                        traditional_risk * traditional_weight)
        
        # Determine fraud status
        is_fraud = combined_risk > self.config['fraud_threshold']
        maybe_fraud = combined_risk > self.config['maybe_fraud_threshold'] and not is_fraud
        
        # Calculate confidence
        confidence = combined_risk * 100
        
        # Generate quantum-specific explanation
        explanation = self._generate_quantum_explanation(transaction, quantum_output, combined_risk)
        
        return {
            'is_fraud': bool(is_fraud),
            'maybe_fraud': bool(maybe_fraud),
            'fraud_status': 'FRAUD' if is_fraud else 'MAYBE_FRAUD' if maybe_fraud else 'LEGITIMATE',
            'confidence': float(confidence),
            'quantum_prediction': float(quantum_output),
            'traditional_risk': float(traditional_risk),
            'combined_risk': float(combined_risk),
            'quantum_backend': self.config['quantum_backend'],
            'explanation': explanation,
            'quantum_features': features.tolist()
        }
    
    def _calculate_traditional_risk(self, transaction):
        """Calculate traditional risk factors (very low weight in quantum-only mode)"""
        amount = transaction.get('amount', 0)
        user_balance = transaction.get('user_balance', 100000)
        location = transaction.get('location', 'Unknown')
        merchant = transaction.get('merchant', 'Unknown')
        hour = transaction.get('hour', 12)
        
        # Simple traditional risk calculation
        amount_ratio = amount / user_balance if user_balance > 0 else 0
        
        location_risk = 0.1
        if location in ['Cayman Islands', 'Mauritius']:
            location_risk = 0.8
        elif location in ['Dubai', 'Singapore']:
            location_risk = 0.4
        
        merchant_risk = 0.1
        if merchant in ['Online Casino', 'Cryptocurrency Exchange']:
            merchant_risk = 0.8
        elif merchant in ['Gambling', 'Peer Transfer']:
            merchant_risk = 0.5
        
        time_risk = 0.6 if hour in [0, 1, 2, 3, 4] else 0.0
        
        traditional_risk = (amount_ratio * 0.3 + location_risk * 0.3 + 
                           merchant_risk * 0.3 + time_risk * 0.1)
        
        return min(traditional_risk, 1.0)
    
    def _generate_quantum_explanation(self, transaction, quantum_output, combined_risk):
        """Generate quantum-specific explanation"""
        amount = transaction.get('amount', 0)
        location = transaction.get('location', 'Unknown')
        merchant = transaction.get('merchant', 'Unknown')
        
        explanation_parts = []
        
        if combined_risk > self.config['fraud_threshold']:
            explanation_parts.append("🚨 QUANTUM FRAUD DETECTED")
            explanation_parts.append(f"Quantum prediction: {quantum_output:.3f}")
        elif combined_risk > self.config['maybe_fraud_threshold']:
            explanation_parts.append("⚠️ QUANTUM SUSPICIOUS")
            explanation_parts.append(f"Quantum prediction: {quantum_output:.3f}")
        else:
            explanation_parts.append("✅ QUANTUM LEGITIMATE")
            explanation_parts.append(f"Quantum prediction: {quantum_output:.3f}")
        
        explanation_parts.append(f"Backend: {self.config['quantum_backend']}")
        explanation_parts.append(f"Amount: ₹{amount:,.0f}")
        explanation_parts.append(f"Location: {location}")
        explanation_parts.append(f"Merchant: {merchant}")
        
        return " | ".join(explanation_parts)
    
    def quantum_demo(self):
        """Demonstrate quantum-only fraud detection"""
        print("🚀 QUANTUM-ONLY FRAUD DETECTION DEMO")
        print("=" * 60)
        
        # Train the quantum model
        print("🎯 Training quantum neural network...")
        training_results = self.train_quantum_model()
        
        print(f"\n✅ Training completed!")
        print(f"   Quantum Accuracy: {training_results['quantum_accuracy']:.3f}")
        print(f"   Quantum Backend: {training_results['quantum_backend']}")
        print(f"   Circuit Parameters: {training_results['circuit_parameters']}")
        
        # Test with various transactions
        test_transactions = [
            {
                'transaction_id': 'QTN_TEST_001',
                'user_id': 'quantum_user_1',
                'amount': 50000,
                'user_balance': 100000,
                'location': 'Mumbai',
                'merchant': 'Amazon India',
                'user_age': 30,
                'hour': 14
            },
            {
                'transaction_id': 'QTN_TEST_002',
                'user_id': 'quantum_user_2',
                'amount': 100000,
                'user_balance': 200000,
                'location': 'Cayman Islands',
                'merchant': 'Offshore Bank',
                'user_age': 25,
                'hour': 3
            },
            {
                'transaction_id': 'QTN_TEST_003',
                'user_id': 'quantum_user_3',
                'amount': 75000,
                'user_balance': 150000,
                'location': 'Dubai',
                'merchant': 'Online Casino',
                'user_age': 28,
                'hour': 23
            }
        ]
        
        print(f"\n🧪 Testing quantum fraud detection...")
        for i, transaction in enumerate(test_transactions, 1):
            result = self.detect_fraud_quantum(transaction)
            print(f"\n   Test {i}: {transaction['location']} → {transaction['merchant']}")
            print(f"   Amount: ₹{transaction['amount']:,.0f}")
            print(f"   Quantum Prediction: {result['quantum_prediction']:.3f}")
            print(f"   Status: {result['fraud_status']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print(f"   Explanation: {result['explanation']}")
        
        print(f"\n🎉 Quantum-only fraud detection demo completed!")
        print(f"   - All predictions made using quantum neural networks")
        print(f"   - No classical machine learning models used")
        print(f"   - Quantum backend: {self.config['quantum_backend']}")
        print(f"   - Circuit complexity: {training_results['circuit_parameters']} parameters")

def main():
    print("🚀 QUANTUM-ONLY FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("This system uses ONLY quantum neural networks for fraud detection")
    print("No classical machine learning models are used!")
    print("=" * 60)
    
    # Create quantum-only detector
    detector = SimpleQuantumFraudDetector()
    
    # Run quantum demo
    detector.quantum_demo()

if __name__ == "__main__":
    main()





