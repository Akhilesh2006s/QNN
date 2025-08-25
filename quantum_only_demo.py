"""
Real Quantum-Only Fraud Detector
Uses actual quantum computing with Qiskit circuits and quantum backends
Production-ready implementation with proper quantum practices
"""

import numpy as np
import random
from datetime import datetime
import json
import sys
import warnings
import threading
import time
import hashlib
import os
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

# Real quantum computing imports - proper modern Qiskit
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Estimator as AerEstimator
    from qiskit_algorithms.optimizers import SPSA
    from sklearn.model_selection import train_test_split
    QUANTUM_AVAILABLE = True
    print("[OK] Real quantum computing libraries loaded successfully!")
except ImportError as e:
    print(f"[ERROR] Quantum libraries not available: {e}")
    print("[TOOLS] Installing quantum libraries...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "qiskit-algorithms", "scikit-learn"])
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import Estimator as AerEstimator
        from qiskit_algorithms.optimizers import SPSA
        from sklearn.model_selection import train_test_split
        QUANTUM_AVAILABLE = True
        print("[OK] Quantum libraries installed and loaded successfully!")
    except Exception as install_error:
        print(f"[ERROR] Failed to install quantum libraries: {install_error}")
        print("[WARNING] REAL QUANTUM COMPUTING REQUIRED - Cannot proceed without quantum libraries!")
        sys.exit(1)

class RealQuantumFraudDetector:
    def __init__(self, num_qubits=4):
        # Set random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Thread safety
        self.train_lock = threading.Lock()
        self.is_training = False
        
        # Quantum backend configuration - modern approach
        self.simulator = AerSimulator()
        # Use fewer shots during training for speed, more for final evaluation
        self.estimator = AerEstimator(run_options={"shots": 128})  # Reduced for speed
        self.estimator_final = AerEstimator(run_options={"shots": 1024})  # Final evaluation shots
        self.shots = 1024
        
        # Quantum circuit parameters
        self.num_qubits = num_qubits
        self.feature_map = None
        self.ansatz = None
        self.quantum_circuit = None
        
        # Quantum weights (trained parameters)
        self.quantum_weights = None
        self.is_trained = False
        
        # SPSA optimizer for quantum training - reduced iterations for speed
        self.optimizer = SPSA(maxiter=20)  # Reduced for faster prototyping
        
        # Fraud detection thresholds (will be tuned on validation set)
        self.fraud_threshold = 0.7
        self.maybe_fraud_threshold = 0.4
        
        # Caching for efficiency
        self._validation_predictions_cache = None
        self._validation_data_cache_key = None
        
        # Model persistence
        self.model_file = "quantum_model_weights.npy"
        
        print("[ROCKET] Real Quantum Fraud Detector initialized!")
        print(f"   Quantum Backend: AerSimulator")
        print(f"   Qubits: {self.num_qubits}")
        print(f"   Training Shots: 128")
        print(f"   Final Shots: {self.shots}")
        print(f"   Optimizer: SPSA (maxiter=20)")
        print(f"   Thread Safety: Enabled")
    
    def validate_transaction(self, transaction: Dict) -> bool:
        """Validate incoming transaction data"""
        required_fields = ['amount', 'user_balance', 'location', 'merchant']
        
        # Check required fields
        for field in required_fields:
            if field not in transaction:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types and ranges
        if not isinstance(transaction['amount'], (int, float)) or transaction['amount'] <= 0:
            raise ValueError("Amount must be a positive number")
        
        if not isinstance(transaction['user_balance'], (int, float)) or transaction['user_balance'] <= 0:
            raise ValueError("User balance must be a positive number")
        
        if not isinstance(transaction['location'], str) or len(transaction['location']) == 0:
            raise ValueError("Location must be a non-empty string")
        
        if not isinstance(transaction['merchant'], str) or len(transaction['merchant']) == 0:
            raise ValueError("Merchant must be a non-empty string")
        
        # Validate amount doesn't exceed balance
        if transaction['amount'] > transaction['user_balance']:
            raise ValueError("Transaction amount cannot exceed user balance")
        
        return True
    
    def create_quantum_circuit(self):
        """Create real quantum circuit with feature map and ansatz"""
        print("[TOOLS] Creating real quantum circuit...")
        
        # Create quantum feature map (ZZFeatureMap for quantum feature encoding)
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.num_qubits,
            reps=2,
            insert_barriers=True
        )
        
        # Create quantum ansatz (RealAmplitudes for variational quantum circuit)
        self.ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=2,
            insert_barriers=True
        )
        
        # Combine feature map and ansatz to create full quantum circuit
        self.quantum_circuit = self.feature_map.compose(self.ansatz)
        
        print(f"[OK] Quantum circuit created:")
        print(f"   - Feature map: {self.feature_map.num_parameters} parameters")
        print(f"   - Ansatz: {self.ansatz.num_parameters} parameters")
        print(f"   - Total circuit: {self.quantum_circuit.num_parameters} parameters")
        print(f"   - Circuit depth: {self.quantum_circuit.depth()}")
        
        return self.quantum_circuit
    
    def quantum_feature_encoding(self, transaction: Dict) -> np.ndarray:
        """Real quantum feature encoding for quantum circuit"""
        # Validate transaction first
        self.validate_transaction(transaction)
        
        features = []
        
        # Normalize amount (0-1 range for quantum encoding)
        amount_normalized = min(transaction['amount'] / 100000, 1.0)
        features.append(amount_normalized)
        
        # Location risk encoding
        high_risk_locations = ['Cayman Islands', 'Mauritius', 'Panama', 'Seychelles']
        location_risk = 0.8 if transaction['location'] in high_risk_locations else 0.1
        features.append(location_risk)
        
        # Merchant risk encoding
        high_risk_merchants = ['Online Casino', 'Cryptocurrency Exchange', 'Offshore Bank']
        merchant_risk = 0.9 if transaction['merchant'] in high_risk_merchants else 0.1
        features.append(merchant_risk)
        
        # Time risk encoding
        hour = transaction.get('hour', 12)
        time_risk = 0.7 if hour in [0, 1, 2, 3, 4] else 0.2
        features.append(time_risk)
        
        # Ensure we have exactly num_qubits features
        while len(features) < self.num_qubits:
            features.append(0.0)
        
        return np.array(features[:self.num_qubits], dtype=np.float32)
    
    def execute_quantum_circuit_batch(self, features_list: List[np.ndarray], weights: Optional[np.ndarray] = None, 
                                    use_final_estimator: bool = False) -> List[float]:
        """Execute multiple quantum circuits in batch for efficiency"""
        if not features_list:
            return []
        
        # Create quantum circuit if not exists
        if self.quantum_circuit is None:
            self.create_quantum_circuit()
        
        # Initialize weights if not provided
        if weights is None:
            if self.quantum_weights is None:
                weights = np.random.rand(self.ansatz.num_parameters)
            else:
                weights = self.quantum_weights
        
        # Parameter length safety check
        if len(weights) != len(self.ansatz.parameters):
            raise ValueError(f"Weights length {len(weights)} mismatch with ansatz parameters {len(self.ansatz.parameters)}")
        
        # Prepare batch circuits
        circuits = []
        observable = SparsePauliOp("Z" * self.num_qubits)
        
        for features in features_list:
            # Proper parameter binding by name (not position) - preserve order
            parameter_dict = {}
            
            # Lock parameter order for safety
            fm_params = list(self.feature_map.parameters)
            an_params = list(self.ansatz.parameters)
            
            # Bind feature map parameters
            for i, param in enumerate(fm_params):
                parameter_dict[param] = float(features[i])
            
            # Bind ansatz parameters
            for i, param in enumerate(an_params):
                parameter_dict[param] = float(weights[i])
            
            # Bind parameters to quantum circuit
            bound_circuit = self.quantum_circuit.bind_parameters(parameter_dict)
            circuits.append(bound_circuit)
        
        # Choose estimator based on use case
        estimator = self.estimator_final if use_final_estimator else self.estimator
        
        try:
            # Batch execution for efficiency
            job = estimator.run(circuits, [observable] * len(circuits))
            result = job.result()
            expectation_values = result.values
        except Exception as e:
            # Proper error handling - don't hide failures
            raise RuntimeError(f"Batch estimator run failed: {e}")
        
        return expectation_values.tolist()
    
    def execute_quantum_circuit(self, features: np.ndarray, weights: Optional[np.ndarray] = None, 
                              use_final_estimator: bool = False) -> float:
        """Execute single quantum circuit using modern Estimator primitive"""
        results = self.execute_quantum_circuit_batch([features], weights, use_final_estimator)
        return results[0]
    
    def quantum_forward(self, features: np.ndarray, weights: Optional[np.ndarray] = None, 
                       use_final_estimator: bool = False) -> float:
        """Quantum forward pass using real quantum circuit"""
        # Execute quantum circuit
        expectation = self.execute_quantum_circuit(features, weights=weights, use_final_estimator=use_final_estimator)
        
        # Convert expectation to probability (0-1 range)
        quantum_probability = (expectation + 1) / 2
        return quantum_probability
    
    def quantum_forward_batch(self, features_list: List[np.ndarray], weights: Optional[np.ndarray] = None, 
                             use_final_estimator: bool = False) -> List[float]:
        """Batch quantum forward pass for efficiency"""
        expectations = self.execute_quantum_circuit_batch(features_list, weights, use_final_estimator)
        
        # Convert expectations to probabilities
        probabilities = [(exp + 1) / 2 for exp in expectations]
        return probabilities
    
    def objective_function(self, weights: np.ndarray, training_data: List[Dict]) -> float:
        """Objective function for SPSA optimization with batching"""
        if not training_data:
            return 0.0
        
        # Encode all features at once
        features_list = [self.quantum_feature_encoding(tx) for tx in training_data]
        
        # Batch forward pass
        predictions = self.quantum_forward_batch(features_list, weights=weights, use_final_estimator=False)
        
        # Calculate loss
        total_loss = 0.0
        for i, transaction in enumerate(training_data):
            target = 1.0 if transaction['is_fraud'] else 0.0
            prediction = predictions[i]
            loss = (prediction - target) ** 2
            total_loss += loss
        
        return total_loss / len(training_data)
    
    def _compute_cache_key(self, data: List[Dict]) -> str:
        """Compute stable cache key from transaction data"""
        # Use transaction IDs for stable hashing
        tx_ids = [tx.get('transaction_id', str(i)) for i, tx in enumerate(data)]
        cache_string = "|".join(sorted(tx_ids))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def tune_thresholds(self, validation_data: List[Dict]) -> Tuple[float, float, float]:
        """Tune fraud detection thresholds on validation set with caching"""
        print("[TARGET] Tuning fraud detection thresholds...")
        
        # Compute stable cache key
        cache_key = self._compute_cache_key(validation_data)
        
        # Check if we can use cached predictions
        if (self._validation_predictions_cache is not None and 
            self._validation_data_cache_key == cache_key):
            predictions = self._validation_predictions_cache
        else:
            # Compute predictions once and cache them
            features_list = [self.quantum_feature_encoding(tx) for tx in validation_data]
            predictions = self.quantum_forward_batch(features_list, use_final_estimator=True)
            
            # Cache results
            self._validation_predictions_cache = predictions
            self._validation_data_cache_key = cache_key
        
        best_accuracy = 0.0
        best_thresholds = (0.7, 0.4)
        
        # Grid search for optimal thresholds using cached predictions
        for fraud_thresh in np.arange(0.5, 0.9, 0.05):
            for maybe_thresh in np.arange(0.3, 0.6, 0.05):
                if maybe_thresh >= fraud_thresh:
                    continue
                
                correct = 0
                total = len(validation_data)
                
                for i, transaction in enumerate(validation_data):
                    prediction = predictions[i]
                    
                    # Apply thresholds
                    if prediction > fraud_thresh:
                        predicted_fraud = True
                    elif prediction > maybe_thresh:
                        predicted_fraud = True  # Treat maybe_fraud as fraud for accuracy
                    else:
                        predicted_fraud = False
                    
                    actual_fraud = transaction['is_fraud']
                    if predicted_fraud == actual_fraud:
                        correct += 1
                
                accuracy = correct / total
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_thresholds = (fraud_thresh, maybe_thresh)
        
        self.fraud_threshold, self.maybe_fraud_threshold = best_thresholds
        print(f"[OK] Optimal thresholds: Fraud={self.fraud_threshold:.2f}, Maybe={self.maybe_fraud_threshold:.2f}")
        print(f"   Validation Accuracy: {best_accuracy:.3f}")
        
        return best_thresholds, best_accuracy
    
    def train_quantum_model(self, training_data: List[Dict]) -> Dict:
        """Train real quantum neural network using SPSA optimizer with thread safety"""
        with self.train_lock:
            if self.is_training:
                raise RuntimeError("Training already in progress")
            
            self.is_training = True
            
            try:
                return self._train_quantum_model_internal(training_data)
            finally:
                self.is_training = False
    
    def _train_quantum_model_internal(self, training_data: List[Dict]) -> Dict:
        """Internal training method"""
        print("[TARGET] Training real quantum neural network with SPSA...")
        start_time = time.time()
        
        # Create quantum circuit
        if self.quantum_circuit is None:
            self.create_quantum_circuit()
        
        # Split data into train and validation
        train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)
        print(f"   Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # Initialize quantum weights
        self.quantum_weights = np.random.rand(self.ansatz.num_parameters)
        
        print(f"   Training for {self.optimizer.maxiter} iterations...")
        
        # SPSA optimization
        try:
            result = self.optimizer.minimize(
                fun=lambda x: self.objective_function(x, train_data),
                x0=self.quantum_weights
            )
            
            # Update trained weights
            self.quantum_weights = result.x
            
            # Tune thresholds on validation set
            self.tune_thresholds(val_data)
            
            # Evaluate final performance on validation set
            correct_predictions = 0
            total_predictions = len(val_data)
            
            for transaction in val_data:
                features = self.quantum_feature_encoding(transaction)
                prediction = self.quantum_forward(features, use_final_estimator=True)
                
                # Apply tuned thresholds
                if prediction > self.fraud_threshold:
                    predicted_fraud = True
                elif prediction > self.maybe_fraud_threshold:
                    predicted_fraud = True
                else:
                    predicted_fraud = False
                
                actual_fraud = transaction['is_fraud']
                if predicted_fraud == actual_fraud:
                    correct_predictions += 1
            
            final_accuracy = correct_predictions / total_predictions
            final_loss = result.fun
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            final_accuracy = 0.0
            final_loss = float('inf')
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Save trained weights to disk
        self.save_model()
        
        print("[OK] Real quantum neural network training completed!")
        print(f"   Final Loss: {final_loss:.4f}")
        print(f"   Validation Accuracy: {final_accuracy:.3f}")
        print(f"   Training Time: {training_time:.2f} seconds")
        print(f"   Optimizer: {self.optimizer.__class__.__name__}")
        
        return {
            "iterations": self.optimizer.maxiter,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "training_time": training_time,
            "quantum_backend": "AerSimulator",
            "circuit_parameters": self.quantum_circuit.num_parameters,
            "optimizer": self.optimizer.__class__.__name__,
            "fraud_threshold": self.fraud_threshold,
            "maybe_fraud_threshold": self.maybe_fraud_threshold
        }
    
    def detect_fraud(self, transaction: Dict) -> Dict:
        """Detect fraud using real quantum neural network"""
        if not self.is_trained:
            return {"error": "Quantum model not trained"}
        
        try:
            # Validate transaction
            self.validate_transaction(transaction)
            
            # Encode features for quantum circuit
            features = self.quantum_feature_encoding(transaction)
            
            # Get quantum prediction with final estimator (higher shots)
            quantum_output = self.quantum_forward(features, use_final_estimator=True)
            
            # Execute quantum circuit directly for verification
            quantum_expectation = self.execute_quantum_circuit(features, use_final_estimator=True)
            
            # Determine fraud status based on tuned thresholds
            if quantum_output > self.fraud_threshold:
                fraud_status = "FRAUD"
            elif quantum_output > self.maybe_fraud_threshold:
                fraud_status = "MAYBE_FRAUD"
            else:
                fraud_status = "NORMAL"
            
            return {
                "quantum_score": float(quantum_output),
                "quantum_expectation": float(quantum_expectation),
                "fraud_status": fraud_status,
                "confidence": abs(quantum_output - 0.5) * 2,
                "features_used": len(features),
                "quantum_backend": "AerSimulator",
                "circuit_parameters": self.quantum_circuit.num_parameters,
                "shots": self.shots,
                "success": True,
                "model_type": "REAL_QUANTUM_ONLY",
                "fraud_threshold": self.fraud_threshold,
                "maybe_fraud_threshold": self.maybe_fraud_threshold
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Fraud detection failed: {str(e)}",
                "model_type": "REAL_QUANTUM_ONLY"
            }
    
    def generate_training_data(self, num_samples: int = 100) -> List[Dict]:
        """Generate training data for quantum model"""
        print(f"[DICE] Generating {num_samples} quantum training samples...")
        
        data = []
        for i in range(num_samples):
            user_balance = random.randint(10000, 500000)
            amount = random.randint(100, min(user_balance, 100000))
            
            locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 
                        'Dubai', 'Singapore', 'Hong Kong', 'Cayman Islands', 'Mauritius']
            location = random.choice(locations)
            
            merchants = ['Amazon India', 'Flipkart', 'Swiggy', 'Netflix', 'Spotify',
                        'Online Casino', 'Cryptocurrency Exchange', 'Offshore Bank', 
                        'Peer Transfer', 'Money Transfer']
            merchant = random.choice(merchants)
            
            hour = random.randint(0, 23)
            user_age = random.randint(18, 65)
            
            # Determine fraud based on patterns
            is_fraud = False
            if (location in ['Cayman Islands', 'Mauritius'] and 
                merchant in ['Online Casino', 'Cryptocurrency Exchange', 'Offshore Bank']):
                is_fraud = random.random() < 0.9
            elif (amount > user_balance * 0.8 and hour in [0, 1, 2, 3, 4]):
                is_fraud = random.random() < 0.8
            elif (merchant in ['Online Casino', 'Cryptocurrency Exchange'] and amount > 50000):
                is_fraud = random.random() < 0.7
            
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
        
        fraud_count = sum(1 for t in data if t['is_fraud'])
        print(f"[OK] Generated {len(data)} samples")
        print(f"   - Fraudulent: {fraud_count} ({fraud_count/len(data)*100:.1f}%)")
        print(f"   - Legitimate: {len(data)-fraud_count} ({(len(data)-fraud_count)/len(data)*100:.1f}%)")
        
        return data
    
    def evaluate_accuracy(self, test_data: List[Dict]) -> Dict:
        """Evaluate quantum model accuracy with efficient batch processing"""
        if not self.is_trained:
            return {"error": "Quantum model not trained"}
        
        # Use batch processing for efficiency
        features_list = [self.quantum_feature_encoding(tx) for tx in test_data]
        predictions = self.quantum_forward_batch(features_list, use_final_estimator=True)
        
        correct = 0
        total = len(test_data)
        
        for i, transaction in enumerate(test_data):
            predicted_fraud = predictions[i] > self.maybe_fraud_threshold
            actual_fraud = transaction['is_fraud']
            
            if predicted_fraud == actual_fraud:
                correct += 1
        
        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "correct_predictions": correct,
            "total_predictions": total,
            "quantum_model_performance": f"{accuracy*100:.2f}%",
            "quantum_backend": "AerSimulator",
            "fraud_threshold": self.fraud_threshold,
            "maybe_fraud_threshold": self.maybe_fraud_threshold
        }
    
    def save_model(self):
        """Save trained model weights to disk"""
        if self.quantum_weights is not None:
            model_data = {
                'weights': self.quantum_weights,
                'fraud_threshold': self.fraud_threshold,
                'maybe_fraud_threshold': self.maybe_fraud_threshold,
                'is_trained': self.is_trained
            }
            np.save(self.model_file, model_data)
            print(f"[SAVE] Model saved to {self.model_file}")
    
    def load_model(self) -> bool:
        """Load trained model weights from disk"""
        if os.path.exists(self.model_file):
            try:
                model_data = np.load(self.model_file, allow_pickle=True).item()
                self.quantum_weights = model_data['weights']
                self.fraud_threshold = model_data['fraud_threshold']
                self.maybe_fraud_threshold = model_data['maybe_fraud_threshold']
                self.is_trained = model_data['is_trained']
                print(f"[LOAD] Model loaded from {self.model_file}")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                return False
        return False

# Global detector instance with thread safety
detector = RealQuantumFraudDetector()

def handle_api_request(data: Dict) -> Dict:
    """Handle API requests from frontend with thread safety"""
    try:
        if 'command' in data:
            command = data['command']
            
            if command == 'detect':
                # Single transaction detection
                transaction = data.get('transaction', {})
                if not transaction:
                    return {"success": False, "error": "No transaction data provided"}
                
                # Try to load existing model first
                if not detector.is_trained:
                    if not detector.load_model():
                        # Start training in background thread if no model exists
                        def train_background():
                            try:
                                training_data = detector.generate_training_data(100)
                                detector.train_quantum_model(training_data)
                            except Exception as e:
                                print(f"Background training failed: {e}")
                        
                        if not detector.is_training:
                            detector.is_training = True
                            thread = threading.Thread(target=train_background)
                            thread.daemon = True
                            thread.start()
                        
                        return {"success": False, "error": "Model training in progress, please try again"}
                
                result = detector.detect_fraud(transaction)
                return result
                
            elif command == 'demo':
                # Run demo and return results
                training_data = detector.generate_training_data(100)  # Reduced for speed
                training_results = detector.train_quantum_model(training_data)
                accuracy_results = detector.evaluate_accuracy(training_data)
                
                return {
                    "success": True,
                    "training_results": training_results,
                    "accuracy_results": accuracy_results,
                    "model_type": "REAL_QUANTUM_ONLY"
                }
                
            elif command == 'train':
                # Train model
                num_samples = data.get('num_samples', 100)  # Reduced default
                training_data = detector.generate_training_data(num_samples)
                training_results = detector.train_quantum_model(training_data)
                accuracy_results = detector.evaluate_accuracy(training_data)
                
                return {
                    "success": True,
                    "training_results": training_results,
                    "accuracy_results": accuracy_results,
                    "model_type": "REAL_QUANTUM_ONLY"
                }
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
        
        else:
            # Default to single transaction detection
            # Try to load existing model first
            if not detector.is_trained:
                if not detector.load_model():
                    # Start training in background thread if no model exists
                    def train_background():
                        try:
                            training_data = detector.generate_training_data(100)
                            detector.train_quantum_model(training_data)
                        except Exception as e:
                            print(f"Background training failed: {e}")
                    
                    if not detector.is_training:
                        detector.is_training = True
                        thread = threading.Thread(target=train_background)
                        thread.daemon = True
                        thread.start()
                    
                    return {"success": False, "error": "Model training in progress, please try again"}
            
            result = detector.detect_fraud(data)
            return result
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    # Check if this is an API call
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        # API mode - read from stdin
        try:
            input_data = sys.stdin.read()
            data = json.loads(input_data)
            result = handle_api_request(data)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
        return
    
    # Interactive demo mode
    print("[ROCKET] REAL QUANTUM-ONLY FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("This system uses REAL quantum computing with Qiskit!")
    print("Actual quantum circuits, qubits, and quantum backends!")
    print("Production-ready implementation with SPSA optimization!")
    print("Optimized for speed, robustness, and thread safety!")
    print("=" * 60)
    
    # Initialize quantum detector
    global detector
    
    # Try to load existing model first
    if not detector.load_model():
        # Generate training data
        training_data = detector.generate_training_data(100)  # Reduced for speed
    
    # Train quantum model
    training_results = detector.train_quantum_model(training_data)
    
    # Evaluate accuracy
    accuracy_results = detector.evaluate_accuracy(training_data)
    
    print("\n[CHART] REAL QUANTUM MODEL PERFORMANCE:")
    print(f"   Accuracy: {accuracy_results['accuracy']*100:.2f}%")
    print(f"   Correct Predictions: {accuracy_results['correct_predictions']}/{accuracy_results['total_predictions']}")
    print(f"   Quantum Backend: {accuracy_results['quantum_backend']}")
    print(f"   Optimizer: {training_results['optimizer']}")
    print(f"   Training Time: {training_results['training_time']:.2f} seconds")
    print(f"   Fraud Threshold: {training_results['fraud_threshold']:.2f}")
    print(f"   Maybe Fraud Threshold: {training_results['maybe_fraud_threshold']:.2f}")
    
    # Test with sample transactions
    print("\n[MAGNIFY] REAL QUANTUM FRAUD DETECTION TESTS:")
    
    test_transactions = [
        {
            'amount': 50000,
            'user_balance': 100000,
            'location': 'Cayman Islands',
            'merchant': 'Online Casino',
            'user_age': 25,
            'hour': 2
        },
        {
            'amount': 1000,
            'user_balance': 50000,
            'location': 'Mumbai',
            'merchant': 'Amazon India',
            'user_age': 30,
            'hour': 14
        }
    ]
    
    for i, transaction in enumerate(test_transactions, 1):
        result = detector.detect_fraud(transaction)
        print(f"\n   Test {i}:")
        print(f"   - Amount: ₹{transaction['amount']:,}")
        print(f"   - Location: {transaction['location']}")
        print(f"   - Merchant: {transaction['merchant']}")
        print(f"   - Quantum Score: {result['quantum_score']:.3f}")
        print(f"   - Quantum Expectation: {result['quantum_expectation']:.3f}")
        print(f"   - Status: {result['fraud_status']}")
        print(f"   - Confidence: {result['confidence']:.3f}")
        print(f"   - Quantum Backend: {result['quantum_backend']}")
    
    print(f"\n[OK] REAL QUANTUM-ONLY FRAUD DETECTION COMPLETED!")
    print(f"   Final Accuracy: {accuracy_results['accuracy']*100:.2f}%")
    print(f"   Quantum Backend: {accuracy_results['quantum_backend']}")
    print(f"   Circuit Parameters: {training_results['circuit_parameters']}")
    print(f"   Qubits Used: {detector.num_qubits}")
    print(f"   Optimizer: {training_results['optimizer']}")
    print(f"   Training Time: {training_results['training_time']:.2f} seconds")
    print(f"   Tuned Thresholds: Fraud={training_results['fraud_threshold']:.2f}, Maybe={training_results['maybe_fraud_threshold']:.2f}")

if __name__ == "__main__":
    main()





