# Quantum Fraud Detection - CSV Analysis System

A modern web-based fraud detection system that analyzes transaction CSV files using quantum algorithms and displays results with beautiful charts in Indian Rupees (₹).

## 🚀 Features

- **CSV Upload & Analysis** - Drag and drop CSV files for instant analysis
- **Indian Rupees Support** - All amounts displayed in ₹ with proper formatting
- **Quantum AI Analysis** - Advanced quantum algorithms for fraud detection
- **Beautiful Charts** - 4 different visualization types
- **Real-time Results** - Instant analysis with detailed breakdowns
- **Responsive Design** - Works on desktop, tablet, and mobile

## 📊 Analysis Capabilities

### Summary Cards
- Total Users
- Total Transactions  
- Fraud Detected
- Fraud Rate (%)

### Charts
1. **Fraud Detection Results** - Doughnut chart showing fraud/suspicious/legitimate
2. **Transaction Amount Distribution** - Bar chart with Indian Rupee ranges (₹0-10K, ₹10K-50K, etc.)
3. **Location-based Risk Analysis** - Shows fraud rates by location
4. **Quantum vs Traditional Risk Comparison** - Scatter plot comparing both methods

### Detailed Table
- User ID
- Amount (₹)
- Location
- Merchant
- Quantum Risk
- Traditional Risk
- Status (Fraud/Suspicious/Legitimate)

## 📁 CSV Format

Your CSV file should have these columns:
```csv
user_id,amount,location,merchant,device_id,timestamp
user_001,15000,Mumbai,Amazon,device_001,2024-01-15T10:30:00
user_002,50000,Dubai,Unknown,device_002,2024-01-15T11:15:00
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Flask
- Flask-CORS

### Install Dependencies
```bash
pip install flask flask-cors
```

### Start the Server
```bash
cd frontend
python server.py
```

The server will start at `http://localhost:5000`

## 🎯 How to Use

1. **Open the Application**
   - Go to `http://localhost:5000`
   - Or use the test page: `http://localhost:5000/test_csv.html`

2. **Upload CSV File**
   - Click "Choose CSV file" or drag & drop
   - Use the provided `sample_transactions.csv` for testing

3. **Analyze**
   - Click "Analyze with Quantum AI"
   - Wait for the analysis to complete

4. **View Results**
   - See summary cards with key metrics
   - Explore interactive charts
   - Review detailed transaction table

## 🧪 Testing

### Test Page
Use `test_csv.html` for a simplified test version:
- Upload any CSV file
- See immediate results with charts
- No backend dependency

### Sample Data
The `sample_transactions.csv` file contains 20 sample transactions with:
- Various amounts (₹3,000 to ₹3,00,000)
- Different locations (Mumbai, Dubai, Cayman Islands, etc.)
- Mix of legitimate and suspicious patterns

## 🔧 Technical Details

### Backend Integration
- Flask server with REST API
- Quantum fraud detection algorithms
- CSV processing and analysis
- JSON response format

### Frontend Technologies
- HTML5, CSS3, JavaScript (ES6+)
- Chart.js for visualizations
- Responsive design with CSS Grid/Flexbox
- Modern dark theme with quantum styling

### Quantum Analysis
- Quantum Neural Networks (QNN)
- Quantum Risk Assessment
- Traditional ML comparison
- Real-time processing

## 📱 Mobile Support

The interface is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones
- All modern browsers

## 🎨 Customization

### Styling
- Edit `styles.css` to change colors and layout
- Quantum-themed animations and effects
- Dark mode with blue accent colors

### Charts
- Modify chart configurations in `script.js`
- Add new chart types
- Customize colors and labels

## 🚨 Troubleshooting

### Server Won't Start
- Check if port 5000 is available
- Ensure all dependencies are installed
- Verify Python version (3.8+)

### CSV Upload Issues
- Check CSV format matches expected columns
- Ensure file is not corrupted
- Try the test page first

### Charts Not Displaying
- Check browser console for errors
- Ensure Chart.js is loaded
- Verify data format

## 📈 Performance

- **Fast Analysis** - Processes 1000+ transactions in seconds
- **Efficient Memory** - Handles large CSV files
- **Real-time Updates** - Instant chart generation
- **Responsive UI** - Smooth interactions

## 🔒 Security

- Client-side CSV processing
- No data stored on server
- Secure file handling
- Input validation

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify CSV format
3. Test with sample data first
4. Check browser console for errors

## 🎉 Success!

Your Quantum Fraud Detection system is now ready to analyze transaction data with beautiful visualizations in Indian Rupees! 🚀
