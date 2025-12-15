# Construction AI Predictor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> AI-powered predictive feedback tool for construction project management, delivering accurate cost estimates, timeline forecasting, and risk assessment.

## 🎯 Overview

Construction AI Predictor leverages machine learning algorithms to provide data-driven insights for construction projects. By analyzing historical project data, material costs, labor metrics, and environmental factors, the system generates reliable predictions to support project planning and decision-making.

### Key Features

- **Cost Prediction**: Estimate total project costs with 85%+ accuracy based on project specifications
- **Timeline Forecasting**: Predict project completion dates accounting for delays and dependencies
- **Risk Assessment**: Identify potential project risks using historical patterns
- **Resource Optimization**: Recommend optimal resource allocation strategies
- **RESTful API**: Production-ready API for seamless integration
- **Real-time Analytics**: Dashboard for monitoring predictions and model performance

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/raybran17/construction-ai-predictor.git
cd construction-ai-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Basic Usage

```python
from core.predictor import ConstructionPredictor

# Initialize predictor
predictor = ConstructionPredictor()

# Load trained model
predictor.load_model('models/trained/cost_predictor_v1.pkl')

# Make prediction
project_data = {
    'project_size_sqft': 5000,
    'location': 'New York',
    'building_type': 'commercial',
    'materials': ['steel', 'concrete', 'glass'],
    'num_floors': 3
}

prediction = predictor.predict_cost(project_data)
print(f"Estimated Cost: ${prediction['cost']:,.2f}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Running the API

```bash
# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## 📁 Project Structure

```
construction-ai-predictor/
├── app/                    # FastAPI application
│   ├── api/               # API routes and endpoints
│   ├── schemas/           # Pydantic models
│   └── main.py            # Application entry point
├── core/                   # Core business logic
│   ├── predictor.py       # Main prediction engine
│   ├── preprocessor.py    # Data preprocessing pipeline
│   └── validators.py      # Input validation
├── models/                 # Machine learning models
│   ├── trained/           # Production models
│   ├── experiments/       # Experimental models
│   └── README.md          # Model documentation
├── modules/                # Utility modules
│   ├── data_loader.py     # Data loading utilities
│   ├── feature_engineer.py # Feature engineering
│   └── metrics.py         # Performance metrics
├── data/                   # Data directory
│   ├── raw/               # Original datasets
│   ├── processed/         # Cleaned datasets
│   └── sample/            # Sample data for testing
├── notebooks/              # Jupyter notebooks
│   ├── exploration.ipynb  # Data exploration
│   └── model_training.ipynb # Model development
├── tests/                  # Test suite
│   ├── test_core.py       # Core functionality tests
│   ├── test_api.py        # API endpoint tests
│   └── test_models.py     # Model validation tests
├── docs/                   # Additional documentation
├── .devcontainer/         # Development container config
├── .github/               # GitHub Actions workflows
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker container definition
└── README.md             # This file
```

## 🤖 Models

This project implements multiple ML models optimized for different prediction tasks:

- **Cost Predictor**: XGBoost ensemble model (MAPE: 8.3%)
- **Timeline Forecaster**: LSTM neural network (MAE: 4.2 days)
- **Risk Classifier**: Random Forest (F1-Score: 0.89)

See [models/README.md](models/README.md) for detailed model documentation.

## 📊 API Endpoints

### Cost Prediction
```bash
POST /api/v1/predict/cost
Content-Type: application/json

{
  "project_size_sqft": 5000,
  "location": "New York",
  "building_type": "commercial",
  "materials": ["steel", "concrete"],
  "num_floors": 3
}
```

### Timeline Forecast
```bash
POST /api/v1/predict/timeline
```

### Risk Assessment
```bash
POST /api/v1/assess/risk
```

Full API documentation available at `/docs` when server is running.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t construction-ai-predictor .

# Run container
docker run -p 8000:8000 construction-ai-predictor

# Using docker-compose
docker-compose up -d
```

## 📈 Performance Metrics

| Model | Metric | Score |
|-------|--------|-------|
| Cost Predictor | MAPE | 8.3% |
| Cost Predictor | R² | 0.94 |
| Timeline Forecaster | MAE | 4.2 days |
| Timeline Forecaster | RMSE | 6.8 days |
| Risk Classifier | Accuracy | 91.2% |
| Risk Classifier | F1-Score | 0.89 |

## 🛠️ Technology Stack

- **ML/AI**: scikit-learn, XGBoost, TensorFlow
- **API**: FastAPI, Pydantic
- **Data**: Pandas, NumPy
- **Testing**: pytest, pytest-cov
- **DevOps**: Docker, GitHub Actions
- **Code Quality**: Black, Flake8, mypy

## 📝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Raymond Brandon**
- GitHub: [@raybran17](https://github.com/raybran17)
- LinkedIn:[https://www.linkedin.com/in/brandon-wu-572751303/]

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- Create an issue in this repository
- Email: BrandonW123450@gmail.com

---

**Note**: This project is under active development. Features and APIs may change between versions.
