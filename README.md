# LSTM Text Prediction System

A production-ready text prediction system powered by bidirectional LSTM neural networks. The system collects Wikipedia articles on AI/ML topics, trains a deep learning model with GPU acceleration, and exposes prediction capabilities through a RESTful API built with FastAPI.

## Features

- **Automated Data Pipeline**: Wikipedia article collection, preprocessing, tokenization, and sequence generation
- **GPU-Accelerated Training**: NVIDIA GPU support with automatic CPU fallback
- **Bidirectional LSTM Architecture**: 256-dim embeddings, 512-unit bidirectional LSTM, 256-unit unidirectional LSTM
- **Multiple Prediction Modes**:
  - Single next-word prediction with temperature sampling
  - Top-k predictions with probabilities
  - Batch prediction processing
  - Text completion with stop words
- **Production-Ready API**: FastAPI with Swagger docs, CORS support, health monitoring, metrics tracking
- **Type-Safe Codebase**: Full type hints, comprehensive docstrings, mypy strict mode compliance
- **Beautiful Console Output**: Rich library for progress bars, tables, and formatted output

## Technology Stack

- **ML Framework**: TensorFlow 2.x with GPU support
- **API Framework**: FastAPI with Uvicorn ASGI server
- **Data Source**: Wikipedia API
- **Validation**: Pydantic models
- **Testing**: pytest with async support
- **Console UI**: Rich library
- **Language**: Python 3.10+

## Project Structure

```
lstm-text-prediction/
├── src/
│   ├── data/                    # Data pipeline modules
│   │   ├── collector.py         # Wikipedia data collection
│   │   ├── preprocessor.py      # Text cleaning and preprocessing
│   │   ├── tokenizer.py         # Vocabulary building and tokenization
│   │   └── sequence_generator.py # Training sequence creation
│   ├── model/                   # Model modules
│   │   ├── gpu_manager.py       # GPU detection and configuration
│   │   ├── lstm_model.py        # LSTM architecture definition
│   │   ├── trainer.py           # Model training logic
│   │   └── predictor.py         # Prediction engine
│   ├── api/                     # API modules
│   │   ├── app.py               # FastAPI application
│   │   ├── models.py            # Pydantic request/response models
│   │   ├── endpoints/           # API endpoint routers
│   │   └── middleware/          # Custom middleware
│   └── utils/                   # Utility modules
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging configuration
├── scripts/
│   ├── run_training.py          # Training pipeline script
│   └── run_api.py               # API server script
├── tests/                       # Test suite
├── data/
│   ├── raw/                     # Raw Wikipedia articles
│   └── processed/               # Processed training data
├── models/                      # Saved models and tokenizers
├── docs/                        # Documentation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA support for GPU acceleration

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd lstm-text-prediction
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: GPU Setup (Optional but Recommended)

For GPU acceleration on Windows with NVIDIA hardware:

1. **Install NVIDIA GPU Drivers**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Install the latest driver for your GPU

2. **Install CUDA Toolkit**
   - Download CUDA Toolkit 11.8 from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation wizard

3. **Install cuDNN**
   - Download cuDNN 8.6 from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA installation directory

4. **Verify GPU Setup**
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

If no GPU is available, the system will automatically fall back to CPU training.

## Usage

### Training the Model

Run the complete training pipeline to collect data, preprocess text, and train the LSTM model:

```bash
python scripts/run_training.py
```

This script will:
1. Collect 20 Wikipedia articles on AI/ML topics
2. Preprocess and clean the text data
3. Build a vocabulary of 10,000 most frequent words
4. Generate training sequences with sliding window
5. Configure GPU acceleration (if available)
6. Train the bidirectional LSTM model for 50 epochs
7. Save the model architecture, weights, and tokenizer

**Training Time**: 
- With GPU: ~30-60 minutes
- With CPU: ~2-4 hours

### Starting the API Server

After training completes, start the API server:

```bash
# Production mode
python scripts/run_api.py

# Development mode with auto-reload
python scripts/run_api.py --reload

# Custom port
python scripts/run_api.py --port 8080
```

The API will be available at:
- **API Root**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health and Information

#### GET /
Welcome message and API version.

```bash
curl http://localhost:8000/
```

#### GET /health
Health check with GPU status and uptime.

```bash
curl http://localhost:8000/health
```

#### GET /metrics
API usage statistics and performance metrics.

```bash
curl http://localhost:8000/metrics
```

### Prediction Endpoints

#### POST /predict
Predict the next word given input text.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "machine learning is a field of", "temperature": 1.0}'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "machine learning is a field of", "temperature": 1.0}
)
print(response.json())
# Output: {"prediction": "artificial", "input_text": "machine learning is a field of"}
```

#### POST /predict/top-k
Get top-k predictions with probabilities.

```bash
curl -X POST http://localhost:8000/predict/top-k \
  -H "Content-Type: application/json" \
  -d '{"text": "deep learning uses neural", "temperature": 1.0, "k": 5}'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/top-k",
    json={"text": "deep learning uses neural", "temperature": 1.0, "k": 5}
)
print(response.json())
# Output: {
#   "predictions": [
#     {"word": "networks", "probability": 0.85},
#     {"word": "network", "probability": 0.10},
#     {"word": "architectures", "probability": 0.03},
#     ...
#   ],
#   "input_text": "deep learning uses neural"
# }
```

#### POST /predict/batch
Process multiple predictions in a single request.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["machine learning", "deep learning", "neural network"], "temperature": 1.0}'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": ["machine learning", "deep learning", "neural network"],
        "temperature": 1.0
    }
)
print(response.json())
# Output: {
#   "predictions": ["algorithms", "models", "architecture"],
#   "input_texts": ["machine learning", "deep learning", "neural network"]
# }
```

#### POST /predict/complete
Generate text completion until stop words are reached.

```bash
curl -X POST http://localhost:8000/predict/complete \
  -H "Content-Type: application/json" \
  -d '{"text": "artificial intelligence is", "temperature": 1.0, "max_length": 20}'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/complete",
    json={
        "text": "artificial intelligence is",
        "temperature": 1.0,
        "max_length": 20,
        "stop_words": [".", "!", "?"]
    }
)
print(response.json())
# Output: {
#   "completion": "artificial intelligence is a branch of computer science that focuses on creating intelligent machines.",
#   "input_text": "artificial intelligence is",
#   "stopped_by": "."
# }
```

### Model Information

#### GET /model/info
Get model architecture details and LSTM equations.

```bash
curl http://localhost:8000/model/info
```

#### GET /model/vocabulary?search=learning
Search the model vocabulary.

```bash
curl "http://localhost:8000/model/vocabulary?search=learning"
```

**Python Example**:
```python
import requests

response = requests.get(
    "http://localhost:8000/model/vocabulary",
    params={"search": "learning"}
)
print(response.json())
# Output: {
#   "query": "learning",
#   "matches": [
#     {"word": "learning", "index": 42},
#     {"word": "machine_learning", "index": 156},
#     ...
#   ],
#   "total_matches": 5
# }
```

## API Parameters

### Temperature Sampling

The `temperature` parameter controls prediction randomness:
- **0.1 - 0.5**: More deterministic, conservative predictions
- **1.0**: Balanced (default)
- **1.5 - 2.0**: More creative, diverse predictions

### Top-K Predictions

The `k` parameter specifies how many predictions to return:
- **Default**: 5
- **Range**: 1-50

### Batch Processing

The `texts` parameter accepts up to 20 input texts for batch processing.

### Text Completion

- `max_length`: Maximum number of words to generate (default: 50)
- `stop_words`: List of words that stop generation (default: [".", "?", "!", "\n"])

## Configuration

All system parameters are centralized in `src/utils/config.py`:

### Data Pipeline
- `VOCABULARY_SIZE`: 10,000 words
- `SEQUENCE_LENGTH`: 50 tokens
- `MIN_WORDS_PER_LINE`: 3 words

### Model Architecture
- `EMBEDDING_DIM`: 256 dimensions
- `BIDIRECTIONAL_LSTM_UNITS`: 512 units
- `UNIDIRECTIONAL_LSTM_UNITS`: 256 units
- `DROPOUT_RATE`: 0.3

### Training
- `EPOCHS`: 50
- `CPU_BATCH_SIZE`: 256
- `GPU_BATCH_SIZE`: 512
- `LEARNING_RATE`: 0.001

### API
- `API_HOST`: 0.0.0.0
- `API_PORT`: 8000

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_api_endpoints_basic.py

# Run with verbose output
pytest -v
```

## Troubleshooting

### Issue: Model files not found

**Error**: `Model files not found: models/lstm_model.json`

**Solution**: Run the training script first:
```bash
python scripts/run_training.py
```

### Issue: GPU not detected

**Error**: `No GPU detected, configured for CPU-only training`

**Solution**: 
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Verify TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Issue: Out of memory during training

**Error**: `ResourceExhaustedError: OOM when allocating tensor`

**Solution**: 
1. Reduce batch size in `src/utils/config.py`:
   ```python
   GPU_BATCH_SIZE = 256  # Reduce from 512
   ```
2. Enable memory growth (already enabled by default)
3. Close other GPU-intensive applications

### Issue: Import errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure you're running scripts from the project root:
```bash
cd lstm-text-prediction
python scripts/run_training.py
```

### Issue: Wikipedia API rate limiting

**Error**: `Failed to retrieve article: Rate limit exceeded`

**Solution**: The collector handles rate limiting automatically. If issues persist, add delays between requests in `src/data/collector.py`.

### Issue: Port already in use

**Error**: `OSError: [Errno 98] Address already in use`

**Solution**: Use a different port:
```bash
python scripts/run_api.py --port 8080
```

## Performance

### Training Performance
- **GPU (NVIDIA RTX 3080)**: ~45 minutes for 50 epochs
- **CPU (Intel i7-10700K)**: ~3 hours for 50 epochs

### API Performance
- **Average Response Time**: <100ms for single predictions
- **Throughput**: ~100 requests/second
- **Batch Processing**: ~50ms per text in batch

### Model Metrics
- **Vocabulary Size**: 10,000 words
- **Corpus Coverage**: ~95%
- **Validation Accuracy**: ~40-50% (typical for next-word prediction)
- **Model Parameters**: ~15M trainable parameters

## Documentation

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **LSTM Mathematics**: See `docs/lstm_math.md` for detailed equations
- **Code Documentation**: All modules include comprehensive docstrings

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper type hints and docstrings
4. Run tests: `pytest`
5. Run type checking: `mypy src --strict`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wikipedia API for providing training data
- TensorFlow team for the deep learning framework
- FastAPI team for the excellent web framework
- Rich library for beautiful console output

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using Python, TensorFlow, and FastAPI**
