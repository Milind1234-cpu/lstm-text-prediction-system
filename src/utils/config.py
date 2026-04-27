"""Configuration management for LSTM Text Prediction System.

This module centralizes all hyperparameters, paths, and configuration settings
for the entire system including data pipeline, model training, and API deployment.
"""

from pathlib import Path
from typing import Final

# ============================================================================
# Project Paths
# ============================================================================

# Root directory of the project
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

# Model directories
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
MODEL_ARCHITECTURE_PATH: Final[Path] = MODELS_DIR / "lstm_model.json"
MODEL_WEIGHTS_PATH: Final[Path] = MODELS_DIR / "lstm_weights.weights.h5"
TOKENIZER_CONFIG_PATH: Final[Path] = MODELS_DIR / "tokenizer_config.json"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data Collection Configuration
# ============================================================================

# Wikipedia topics for AI/ML data collection
WIKIPEDIA_TOPICS: Final[list[str]] = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Natural language processing",
    "Computer vision",
    "Reinforcement learning",
    "Supervised learning",
    "Unsupervised learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Long short-term memory",
    "Transformer (machine learning)",
    "Attention mechanism",
    "Generative adversarial network",
    "Support vector machine",
    "Decision tree",
    "Random forest",
    "Gradient boosting",
    "TensorFlow"
]

# Number of articles to collect (must match WIKIPEDIA_TOPICS length)
NUM_ARTICLES: Final[int] = 20

# ============================================================================
# Text Preprocessing Configuration
# ============================================================================

# Minimum number of words per line to keep during preprocessing
MIN_WORDS_PER_LINE: Final[int] = 3

# ============================================================================
# Tokenization Configuration
# ============================================================================

# Vocabulary size (number of most frequent words to keep)
VOCABULARY_SIZE: Final[int] = 10000

# Special tokens
UNKNOWN_TOKEN: Final[str] = "<UNK>"
PADDING_TOKEN: Final[str] = "<PAD>"

# ============================================================================
# Sequence Generation Configuration
# ============================================================================

# Length of input sequences for training
SEQUENCE_LENGTH: Final[int] = 50

# Stride for sliding window sequence generation
SEQUENCE_STRIDE: Final[int] = 1

# Train/validation split ratio
TRAIN_SPLIT_RATIO: Final[float] = 0.8
VALIDATION_SPLIT_RATIO: Final[float] = 0.2

# ============================================================================
# Model Architecture Configuration
# ============================================================================

# Embedding layer configuration
EMBEDDING_DIM: Final[int] = 256

# LSTM layer configuration
BIDIRECTIONAL_LSTM_UNITS: Final[int] = 512
UNIDIRECTIONAL_LSTM_UNITS: Final[int] = 256

# Dropout configuration
DROPOUT_RATE: Final[float] = 0.3

# Output layer configuration (must match VOCABULARY_SIZE)
OUTPUT_UNITS: Final[int] = VOCABULARY_SIZE

# ============================================================================
# Training Configuration
# ============================================================================

# Number of training epochs
EPOCHS: Final[int] = 50

# Batch size for CPU training
CPU_BATCH_SIZE: Final[int] = 256

# Batch size for GPU training (larger for better GPU utilization)
GPU_BATCH_SIZE: Final[int] = 512

# Optimizer configuration
LEARNING_RATE: Final[float] = 0.001

# ============================================================================
# GPU Configuration
# ============================================================================

# Enable mixed precision training (float16) for GPU
ENABLE_MIXED_PRECISION: Final[bool] = True

# Enable memory growth to prevent allocation errors
ENABLE_MEMORY_GROWTH: Final[bool] = True

# Enable CUDA malloc async allocator
ENABLE_CUDA_MALLOC_ASYNC: Final[bool] = True

# ============================================================================
# Prediction Configuration
# ============================================================================

# Default temperature for sampling (1.0 = no modification)
DEFAULT_TEMPERATURE: Final[float] = 1.0

# Temperature range constraints
MIN_TEMPERATURE: Final[float] = 0.1
MAX_TEMPERATURE: Final[float] = 2.0

# Default number of top-k predictions
DEFAULT_TOP_K: Final[int] = 5

# Maximum number of top-k predictions
MAX_TOP_K: Final[int] = 50

# Maximum batch size for batch predictions
MAX_BATCH_SIZE: Final[int] = 20

# Default maximum length for text completion
DEFAULT_MAX_COMPLETION_LENGTH: Final[int] = 50

# Default stop words for text completion
DEFAULT_STOP_WORDS: Final[list[str]] = [".", "?", "!", "\n"]

# ============================================================================
# API Configuration
# ============================================================================

# API server host and port
API_HOST: Final[str] = "0.0.0.0"
API_PORT: Final[int] = 8000

# API metadata
API_TITLE: Final[str] = "LSTM Text Prediction API"
API_VERSION: Final[str] = "1.0.0"
API_DESCRIPTION: Final[str] = (
    "Production-ready text prediction API powered by bidirectional LSTM neural network. "
    "Provides next-word prediction, top-k predictions, batch processing, and text completion."
)

# CORS configuration
CORS_ALLOW_ORIGINS: Final[list[str]] = ["*"]
CORS_ALLOW_METHODS: Final[list[str]] = ["*"]
CORS_ALLOW_HEADERS: Final[list[str]] = ["*"]

# API limits
MAX_VOCABULARY_SEARCH_RESULTS: Final[int] = 100

# ============================================================================
# Logging Configuration
# ============================================================================

# Log file path
LOG_FILE_PATH: Final[Path] = PROJECT_ROOT / "api.log"

# Log format
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Console logging level
CONSOLE_LOG_LEVEL: Final[str] = "INFO"

# File logging level
FILE_LOG_LEVEL: Final[str] = "DEBUG"
