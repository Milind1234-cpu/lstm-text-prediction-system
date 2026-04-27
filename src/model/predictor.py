"""Prediction engine for LSTM text prediction.

This module provides the Predictor class that loads trained LSTM models and tokenizers,
and generates predictions with various modes: single next word, top-k predictions,
batch processing, and text completion with stop words.

Supports both TensorFlow and PyTorch models.
"""

import numpy as np
from pathlib import Path
from typing import Literal

from ..data.tokenizer import Tokenizer
from ..utils.config import (
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_STOP_WORDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    MAX_BATCH_SIZE,
    MAX_TOP_K,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    MODEL_ARCHITECTURE_PATH,
    MODEL_WEIGHTS_PATH,
    MODELS_DIR,
    SEQUENCE_LENGTH,
    TOKENIZER_CONFIG_PATH,
)
from ..utils.logger import print_error, print_success, setup_logger

# Try importing TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from .lstm_model import Perplexity
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    keras = None  # type: ignore
    Perplexity = None  # type: ignore

# Try importing PyTorch
try:
    import torch
    from .lstm_model_pytorch import LSTMModelPyTorch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None  # type: ignore
    LSTMModelPyTorch = None  # type: ignore

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Predictor Class
# ============================================================================


class Predictor:
    """Prediction engine for LSTM text prediction.
    
    This class loads a trained LSTM model and tokenizer, and provides multiple
    prediction modes including single next word prediction, top-k predictions
    with probabilities, batch processing, and text completion with stop words.
    All predictions support temperature sampling for controlling randomness.
    
    Supports both TensorFlow and PyTorch models.
    
    Attributes:
        model: Loaded LSTM model (TensorFlow or PyTorch)
        tokenizer: Loaded Tokenizer instance with vocabulary
        sequence_length: Length of input sequences (50 tokens)
        is_loaded: Boolean indicating if model and tokenizer are loaded
        model_type: Type of loaded model ('tensorflow' or 'pytorch')
        device: Device for PyTorch models ('cuda' or 'cpu')
    """
    
    def __init__(self) -> None:
        """Initialize the Predictor.
        
        The model and tokenizer are not loaded during initialization.
        Call load_model() to load them from saved files.
        """
        self.model: keras.Sequential | LSTMModelPyTorch | None = None  # type: ignore
        self.tokenizer: Tokenizer | None = None
        self.sequence_length: int = SEQUENCE_LENGTH
        self.is_loaded: bool = False
        self.model_type: Literal['tensorflow', 'pytorch'] | None = None
        self.device: str = 'cpu'
        
        logger.info("Predictor initialized")
    
    def load_model(
        self,
        architecture_path: Path | None = None,
        weights_path: Path | None = None,
        tokenizer_path: Path | None = None,
        model_type: Literal['auto', 'tensorflow', 'pytorch'] = 'auto',
    ) -> None:
        """Load model and tokenizer from saved files.
        
        Automatically detects model type based on file extensions if model_type='auto'.
        PyTorch models use .pth extension, TensorFlow models use .h5 extension.
        
        Args:
            architecture_path: Path to model architecture JSON file (TensorFlow only).
                Defaults to MODEL_ARCHITECTURE_PATH from config.
            weights_path: Path to model weights file (.pth for PyTorch, .h5 for TensorFlow).
                Defaults to MODEL_WEIGHTS_PATH from config.
            tokenizer_path: Path to tokenizer config JSON file.
                Defaults to TOKENIZER_CONFIG_PATH from config.
            model_type: Type of model to load ('auto', 'tensorflow', or 'pytorch').
                Defaults to 'auto' which auto-detects based on available files.
        
        Raises:
            FileNotFoundError: If any of the required files do not exist
            ValueError: If files are invalid or corrupted, or if required framework is not available
            IOError: If files cannot be read
        """
        tokenizer_path = tokenizer_path or TOKENIZER_CONFIG_PATH
        
        # Auto-detect model type if requested
        if model_type == 'auto':
            pytorch_weights = MODELS_DIR / "lstm_weights_pytorch.pth"
            tensorflow_weights = MODEL_WEIGHTS_PATH
            
            if pytorch_weights.exists():
                model_type = 'pytorch'
                weights_path = pytorch_weights
                logger.info("Auto-detected PyTorch model")
            elif tensorflow_weights.exists():
                model_type = 'tensorflow'
                weights_path = tensorflow_weights
                architecture_path = architecture_path or MODEL_ARCHITECTURE_PATH
                logger.info("Auto-detected TensorFlow model")
            else:
                error_msg = "No trained model found. Please train a model first."
                logger.error(error_msg)
                print_error(error_msg, title="Model Not Found")
                raise FileNotFoundError(error_msg)
        else:
            # Use provided or default paths
            if model_type == 'pytorch':
                weights_path = weights_path or (MODELS_DIR / "lstm_weights_pytorch.pth")
            else:  # tensorflow
                weights_path = weights_path or MODEL_WEIGHTS_PATH
                architecture_path = architecture_path or MODEL_ARCHITECTURE_PATH
        
        # Load based on model type
        if model_type == 'pytorch':
            self._load_pytorch_model(weights_path, tokenizer_path)
        elif model_type == 'tensorflow':
            self._load_tensorflow_model(architecture_path, weights_path, tokenizer_path)
        else:
            error_msg = f"Invalid model_type: {model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _load_tensorflow_model(
        self,
        architecture_path: Path,
        weights_path: Path,
        tokenizer_path: Path,
    ) -> None:
        """Load TensorFlow model.
        
        Args:
            architecture_path: Path to model architecture JSON file
            weights_path: Path to model weights H5 file
            tokenizer_path: Path to tokenizer config JSON file
        
        Raises:
            ValueError: If TensorFlow is not available
            FileNotFoundError: If required files do not exist
        """
        if not TENSORFLOW_AVAILABLE:
            error_msg = "TensorFlow is not installed. Install it to use TensorFlow models."
            logger.error(error_msg)
            print_error(error_msg, title="TensorFlow Not Available")
            raise ValueError(error_msg)
        
        # Check if all files exist
        missing_files = []
        if not architecture_path.exists():
            missing_files.append(str(architecture_path))
        if not weights_path.exists():
            missing_files.append(str(weights_path))
        if not tokenizer_path.exists():
            missing_files.append(str(tokenizer_path))
        
        if missing_files:
            error_msg = f"Missing model files: {', '.join(missing_files)}"
            logger.error(error_msg)
            print_error(error_msg, title="Model Files Not Found")
            raise FileNotFoundError(error_msg)
        
        try:
            # Load model architecture
            logger.info(f"Loading TensorFlow model architecture from {architecture_path}")
            with open(architecture_path, 'r', encoding='utf-8') as f:
                architecture_json = f.read()
            
            # Create model from JSON with custom objects
            self.model = keras.models.model_from_json(
                architecture_json,
                custom_objects={'Perplexity': Perplexity}
            )
            
            # Load model weights
            logger.info(f"Loading TensorFlow model weights from {weights_path}")
            self.model.load_weights(weights_path)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = Tokenizer()
            self.tokenizer.load_vocabulary(tokenizer_path)
            
            self.is_loaded = True
            self.model_type = 'tensorflow'
            
            logger.info("TensorFlow model and tokenizer loaded successfully")
            print_success(
                f"TensorFlow model and tokenizer loaded successfully!\n"
                f"Architecture: {architecture_path}\n"
                f"Weights: {weights_path}\n"
                f"Tokenizer: {tokenizer_path}",
                title="Model Loaded"
            )
        
        except Exception as e:
            error_msg = f"Failed to load TensorFlow model: {e}"
            logger.error(error_msg)
            print_error(error_msg, title="Model Load Failed")
            raise
    
    def _load_pytorch_model(
        self,
        weights_path: Path,
        tokenizer_path: Path,
    ) -> None:
        """Load PyTorch model.
        
        Args:
            weights_path: Path to model weights PTH file
            tokenizer_path: Path to tokenizer config JSON file
        
        Raises:
            ValueError: If PyTorch is not available
            FileNotFoundError: If required files do not exist
        """
        if not PYTORCH_AVAILABLE:
            error_msg = "PyTorch is not installed. Install it to use PyTorch models."
            logger.error(error_msg)
            print_error(error_msg, title="PyTorch Not Available")
            raise ValueError(error_msg)
        
        # Check if all files exist
        missing_files = []
        if not weights_path.exists():
            missing_files.append(str(weights_path))
        if not tokenizer_path.exists():
            missing_files.append(str(tokenizer_path))
        
        if missing_files:
            error_msg = f"Missing model files: {', '.join(missing_files)}"
            logger.error(error_msg)
            print_error(error_msg, title="Model Files Not Found")
            raise FileNotFoundError(error_msg)
        
        try:
            # Load tokenizer first to get vocabulary size
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            self.tokenizer = Tokenizer()
            self.tokenizer.load_vocabulary(tokenizer_path)
            
            # Determine device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Create model instance
            logger.info(f"Loading PyTorch model from {weights_path}")
            self.model = LSTMModelPyTorch(device=self.device)
            
            # Load model weights
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            
            self.is_loaded = True
            self.model_type = 'pytorch'
            
            logger.info("PyTorch model and tokenizer loaded successfully")
            print_success(
                f"PyTorch model and tokenizer loaded successfully!\n"
                f"Weights: {weights_path}\n"
                f"Tokenizer: {tokenizer_path}\n"
                f"Device: {self.device.upper()}",
                title="Model Loaded"
            )
        
        except Exception as e:
            error_msg = f"Failed to load PyTorch model: {e}"
            logger.error(error_msg)
            print_error(error_msg, title="Model Load Failed")
            raise
    
    def _validate_loaded(self) -> None:
        """Validate that model and tokenizer are loaded.
        
        Raises:
            ValueError: If model or tokenizer is not loaded
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            error_msg = "Model not loaded. Call load_model() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_temperature(self, temperature: float) -> None:
        """Validate temperature parameter.
        
        Args:
            temperature: Temperature value to validate
        
        Raises:
            ValueError: If temperature is outside valid range
        """
        if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
            error_msg = (
                f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, "
                f"got {temperature}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _prepare_input(self, text: str) -> np.ndarray:
        """Prepare input text for prediction.
        
        Tokenizes the input text and pads or truncates to sequence_length.
        
        Args:
            text: Input text string
        
        Returns:
            Numpy array of shape (1, sequence_length) ready for model input
        
        Raises:
            ValueError: If text is empty or tokenizer is not loaded
        """
        if not text or not text.strip():
            error_msg = "Input text cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self._validate_loaded()
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        
        # Tokenize text
        sequence = self.tokenizer.text_to_sequence(text)
        
        # Pad or truncate to sequence_length
        if len(sequence) < self.sequence_length:
            # Pad with padding token index (0)
            padding_length = self.sequence_length - len(sequence)
            sequence = [0] * padding_length + sequence
        else:
            # Truncate to last sequence_length tokens
            sequence = sequence[-self.sequence_length:]
        
        # Convert to numpy array with batch dimension
        input_array = np.array([sequence], dtype=np.int32)
        
        logger.debug(f"Prepared input: {input_array.shape}")
        
        return input_array
    
    def _apply_temperature(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature sampling to logits.
        
        Temperature controls the randomness of predictions:
        - temperature < 1.0: More confident, less random
        - temperature = 1.0: No modification
        - temperature > 1.0: Less confident, more random
        
        Args:
            logits: Raw model output logits
            temperature: Temperature parameter
        
        Returns:
            Temperature-scaled probabilities
        """
        # Scale logits by temperature
        scaled_logits = logits / temperature
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities
    
    def predict_next_word(
        self,
        text: str,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Predict the single most likely next word.
        
        Args:
            text: Input text string
            temperature: Sampling temperature (default: 1.0)
        
        Returns:
            The predicted next word
        
        Raises:
            ValueError: If model not loaded, text is empty, or temperature is invalid
        """
        self._validate_loaded()
        self._validate_temperature(temperature)
        assert self.model is not None, "Model must be loaded"
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        
        logger.info(f"Predicting next word for text: '{text[:50]}...'")
        
        # Prepare input
        input_array = self._prepare_input(text)
        
        # Get model predictions based on model type
        if self.model_type == 'pytorch':
            # PyTorch prediction
            input_tensor = torch.from_numpy(input_array).long().to(self.device)
            with torch.no_grad():
                predictions = self.model(input_tensor)
            logits = predictions[0].cpu().numpy()
        else:
            # TensorFlow prediction
            predictions = self.model.predict(input_array, verbose=0)
            logits = predictions[0]
        
        # Apply temperature sampling
        probabilities = self._apply_temperature(logits, temperature)
        
        # Get the word with highest probability
        predicted_index = np.argmax(probabilities)
        predicted_word = self.tokenizer.get_index_word(int(predicted_index))
        
        logger.info(f"Predicted word: '{predicted_word}'")
        
        return predicted_word
    
    def predict_top_k(
        self,
        text: str,
        k: int = DEFAULT_TOP_K,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> list[dict[str, str | float]]:
        """Predict top-k most likely next words with probabilities.
        
        Args:
            text: Input text string
            k: Number of predictions to return (default: 5, max: 50)
            temperature: Sampling temperature (default: 1.0)
        
        Returns:
            List of dictionaries with 'word' and 'probability' keys,
            sorted by probability in descending order
        
        Raises:
            ValueError: If model not loaded, text is empty, temperature is invalid,
                or k is out of range
        """
        self._validate_loaded()
        self._validate_temperature(temperature)
        assert self.model is not None, "Model must be loaded"
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        
        # Validate k parameter
        if k < 1:
            error_msg = f"k must be at least 1, got {k}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Limit k to MAX_TOP_K
        if k > MAX_TOP_K:
            logger.warning(f"k={k} exceeds maximum {MAX_TOP_K}, limiting to {MAX_TOP_K}")
            k = MAX_TOP_K
        
        logger.info(f"Predicting top-{k} words for text: '{text[:50]}...'")
        
        # Prepare input
        input_array = self._prepare_input(text)
        
        # Get model predictions based on model type
        if self.model_type == 'pytorch':
            # PyTorch prediction
            input_tensor = torch.from_numpy(input_array).long().to(self.device)
            with torch.no_grad():
                predictions = self.model(input_tensor)
            logits = predictions[0].cpu().numpy()
        else:
            # TensorFlow prediction
            predictions = self.model.predict(input_array, verbose=0)
            logits = predictions[0]
        
        # Apply temperature sampling
        probabilities = self._apply_temperature(logits, temperature)
        
        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        
        # Build result list
        results = []
        for idx in top_k_indices:
            word = self.tokenizer.get_index_word(int(idx))
            probability = float(probabilities[idx])
            result_item: dict[str, str | float] = {
                'word': word,
                'probability': probability
            }
            results.append(result_item)
        
        logger.info(f"Predicted top-{k} words: {[r['word'] for r in results]}")
        
        return results
    
    def predict_batch(
        self,
        texts: list[str],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> list[str]:
        """Predict next words for multiple texts in batch.
        
        Processes multiple input texts using vectorized operations for efficiency.
        
        Args:
            texts: List of input text strings
            temperature: Sampling temperature applied to all texts (default: 1.0)
        
        Returns:
            List of predicted next words in the same order as input texts
        
        Raises:
            ValueError: If model not loaded, batch is empty, batch size exceeds limit,
                temperature is invalid, or any text is empty
        """
        self._validate_loaded()
        self._validate_temperature(temperature)
        
        # Validate batch
        if not texts:
            error_msg = "Batch cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(texts) > MAX_BATCH_SIZE:
            error_msg = (
                f"Batch size {len(texts)} exceeds maximum {MAX_BATCH_SIZE}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Predicting batch of {len(texts)} texts")
        
        # Prepare all inputs
        input_arrays = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                error_msg = f"Text at index {i} is empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            input_array = self._prepare_input(text)
            input_arrays.append(input_array[0])  # Remove batch dimension
        
        assert self.model is not None, "Model must be loaded"
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        
        # Stack into batch
        batch_input = np.array(input_arrays, dtype=np.int32)
        
        # Get model predictions for entire batch based on model type
        if self.model_type == 'pytorch':
            # PyTorch prediction
            input_tensor = torch.from_numpy(batch_input).long().to(self.device)
            with torch.no_grad():
                predictions = self.model(input_tensor)
            predictions = predictions.cpu().numpy()
        else:
            # TensorFlow prediction
            predictions = self.model.predict(batch_input, verbose=0)
        
        # Process each prediction
        results = []
        for i, logits in enumerate(predictions):
            # Apply temperature sampling
            probabilities = self._apply_temperature(logits, temperature)
            
            # Get the word with highest probability
            predicted_index = np.argmax(probabilities)
            predicted_word = self.tokenizer.get_index_word(int(predicted_index))
            
            results.append(predicted_word)
        
        logger.info(f"Predicted batch: {results}")
        
        return results
    
    def complete_text(
        self,
        text: str,
        max_length: int = DEFAULT_MAX_COMPLETION_LENGTH,
        stop_words: list[str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Generate text completion until stop words or max length.
        
        Iteratively predicts next words and appends them to the input text
        until a stop word is encountered or maximum length is reached.
        
        Args:
            text: Input text string to complete
            max_length: Maximum number of words to generate (default: 50)
            stop_words: List of words that stop generation (default: [".", "?", "!", "\\n"])
            temperature: Sampling temperature (default: 1.0)
        
        Returns:
            Completed text string
        
        Raises:
            ValueError: If model not loaded, text is empty, temperature is invalid,
                or max_length is invalid
        """
        self._validate_loaded()
        self._validate_temperature(temperature)
        
        if max_length < 1:
            error_msg = f"max_length must be at least 1, got {max_length}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Use default stop words if not provided
        if stop_words is None:
            stop_words = DEFAULT_STOP_WORDS
        
        logger.info(
            f"Completing text: '{text[:50]}...' "
            f"(max_length={max_length}, stop_words={stop_words})"
        )
        
        # Start with input text
        current_text = text
        generated_words = []
        
        # Generate words iteratively
        for i in range(max_length):
            # Predict next word
            next_word = self.predict_next_word(current_text, temperature)
            
            # Check if stop word
            if next_word in stop_words:
                logger.info(f"Stop word '{next_word}' encountered at iteration {i+1}")
                generated_words.append(next_word)
                break
            
            # Append to current text
            generated_words.append(next_word)
            current_text = current_text + " " + next_word
        
        # Build completed text
        completed_text = text + " " + " ".join(generated_words)
        
        logger.info(f"Generated {len(generated_words)} words")
        
        return completed_text


# ============================================================================
# Module-Level Functions
# ============================================================================


def create_predictor() -> Predictor:
    """Create and initialize a Predictor instance.
    
    This is a convenience function for creating a Predictor instance.
    The model and tokenizer must be loaded separately using load_model().
    
    Returns:
        Initialized Predictor instance
    """
    return Predictor()
