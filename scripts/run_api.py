"""API server startup script for LSTM text prediction system.

This script starts the FastAPI server with Uvicorn, verifies model files exist,
and displays server information including URLs for API access and documentation.

Usage:
    python scripts/run_api.py              # Production mode
    python scripts/run_api.py --reload     # Development mode with auto-reload
    python scripts/run_api.py --port 8080  # Custom port
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

from src.utils.config import (
    API_HOST,
    API_PORT,
    API_TITLE,
    API_VERSION,
    MODEL_ARCHITECTURE_PATH,
    MODEL_WEIGHTS_PATH,
    TOKENIZER_CONFIG_PATH,
)
from src.utils.logger import (
    console,
    print_panel,
    print_success,
    print_error,
    create_table,
    setup_logger,
)

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# Model Verification
# ============================================================================


def verify_model_files() -> bool:
    """Verify that all required model files exist.
    
    Returns:
        True if all files exist, False otherwise
    """
    required_files = {
        "Model Architecture": MODEL_ARCHITECTURE_PATH,
        "Model Weights": MODEL_WEIGHTS_PATH,
        "Tokenizer Config": TOKENIZER_CONFIG_PATH,
    }
    
    missing_files = []
    
    for file_type, file_path in required_files.items():
        if not file_path.exists():
            missing_files.append((file_type, file_path))
            logger.error(f"Missing {file_type}: {file_path}")
    
    if missing_files:
        # Display error with missing files
        error_table = create_table(
            "Missing Model Files",
            "File Type",
            "Expected Path",
        )
        
        for file_type, file_path in missing_files:
            error_table.add_row(file_type, str(file_path))
        
        console.print()
        console.print(error_table)
        console.print()
        
        print_error(
            "Model files not found!\n\n"
            "Please run the training script first to generate model files:\n"
            "  python scripts/run_training.py\n\n"
            "This will:\n"
            "  1. Collect Wikipedia data\n"
            "  2. Preprocess and tokenize text\n"
            "  3. Train the LSTM model\n"
            "  4. Save model and tokenizer files",
            title="Model Files Not Found",
        )
        
        return False
    
    # All files exist - display verification success
    files_table = create_table(
        "Model Files Verified",
        "File Type",
        "Path",
        "Size",
    )
    
    for file_type, file_path in required_files.items():
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        files_table.add_row(
            file_type,
            str(file_path),
            f"{size_mb:.2f} MB",
        )
    
    console.print()
    console.print(files_table)
    console.print()
    
    logger.info("All model files verified successfully")
    return True


# ============================================================================
# Server Startup
# ============================================================================


def start_server(host: str, port: int, reload: bool = False) -> None:
    """Start the FastAPI server with Uvicorn.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        reload: Enable auto-reload for development mode
    """
    # Display server configuration
    config_table = create_table(
        "Server Configuration",
        "Setting",
        "Value",
    )
    
    config_table.add_row("API Title", API_TITLE)
    config_table.add_row("API Version", API_VERSION)
    config_table.add_row("Host", host)
    config_table.add_row("Port", str(port))
    config_table.add_row("Mode", "Development (auto-reload)" if reload else "Production")
    
    console.print()
    console.print(config_table)
    console.print()
    
    # Display server URLs
    server_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    
    urls_table = create_table(
        "Server URLs",
        "Endpoint",
        "URL",
    )
    
    urls_table.add_row("API Root", server_url)
    urls_table.add_row("Swagger UI", f"{server_url}/docs")
    urls_table.add_row("ReDoc", f"{server_url}/redoc")
    urls_table.add_row("OpenAPI JSON", f"{server_url}/openapi.json")
    urls_table.add_row("Health Check", f"{server_url}/health")
    
    console.print()
    console.print(urls_table)
    console.print()
    
    print_success(
        f"Starting {API_TITLE} v{API_VERSION}\n\n"
        f"Server will be available at: {server_url}\n"
        f"API documentation: {server_url}/docs\n\n"
        f"Press Ctrl+C to stop the server",
        title="Starting API Server",
    )
    
    logger.info(f"Starting Uvicorn server on {host}:{port} (reload={reload})")
    
    try:
        # Start Uvicorn server
        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True,
        )
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
        print_panel(
            "Server stopped by user",
            title="Server Stopped",
            style="yellow",
            border_style="yellow",
        )
    
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        print_error(
            f"Server failed to start:\n{e}",
            title="Server Error",
        )
        sys.exit(1)


# ============================================================================
# Command Line Interface
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Start the LSTM Text Prediction API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_api.py                    # Start in production mode
  python scripts/run_api.py --reload           # Start with auto-reload (development)
  python scripts/run_api.py --port 8080        # Start on custom port
  python scripts/run_api.py --host 127.0.0.1   # Start on specific host
        """,
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=API_HOST,
        help=f"Host address to bind to (default: {API_HOST})",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=API_PORT,
        help=f"Port number to listen on (default: {API_PORT})",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development mode",
    )
    
    return parser.parse_args()


# ============================================================================
# Main Function
# ============================================================================


def main() -> None:
    """Main function to start the API server."""
    
    # Display welcome message
    print_panel(
        f"{API_TITLE} v{API_VERSION}\n\n"
        "Production-ready text prediction API powered by\n"
        "bidirectional LSTM neural network.\n\n"
        "Features:\n"
        "  • Next-word prediction with temperature sampling\n"
        "  • Top-k predictions with probabilities\n"
        "  • Batch prediction processing\n"
        "  • Text completion with stop words\n"
        "  • Health monitoring and metrics tracking\n"
        "  • Interactive API documentation (Swagger UI)",
        title="LSTM Text Prediction API",
        style="bold cyan",
        border_style="cyan",
    )
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("API server startup initiated")
    logger.info(f"Host: {args.host}, Port: {args.port}, Reload: {args.reload}")
    
    # Verify model files exist
    print_panel(
        "Verifying model files...",
        title="Model Verification",
        style="bold blue",
        border_style="blue",
    )
    
    if not verify_model_files():
        logger.error("Model verification failed - exiting")
        sys.exit(1)
    
    print_success(
        "Model files verified successfully!",
        title="Verification Complete",
    )
    
    # Start the server
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


# ============================================================================
# Script Entry Point
# ============================================================================


if __name__ == "__main__":
    main()
