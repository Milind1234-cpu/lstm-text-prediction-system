"""GPU detection and configuration for TensorFlow training.

This module provides GPU management functionality including NVIDIA GPU detection,
memory growth configuration, mixed precision training setup, and CPU fallback.
"""

import os
from typing import Final

import tensorflow as tf
from rich.table import Table

from ..utils.config import (
    ENABLE_CUDA_MALLOC_ASYNC,
    ENABLE_MEMORY_GROWTH,
    ENABLE_MIXED_PRECISION,
)
from ..utils.logger import console, print_error, print_panel, print_success, setup_logger

# ============================================================================
# Logger Setup
# ============================================================================

logger = setup_logger(__name__)


# ============================================================================
# GPU Manager Class
# ============================================================================


class GPUManager:
    """Manages GPU detection and configuration for TensorFlow training.

    This class handles NVIDIA GPU detection, memory growth configuration,
    mixed precision training setup, CUDA malloc async allocator configuration,
    and CPU-only fallback when no GPU is available.

    Attributes:
        gpu_available: Whether a GPU is available for training
        gpu_devices: List of available GPU devices
        device_name: Name of the GPU device (or "CPU" if no GPU)
        memory_info: GPU memory information string
    """

    def __init__(self) -> None:
        """Initialize GPU manager and configure GPU/CPU settings."""
        self.gpu_available: bool = False
        self.gpu_devices: list[tf.config.PhysicalDevice] = []
        self.device_name: str = "CPU"
        self.memory_info: str = "N/A"

        # Configure GPU or CPU
        self._detect_and_configure_gpu()

    def _detect_and_configure_gpu(self) -> None:
        """Detect available GPUs and configure TensorFlow settings.

        This method performs the following steps:
        1. Detects available NVIDIA GPUs using TensorFlow
        2. If GPU found: configures memory growth, mixed precision, and CUDA malloc
        3. If no GPU found: configures CPU-only training
        4. Logs configuration status with Rich panels
        """
        try:
            # Detect available GPUs
            self.gpu_devices = tf.config.list_physical_devices("GPU")

            if self.gpu_devices:
                self._configure_gpu()
            else:
                self._configure_cpu()

        except Exception as e:
            logger.error(f"Error during GPU detection: {e}")
            print_error(
                f"GPU detection failed: {e}\nFalling back to CPU-only mode.",
                title="GPU Detection Error",
            )
            self._configure_cpu()

    def _configure_gpu(self) -> None:
        """Configure GPU settings for training.

        Configures:
        - Memory growth to prevent allocation errors
        - Mixed precision training (float16) for better performance
        - CUDA malloc async allocator for improved memory management
        """
        try:
            self.gpu_available = True
            gpu_device = self.gpu_devices[0]

            # Get GPU device name
            device_details = tf.config.experimental.get_device_details(gpu_device)
            self.device_name = device_details.get("device_name", "Unknown GPU")

            # Configure memory growth to prevent allocation errors
            if ENABLE_MEMORY_GROWTH:
                tf.config.experimental.set_memory_growth(gpu_device, True)
                logger.info("Enabled GPU memory growth")

            # Configure mixed precision training (float16)
            if ENABLE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Enabled mixed precision training (float16)")

            # Configure CUDA malloc async allocator
            if ENABLE_CUDA_MALLOC_ASYNC:
                os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
                logger.info("Enabled CUDA malloc async allocator")

            # Get memory information
            try:
                gpu_details = tf.config.experimental.get_memory_info(gpu_device.name)
                total_memory_gb = gpu_details.get("current", 0) / (1024**3)
                self.memory_info = f"{total_memory_gb:.2f} GB current"
            except Exception:
                # Memory info not available on all platforms
                self.memory_info = "Memory info not available"

            # Display GPU configuration with Rich panel
            self._display_gpu_config()

            logger.info(f"GPU configuration complete: {self.device_name}")

        except Exception as e:
            logger.error(f"Error configuring GPU: {e}")
            print_error(
                f"GPU configuration failed: {e}\nFalling back to CPU-only mode.",
                title="GPU Configuration Error",
            )
            self._configure_cpu()

    def _configure_cpu(self) -> None:
        """Configure CPU-only training when no GPU is available."""
        self.gpu_available = False
        self.device_name = "CPU"
        self.memory_info = "N/A"

        # Disable GPU visibility to ensure CPU-only mode
        tf.config.set_visible_devices([], "GPU")

        logger.info("No GPU detected, configured for CPU-only training")

        # Display CPU configuration with Rich panel
        self._display_cpu_config()

    def _display_gpu_config(self) -> None:
        """Display GPU configuration information using Rich panels and tables."""
        # Create configuration table
        config_table = Table(
            title="GPU Configuration",
            show_header=True,
            header_style="bold magenta",
            border_style="green",
        )

        config_table.add_column("Setting", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="green")

        config_table.add_row("Device", self.device_name)
        config_table.add_row("Memory", self.memory_info)
        config_table.add_row(
            "Memory Growth", "Enabled" if ENABLE_MEMORY_GROWTH else "Disabled"
        )
        config_table.add_row(
            "Mixed Precision", "Enabled (float16)" if ENABLE_MIXED_PRECISION else "Disabled"
        )
        config_table.add_row(
            "CUDA Malloc Async", "Enabled" if ENABLE_CUDA_MALLOC_ASYNC else "Disabled"
        )

        console.print()
        console.print(config_table)
        console.print()

        print_success(
            f"GPU acceleration enabled with {self.device_name}",
            title="GPU Configuration Complete",
        )

    def _display_cpu_config(self) -> None:
        """Display CPU configuration information using Rich panels."""
        # Create configuration table
        config_table = Table(
            title="CPU Configuration",
            show_header=True,
            header_style="bold magenta",
            border_style="yellow",
        )

        config_table.add_column("Setting", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="yellow")

        config_table.add_row("Device", "CPU")
        config_table.add_row("GPU Available", "No")
        config_table.add_row("Training Mode", "CPU-only")

        console.print()
        console.print(config_table)
        console.print()

        print_panel(
            "No GPU detected. Training will use CPU only.\n"
            "For faster training, ensure NVIDIA GPU drivers and CUDA are installed.",
            title="CPU-Only Mode",
            style="yellow",
            border_style="yellow",
        )

    def get_batch_size(self, cpu_batch_size: int, gpu_batch_size: int) -> int:
        """Get appropriate batch size based on GPU availability.

        Args:
            cpu_batch_size: Batch size to use for CPU training
            gpu_batch_size: Batch size to use for GPU training

        Returns:
            Appropriate batch size based on available hardware
        """
        return gpu_batch_size if self.gpu_available else cpu_batch_size

    def get_device_info(self) -> dict[str, str | bool]:
        """Get GPU/CPU device information.

        Returns:
            Dictionary containing device information:
            - gpu_available: Whether GPU is available
            - device_name: Name of the device
            - memory_info: Memory information string
        """
        return {
            "gpu_available": self.gpu_available,
            "device_name": self.device_name,
            "memory_info": self.memory_info,
        }


# ============================================================================
# Module-Level Functions
# ============================================================================


def initialize_gpu() -> GPUManager:
    """Initialize and configure GPU manager.

    This is a convenience function for creating and configuring a GPU manager.

    Returns:
        Configured GPUManager instance
    """
    return GPUManager()
