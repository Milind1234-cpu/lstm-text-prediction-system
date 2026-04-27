"""Logging configuration with Rich console output.

This module provides beautiful console logging using the Rich library
with progress bars, tables, panels, and colored output for better visibility.
"""

import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .config import CONSOLE_LOG_LEVEL, FILE_LOG_LEVEL, LOG_FILE_PATH, LOG_FORMAT

# ============================================================================
# Rich Console Instance
# ============================================================================

# Global console instance for rich output
console = Console()

# ============================================================================
# Logger Configuration
# ============================================================================


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with Rich console handler and file handler.

    Args:
        name: Name of the logger (typically __name__ of the calling module)

    Returns:
        Configured logger instance with Rich console and file handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Rich console handler for beautiful terminal output
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
    )
    console_handler.setLevel(getattr(logging, CONSOLE_LOG_LEVEL))
    logger.addHandler(console_handler)

    # File handler for persistent logging
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, FILE_LOG_LEVEL))
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# Progress Bar Utilities
# ============================================================================


def create_progress_bar() -> Progress:
    """Create a Rich progress bar with standard columns.

    Returns:
        Configured Progress instance with spinner, text, bar, percentage,
        completed/total, time elapsed, and time remaining columns
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


# ============================================================================
# Table Utilities
# ============================================================================


def create_table(title: str, *columns: str, **kwargs: Any) -> Table:
    """Create a Rich table with standard styling.

    Args:
        title: Title of the table
        *columns: Column names for the table
        **kwargs: Additional keyword arguments passed to Table constructor

    Returns:
        Configured Table instance with specified columns
    """
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
        **kwargs,
    )

    for column in columns:
        table.add_column(column, style="cyan")

    return table


# ============================================================================
# Panel Utilities
# ============================================================================


def print_panel(
    content: str,
    title: str | None = None,
    style: str = "blue",
    border_style: str = "blue",
) -> None:
    """Print content in a Rich panel with styling.

    Args:
        content: Text content to display in the panel
        title: Optional title for the panel
        style: Style for the panel content (default: "blue")
        border_style: Style for the panel border (default: "blue")
    """
    panel = Panel(
        content,
        title=title,
        style=style,
        border_style=border_style,
        expand=False,
    )
    console.print(panel)


def print_success(message: str, title: str = "Success") -> None:
    """Print a success message in a green panel.

    Args:
        message: Success message to display
        title: Panel title (default: "Success")
    """
    print_panel(message, title=title, style="green", border_style="green")


def print_error(message: str, title: str = "Error") -> None:
    """Print an error message in a red panel.

    Args:
        message: Error message to display
        title: Panel title (default: "Error")
    """
    print_panel(message, title=title, style="red", border_style="red")


def print_warning(message: str, title: str = "Warning") -> None:
    """Print a warning message in a yellow panel.

    Args:
        message: Warning message to display
        title: Panel title (default: "Warning")
    """
    print_panel(message, title=title, style="yellow", border_style="yellow")


def print_info(message: str, title: str = "Info") -> None:
    """Print an info message in a blue panel.

    Args:
        message: Info message to display
        title: Panel title (default: "Info")
    """
    print_panel(message, title=title, style="blue", border_style="blue")


# ============================================================================
# Separator Utilities
# ============================================================================


def print_separator(char: str = "=", length: int = 80, style: str = "blue") -> None:
    """Print a separator line.

    Args:
        char: Character to use for the separator (default: "=")
        length: Length of the separator line (default: 80)
        style: Style for the separator (default: "blue")
    """
    console.print(char * length, style=style)


def print_header(text: str, style: str = "bold blue") -> None:
    """Print a header with separators.

    Args:
        text: Header text to display
        style: Style for the header text (default: "bold blue")
    """
    print_separator()
    console.print(f"\n{text}\n", style=style, justify="center")
    print_separator()
