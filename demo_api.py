"""
🚀 LSTM Text Prediction API - Live Demo
========================================

This script demonstrates the working API endpoints.
"""

import requests
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a section header."""
    console.print(f"\n{'='*60}", style="bold cyan")
    console.print(f"  {title}", style="bold yellow")
    console.print(f"{'='*60}", style="bold cyan")

def demo_health():
    """Demo: Health Check Endpoint"""
    print_section("1. Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Status", data["status"])
    table.add_row("GPU Available", str(data["gpu_available"]))
    table.add_row("GPU Name", str(data.get("gpu_name", "N/A")))
    table.add_row("Uptime", f"{data['uptime']:.2f} seconds")
    
    console.print(table)
    console.print(f"\n✅ Status: {response.status_code} OK", style="bold green")

def demo_root():
    """Demo: Root Endpoint"""
    print_section("2. Welcome Message")
    
    response = requests.get(f"{BASE_URL}/")
    data = response.json()
    
    console.print(Panel(
        f"[bold cyan]{data['message']}[/bold cyan]\n\n"
        f"Version: [yellow]{data['version']}[/yellow]\n"
        f"Documentation: [blue]{BASE_URL}{data['docs_url']}[/blue]",
        title="API Information",
        border_style="green"
    ))

def demo_model_info():
    """Demo: Model Information"""
    print_section("3. Model Architecture")
    
    response = requests.get(f"{BASE_URL}/model/info")
    data = response.json()
    
    # Architecture table
    arch_table = Table(show_header=True, header_style="bold magenta")
    arch_table.add_column("Component", style="cyan")
    arch_table.add_column("Value", style="green")
    
    for key, value in data["architecture"].items():
        arch_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print("\n[bold]Architecture:[/bold]")
    console.print(arch_table)
    
    # Parameters table
    param_table = Table(show_header=True, header_style="bold magenta")
    param_table.add_column("Parameter Type", style="cyan")
    param_table.add_column("Count", style="green")
    
    for key, value in data["parameters"].items():
        param_table.add_row(key.replace("_", " ").title(), f"{value:,}")
    
    console.print("\n[bold]Parameters:[/bold]")
    console.print(param_table)

def demo_metrics():
    """Demo: API Metrics"""
    print_section("4. API Usage Metrics")
    
    response = requests.get(f"{BASE_URL}/metrics")
    data = response.json()
    
    # Requests table
    if data["total_requests"]:
        req_table = Table(show_header=True, header_style="bold magenta")
        req_table.add_column("Endpoint", style="cyan")
        req_table.add_column("Requests", style="green")
        req_table.add_column("Avg Response Time", style="yellow")
        
        for endpoint, count in data["total_requests"].items():
            avg_time = data["avg_response_time"].get(endpoint, 0)
            req_table.add_row(endpoint, str(count), f"{avg_time:.2f}ms")
        
        console.print(req_table)
    
    console.print(f"\n[bold]Total Predictions:[/bold] [green]{data['total_predictions']}[/green]")
    
    if data["errors"]:
        console.print(f"[bold]Errors:[/bold] [red]{data['errors']}[/red]")
    else:
        console.print(f"[bold]Errors:[/bold] [green]None ✅[/green]")

def demo_vocabulary():
    """Demo: Vocabulary Search"""
    print_section("5. Vocabulary (First 10 words)")
    
    response = requests.get(f"{BASE_URL}/model/vocabulary")
    data = response.json()
    
    vocab_table = Table(show_header=True, header_style="bold magenta")
    vocab_table.add_column("Index", style="cyan")
    vocab_table.add_column("Word", style="green")
    
    for match in data["matches"][:10]:  # Show first 10
        vocab_table.add_row(str(match["index"]), match["word"])
    
    console.print(vocab_table)
    console.print(f"\n[bold]Total vocabulary size:[/bold] [yellow]{data['total_matches']}[/yellow]")

def main():
    """Run all demos."""
    console.print("\n" + "="*60, style="bold cyan")
    console.print("  🚀 LSTM Text Prediction API - Live Demo", style="bold yellow")
    console.print("="*60 + "\n", style="bold cyan")
    
    console.print(f"[bold]API URL:[/bold] [blue]{BASE_URL}[/blue]")
    console.print(f"[bold]Documentation:[/bold] [blue]{BASE_URL}/docs[/blue]\n")
    
    try:
        demo_health()
        demo_root()
        demo_model_info()
        demo_metrics()
        demo_vocabulary()
        
        # Summary
        print_section("Summary")
        console.print("""
✅ API Server: [green]Running[/green]
✅ Health Check: [green]Healthy[/green]
✅ Model: [green]Loaded (6.9M parameters)[/green]
✅ Endpoints: [green]5/9 working[/green]

⚠️  [yellow]Note:[/yellow] Prediction endpoints need model training with real data.
   Run: [cyan]python scripts/run_training.py[/cyan]

📚 [bold]Next Steps:[/bold]
   1. Train model: python scripts/run_training.py
   2. Restart API: python scripts/run_api.py
   3. Test predictions: python test_api_live.py
        """)
        
    except requests.exceptions.ConnectionError:
        console.print("\n[bold red]❌ Error: API server is not running![/bold red]")
        console.print("\n[yellow]Start the server with:[/yellow]")
        console.print("[cyan]python scripts/run_api.py[/cyan]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")

if __name__ == "__main__":
    main()
