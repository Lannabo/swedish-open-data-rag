#!/usr/bin/env python3
import logging
import time
import torch
import os
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import track
from rag_pipeline import DataportalRAG

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("test_rag")
console = Console()

def test_gpu_availability():
    """Verify GPU is available and print capabilities."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! This test requires GPU support.\n"
            "Please ensure NVIDIA drivers and CUDA 11.8+ are installed."
        )
    
    device_name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
    cuda_version = torch.version.cuda
    
    console.print(Panel.fit(
        f"[green]GPU Device: {device_name}\n"
        f"CUDA Version: {cuda_version}\n"
        f"Total Memory: {memory:.2f}MB",
        title="GPU Information"
    ))

def run_sample_queries(rag: DataportalRAG):
    """Run a set of sample queries to test the system."""
    test_queries = [
        "Vilka luftkvalitetsmätningar finns tillgängliga för Stockholm?",
        "Hur många kommuner delar öppna data om kollektivtrafik?",
        "Finns det data om energiförbrukning i svenska städer?",
    ]
    
    results = []
    for query in track(test_queries, description="Processing queries..."):
        start_time = time.time()
        try:
            response = rag.query(query)
            duration = time.time() - start_time
            results.append({
                "query": query,
                "response": response,
                "duration": duration
            })
        except Exception as e:
            logger.error(f"Query failed: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "duration": time.time() - start_time
            })
    
    return results

def print_results(results):
    """Print query results in a formatted way."""
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold cyan]Test Query {i}[/bold cyan]")
        console.print(Panel(
            f"[yellow]Query:[/yellow] {result['query']}\n\n"
            f"[green]Response:[/green] {result.get('response', 'Error: ' + result.get('error', 'Unknown error'))}\n\n"
            f"[blue]Duration:[/blue] {result['duration']:.2f} seconds",
            title=f"Test {i} Results",
            expand=False
        ))

def main():
    """Run main test sequence."""
    console.print("[bold]Starting RAG Pipeline Test[/bold]\n")
    
    # Check for HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    
    try:
        # Check GPU
        console.print("\n[bold yellow]Step 1: Checking GPU Configuration...[/bold yellow]")
        test_gpu_availability()
        
        # Initialize RAG system
        console.print("\n[bold yellow]Step 2: Initializing RAG System...[/bold yellow]")
        console.print("[dim]- Loading models (this may take a few minutes)")
        console.print("[dim]- Initializing database")
        console.print("[dim]- Setting up FAISS index")
        
        rag = DataportalRAG(
            db_path="data/rag_database.db",
            index_path="data/document_index.faiss",
            hf_token=hf_token,
            base_url="https://dataportal.se",  # Remove www.
            cache_size=100,
            batch_size=16
        )
        
        # Update knowledge base with small dataset
        console.print("\n[bold yellow]Step 3: Updating Knowledge Base...[/bold yellow]")
        with console.status("[bold green]Fetching initial datasets from dataportal.se...") as status:
            try:
                rag.update_knowledge_base(limit=10)  # Start with small dataset for testing
                console.print("[green]✓ Successfully loaded initial datasets[/green]")
            except Exception as e:
                console.print(f"[bold red]Failed to fetch datasets: {str(e)}[/bold red]")
                raise
        
        # Run test queries
        console.print("\n[bold yellow]Step 4: Running Test Queries...[/bold yellow]")
        try:
            results = run_sample_queries(rag)
            console.print("[green]✓ Successfully completed test queries[/green]")
        except Exception as e:
            console.print(f"[bold red]Query testing failed: {str(e)}[/bold red]")
            raise
        
        # Print results
        console.print("\n[bold yellow]Step 5: Displaying Results...[/bold yellow]")
        print_results(results)
        
        # Print performance summary
        total_duration = sum(r['duration'] for r in results)
        avg_duration = total_duration / len(results)
        
        console.print(Panel(
            f"[green]Total Queries:[/green] {len(results)}\n"
            f"[yellow]Average Response Time:[/yellow] {avg_duration:.2f} seconds\n"
            f"[blue]Total Test Duration:[/blue] {total_duration:.2f} seconds",
            title="Performance Summary"
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[red]Error Details:[/red] {str(e)}\n\n"
            "[yellow]Troubleshooting Steps:[/yellow]\n"
            "1. Check GPU availability and CUDA version\n"
            "2. Verify HuggingFace token is valid\n"
            "3. Ensure network connection to dataportal.se\n"
            "4. Check disk space for database storage",
            title="[red]Test Failed[/red]",
            border_style="red"
        ))
        raise
    finally:
        # Cleanup
        if 'rag' in locals():
            console.print("\n[dim]Cleaning up resources...[/dim]")
            rag.cleanup()
            
    console.print("\n[bold green]✨ Test completed successfully! ✨[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Test interrupted by user[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Test failed with error: {e}[/bold red]")
        raise 