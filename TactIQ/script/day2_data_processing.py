"""
Day 2: Data Processing & Vector Database Setup
===============================================

Reproducible script to:
  • Clean and merge raw data
  • Generate embeddings
  • Populate ChromaDB vector database

Prerequisites: Day 1 data collection completed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import chromadb
from chromadb.config import Settings
from src.embeddings import EmbeddingModel
from src.database import init_chromadb
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
import json
from datetime import datetime

console = Console()


def load_and_merge_data():
    """Load and merge FBRef + Transfermarkt data"""
    console.print("\n[bold cyan]Loading raw data...[/bold cyan]")
    
    # Load FBRef data
    data_dir = Path("data/raw")
    fbref_files = list(data_dir.glob("player_stats_*.csv"))
    
    if not fbref_files:
        raise FileNotFoundError("No FBRef data found. Run day1_data_collection.py first!")
    
    console.print(f"  • Found {len(fbref_files)} FBRef files")
    
    # Merge all FBRef files
    dfs = []
    for file in fbref_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    console.print(f"  • Merged {len(merged_df):,} player records")
    
    # Clean data
    merged_df = merged_df.drop_duplicates(subset=['player', 'season', 'squad'])
    merged_df = merged_df.fillna(0)
    
    console.print(f"  • After cleaning: {len(merged_df):,} unique records")
    
    # Save processed data
    output_path = Path("data/processed/player_stats_complete.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    console.print(f"  ✅ Saved to: {output_path}\n")
    
    return merged_df


def create_player_descriptions(df):
    """Convert player stats to natural language descriptions"""
    console.print("[bold cyan]Creating player descriptions...[/bold cyan]")
    
    descriptions = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing players...", total=len(df))
        
        for _, row in df.iterrows():
            desc = f"{row['player']} ({row['position']}) played for {row['squad']} in {row['season']}. "
            
            # Add key stats based on position
            if 'goals' in row and row['goals'] > 0:
                desc += f"Scored {row['goals']} goals. "
            if 'assists' in row and row['assists'] > 0:
                desc += f"Provided {row['assists']} assists. "
            if 'minutes' in row:
                desc += f"Played {row['minutes']} minutes. "
            
            descriptions.append({
                'player': row['player'],
                'season': row['season'],
                'squad': row['squad'],
                'description': desc.strip()
            })
            
            progress.update(task, advance=1)
    
    desc_df = pd.DataFrame(descriptions)
    output_path = Path("data/processed/player_descriptions_for_embedding.csv")
    desc_df.to_csv(output_path, index=False)
    
    console.print(f"  ✅ Created {len(descriptions):,} descriptions\n")
    
    return desc_df


def ingest_to_vectordb(desc_df):
    """Generate embeddings and populate ChromaDB"""
    console.print("[bold cyan]Setting up vector database...[/bold cyan]")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="db/chroma")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection("players")
        console.print("  • Deleted existing collection")
    except:
        pass
    
    # Create new collection
    embedding_model = EmbeddingModel()
    collection = client.create_collection(
        name="players",
        metadata={"description": "Player stats and descriptions"}
    )
    
    console.print("  • Created new collection: 'players'\n")
    
    # Batch embed and add
    console.print("[bold cyan]Generating embeddings...[/bold cyan]")
    
    batch_size = 100
    total_batches = (len(desc_df) + batch_size - 1) // batch_size
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Embedding batches...", total=total_batches)
        
        for i in range(0, len(desc_df), batch_size):
            batch = desc_df.iloc[i:i+batch_size]
            
            # Generate embeddings
            texts = batch['description'].tolist()
            embeddings = embedding_model.embed_batch(texts)
            
            # Add to collection
            collection.add(
                ids=[f"player_{j}" for j in range(i, i+len(batch))],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{
                    'player': row['player'],
                    'season': row['season'],
                    'squad': row['squad']
                } for _, row in batch.iterrows()]
            )
            
            progress.update(task, advance=1)
    
    console.print(f"  ✅ Added {len(desc_df):,} embeddings to ChromaDB\n")
    
    return collection


def ingest_tactical_blogs(client):
    """Ingest tactical blog content"""
    console.print("[bold cyan]Ingesting tactical blogs...[/bold cyan]")
    
    blog_dir = Path("data/blogs")
    blog_files = list(blog_dir.glob("*.json"))
    
    if not blog_files:
        console.print("  ⚠️  No blog files found, skipping...\n", style="yellow")
        return None
    
    # Load latest blog file
    latest_blog = max(blog_files, key=lambda p: p.stat().st_mtime)
    with open(latest_blog) as f:
        blogs = json.load(f)
    
    console.print(f"  • Found {len(blogs)} articles in {latest_blog.name}")
    
    # Create collection
    try:
        client.delete_collection("tactical_blogs")
    except:
        pass
    
    collection = client.create_collection(
        name="tactical_blogs",
        metadata={"description": "Tactical analysis articles"}
    )
    
    # Chunk and embed articles
    embedding_model = EmbeddingModel()
    
    chunks = []
    for article in blogs:
        # Split into 500-word chunks
        words = article['content'].split()
        for i in range(0, len(words), 500):
            chunk = ' '.join(words[i:i+500])
            chunks.append({
                'text': chunk,
                'title': article.get('title', 'Unknown'),
                'source': article.get('source', 'Unknown'),
                'url': article.get('url', '')
            })
    
    console.print(f"  • Created {len(chunks)} chunks from articles")
    
    # Batch embed
    batch_size = 50
    with Progress() as progress:
        task = progress.add_task("[cyan]Embedding blogs...", total=(len(chunks) + batch_size - 1) // batch_size)
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c['text'] for c in batch]
            embeddings = embedding_model.embed_batch(texts)
            
            collection.add(
                ids=[f"blog_{j}" for j in range(i, i+len(batch))],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{
                    'title': c['title'],
                    'source': c['source'],
                    'url': c['url']
                } for c in batch]
            )
            
            progress.update(task, advance=1)
    
    console.print(f"  ✅ Added {len(chunks)} blog chunks to ChromaDB\n")
    
    return collection


def save_metadata(player_count, blog_count):
    """Save processing metadata"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'player_records': player_count,
        'blog_chunks': blog_count,
        'status': 'complete'
    }
    
    output_path = Path("data/processed/player_stats_metadata.json")
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"  ✅ Saved metadata: {output_path}")


def main():
    """Run complete Day 2 pipeline"""
    
    console.print(Panel.fit(
        "[bold cyan]Day 2: Data Processing & Vector DB Setup[/bold cyan]\n\n"
        "This will:\n"
        "  1. Clean and merge raw data\n"
        "  2. Generate player descriptions\n"
        "  3. Create embeddings\n"
        "  4. Populate ChromaDB\n\n"
        "Expected time: 10-20 minutes\n"
        "Output: ChromaDB at db/chroma/",
        border_style="cyan"
    ))
    
    input("\nPress ENTER to start processing (or Ctrl+C to cancel)...")
    
    try:
        # Step 1: Load and merge
        console.print("\n" + "="*80)
        console.print("[bold green]Step 1/4: Data Loading & Cleaning[/bold green]")
        console.print("="*80)
        merged_df = load_and_merge_data()
        
        # Step 2: Create descriptions
        console.print("\n" + "="*80)
        console.print("[bold green]Step 2/4: Generate Descriptions[/bold green]")
        console.print("="*80)
        desc_df = create_player_descriptions(merged_df)
        
        # Step 3: Vector DB - Players
        console.print("\n" + "="*80)
        console.print("[bold green]Step 3/4: Vector Database (Players)[/bold green]")
        console.print("="*80)
        client = chromadb.PersistentClient(path="db/chroma")
        player_collection = ingest_to_vectordb(desc_df)
        
        # Step 4: Vector DB - Blogs
        console.print("\n" + "="*80)
        console.print("[bold green]Step 4/4: Vector Database (Blogs)[/bold green]")
        console.print("="*80)
        blog_collection = ingest_tactical_blogs(client)
        
        # Save metadata
        blog_count = blog_collection.count() if blog_collection else 0
        save_metadata(len(desc_df), blog_count)
        
        # Summary
        console.print("\n" + "="*80)
        console.print("[bold cyan]DAY 2 COMPLETE![/bold cyan]")
        console.print("="*80 + "\n")
        
        console.print("✅ Vector database ready:")
        console.print(f"   • Players: {len(desc_df):,} records")
        console.print(f"   • Blogs: {blog_count:,} chunks")
        console.print(f"   • Location: db/chroma/\n")
        
        console.print("[bold yellow]Next Step:[/bold yellow]")
        console.print("   Run: python script/day3_crag_query.py\n")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}", style="bold red")
        console.print("\nCheck that Day 1 data collection completed successfully.")
        raise


if __name__ == "__main__":
    main()
