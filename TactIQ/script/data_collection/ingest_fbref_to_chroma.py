"""
Ingest FBref CSV outputs into ChromaDB with rich metadata per player-season.

Usage:
    python ingest_fbref_to_chroma.py --data-dir ./data/raw --collection player_stats

This script expects CSV files saved by `FBrefScraper.fetch_player_stats` named like:
  player_standard_2024_2025.csv
  player_shooting_2024_2025.csv
  player_passing_2024_2025.csv

It will merge available stat tables on (player, team, season) and create one document per player-season
with metadata that includes numeric stat fields (goals, assists, xg, xa, key_passes, passes_completed, touches_per90, pressures, etc.).
"""

import argparse
import glob
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from src.database import VectorDatabase


def load_stat_tables(data_dir: str):
    files = glob.glob(os.path.join(data_dir, 'player_*_*.csv'))
    tables = {}
    for f in files:
        name = Path(f).stem  # e.g., player_standard_2024_2025
        parts = name.split('_')
        if len(parts) >= 3:
            stat_type = parts[1]
            season = '_'.join(parts[2:])
            try:
                df = pd.read_csv(f)
                df['season'] = season.replace('_', '-')
                tables.setdefault(stat_type, []).append(df)
                logger.info(f'Loaded {len(df)} rows from {f}')
            except Exception as e:
                logger.warning(f'Could not read {f}: {e}')
    # Concatenate per stat_type across seasons
    stat_dfs = {}
    for stype, dfs in tables.items():
        stat_dfs[stype] = pd.concat(dfs, ignore_index=True)
    return stat_dfs


def build_player_season_profiles(stat_dfs: dict) -> pd.DataFrame:
    """Merge available stat tables into a single DataFrame keyed by player/team/season."""
    # Start with 'standard' if available
    if 'standard' in stat_dfs:
        base = stat_dfs['standard'].copy()
    else:
        # pick any available as base
        base = None
        for df in stat_dfs.values():
            base = df.copy()
            break
        if base is None:
            return pd.DataFrame()

    # Normalize column names (lowercase)
    base.columns = [c.lower() for c in base.columns]

    key_cols = []
    for col in ['player', 'team', 'season']:
        if col in base.columns:
            key_cols.append(col)
    if not key_cols:
        raise RuntimeError('Base table does not contain player/team/season columns')

    merged = base
    # Merge other stat types (shooting/passing/defense/keeper)
    for stype, df in stat_dfs.items():
        if stype == 'standard':
            continue
        df.columns = [c.lower() for c in df.columns]
        try:
            merged = pd.merge(
                merged,
                df,
                on=['player', 'team', 'season'],
                how='left',
                suffixes=('', f'_{stype}')
            )
        except Exception:
            # fallback: try merging on player+season only
            merged = pd.merge(
                merged,
                df,
                on=['player', 'season'],
                how='left',
                suffixes=('', f'_{stype}')
            )
    return merged


def row_to_document(row: pd.Series) -> (str, dict, str):
    """Convert a DataFrame row into (document_text, metadata, id)"""
    meta = {}
    # Select common numeric columns if present
    numeric_fields = [
        'goals', 'assists', 'xg', 'xa', 'xag', 'key_passes', 'passes_completed', 'passes',
        'touches_per90', 'pressures', 'minutes', 'shots', 'shots_on_target'
    ]
    for f in numeric_fields:
        if f in row.index and pd.notna(row[f]):
            try:
                meta[f] = float(row[f]) if not pd.isna(row[f]) else ''
            except Exception:
                meta[f] = row[f]

    # Add identifiers
    meta['player'] = row.get('player', '')
    meta['team'] = row.get('team', '')
    meta['season'] = row.get('season', '')

    # Build document text summary
    parts = [f"Player: {meta.get('player')}", f"Team: {meta.get('team')}", f"Season: {meta.get('season')}"]
    for k in ['goals', 'assists', 'xg', 'xa', 'key_passes', 'passes_completed', 'touches_per90']:
        if k in meta:
            parts.append(f"{k}: {meta[k]}")
    doc_text = '\n'.join(parts)

    doc_id = f"player_{meta.get('player','').replace(' ','_')}_{meta.get('season','')}_{meta.get('team','').replace(' ','_')}"
    return doc_text, meta, doc_id


def main(data_dir: str, collection_name: str):
    stat_dfs = load_stat_tables(data_dir)
    if not stat_dfs:
        logger.error('No FBref stat CSVs found in data_dir')
        return

    merged = build_player_season_profiles(stat_dfs)
    if merged.empty:
        logger.error('Merged DataFrame is empty')
        return

    db = VectorDatabase(persist_directory='./db/chroma', collection_name=collection_name)

    documents = []
    metadatas = []
    ids = []

    for _, row in merged.iterrows():
        doc_text, meta, doc_id = row_to_document(row)
        documents.append(doc_text)
        metadatas.append(meta)
        ids.append(doc_id)

    logger.info(f'Preparing to add {len(documents)} documents to collection {collection_name}')
    db.add_documents_batch(documents, metadatas, ids, batch_size=200)
    logger.info('Ingestion complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/raw', help='Directory containing FBref CSV outputs')
    parser.add_argument('--collection', type=str, default='player_stats', help='Chroma collection name')
    args = parser.parse_args()
    main(args.data_dir, args.collection)
