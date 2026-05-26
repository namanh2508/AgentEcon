#!/usr/bin/env python3
"""
Download and preprocess financial data for AgentEcon experiments.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.asset_loader import AssetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_all_assets(data_dir: str = "data") -> dict:
    """Download all required financial assets."""
    loader = AssetLoader(data_dir=data_dir)

    assets = ["XAU_USD", "WTI", "EUR_USD", "CASE_SHILLER"]
    results = {}

    for asset in assets:
        logger.info(f"Downloading {asset}...")
        try:
            df = loader.load_asset(asset, force_redownload=False)
            results[asset] = {
                "rows": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "price_range": f"{df['Close'].min():.4f} to {df['Close'].max():.4f}",
            }
            logger.info(f"  {asset}: {results[asset]}")
        except Exception as e:
            logger.error(f"Failed to download {asset}: {e}")
            results[asset] = {"error": str(e)}

    return results


def generate_all_queries(data_dir: str = "data", n_queries_per_asset: int = 100) -> dict:
    """Generate settlement queries for all assets."""
    loader = AssetLoader(data_dir=data_dir)

    seeds = [1234, 2345, 3456, 4567, 5678]
    assets = ["XAU_USD", "WTI", "EUR_USD", "CASE_SHILLER"]
    results = {}

    output_dir = Path(data_dir) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        logger.info(f"Generating queries for {asset}...")
        all_queries = []

        for seed in seeds:
            queries = loader.generate_queries(
                asset_key=asset,
                n_queries=n_queries_per_asset,
                seed=seed,
            )
            all_queries.extend(queries)

        output_file = output_dir / f"{asset}_queries.json"
        manifest_file = loader.save_queries(all_queries, str(output_file))

        results[asset] = {
            "total_queries": len(all_queries),
            "output_file": str(output_file),
            "manifest_file": manifest_file,
        }
        logger.info(f"  {asset}: {len(all_queries)} queries generated")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download and process financial data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Number of queries per asset per seed",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading data",
    )

    args = parser.parse_args()

    if not args.skip_download:
        logger.info("=== Downloading Financial Data ===")
        download_results = download_all_assets(args.data_dir)
        logger.info("Download results:")
        for asset, result in download_results.items():
            logger.info(f"  {asset}: {result}")

    logger.info("\n=== Generating Settlement Queries ===")
    query_results = generate_all_queries(args.data_dir, args.n_queries)
    logger.info("Query generation results:")
    for asset, result in query_results.items():
        logger.info(f"  {asset}: {result}")

    logger.info("\n=== Data Setup Complete ===")


if __name__ == "__main__":
    main()
