#!/usr/bin/env python3
"""Prefetch structured news contexts for generated settlement queries."""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.asset_loader import AssetLoader
from src.data.news_loader import NewsLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download/cache GDELT news contexts")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--assets", nargs="+", default=["XAU_USD", "WTI", "EUR_USD", "CASE_SHILLER"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234])
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--force", action="store_true", help="Refetch and rewrite cached contexts")
    args = parser.parse_args()

    news_loader = NewsLoader(
        data_dir=args.data_dir,
        max_items=args.max_items,
        lookback_days=args.lookback_days,
    )
    loader = AssetLoader(
        data_dir=args.data_dir,
        news_enabled=True,
        news_loader=news_loader,
    )

    manifest_rows = []
    for asset in args.assets:
        for seed in args.seeds:
            queries = loader.generate_queries(asset, n_queries=args.n_queries, seed=seed)
            for query in queries:
                if args.force:
                    context = news_loader.get_context(
                        asset=asset,
                        query_id=query.query_id,
                        query_date=query.timestamp,
                        force=True,
                    )
                    query.news_context_hash = context.context_hash
                    query.news_source = context.source
                    query.news_window = f"{context.window_start}:{context.window_end}"
                    query.n_news_items = context.n_items
                manifest_rows.append(
                    {
                        "asset": asset,
                        "seed": seed,
                        "query_id": query.query_id,
                        "timestamp": query.timestamp.isoformat(),
                        "news_source": query.news_source,
                        "news_window": query.news_window,
                        "n_news_items": query.n_news_items,
                        "news_context_hash": query.news_context_hash,
                    }
                )

    output_path = Path(args.data_dir) / "processed" / "news_contexts_manifest.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_contexts": len(manifest_rows),
                "rows": manifest_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("News context manifest saved to %s", output_path)


if __name__ == "__main__":
    main()
