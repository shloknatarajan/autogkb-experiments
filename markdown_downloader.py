"""Download PMC articles as markdown for all PMCIDs in benchmark data."""

import json
import os
import time
from pathlib import Path

from pubmed_downloader import PubMedDownloader
from dotenv import load_dotenv

load_dotenv()

def collect_pmcids() -> set[str]:
    """Collect unique PMCIDs from benchmark_annotations, sentence_bench, summary_bench, and variant_bench."""
    pmcids = set()
    data_dir = Path("data")

    # From benchmark_annotations (filenames are PMCIDs)
    annotations_dir = data_dir / "benchmark_annotations"
    for f in annotations_dir.glob("*.json"):
        pmcids.add(f.stem)

    # From JSONL bench files
    for bench_file in ["sentence_bench.jsonl", "summary_bench.jsonl", "variant_bench.jsonl"]:
        path = data_dir / bench_file
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "pmcid" in record:
                    pmcids.add(record["pmcid"])

    return pmcids


def main():
    pmcids = sorted(collect_pmcids())
    print(f"Found {len(pmcids)} unique PMCIDs to download")

    out_dir = Path("data/articles")
    out_dir.mkdir(parents=True, exist_ok=True)

    downloader = PubMedDownloader()
    failed = []

    for i, pmcid in enumerate(pmcids, 1):
        out_path = out_dir / f"{pmcid}.md"
        if out_path.exists():
            print(f"[{i}/{len(pmcids)}] {pmcid} — already exists, skipping")
            continue

        print(f"[{i}/{len(pmcids)}] Downloading {pmcid}...")
        try:
            markdown = downloader.single_pmcid_to_markdown(pmcid)
            out_path.write_text(markdown, encoding="utf-8")
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(pmcid)
            continue

        # Rate limit: NCBI allows 3 req/sec without API key, 10/sec with.
        # Each article makes ~2 requests (efetch + BioC supplement), so
        # sleep 1s between articles to stay well under the limit.
        time.sleep(1)

    print(f"\nDone. Downloaded {len(pmcids) - len(failed)}/{len(pmcids)} articles.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
