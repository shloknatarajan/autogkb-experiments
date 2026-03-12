"""
Variant extraction evaluation harness.

Users implement the VariantExtractor interface, then the harness evaluates
their extractor against variant_bench.jsonl ground truth.

Usage:
    # Point to a module containing a VariantExtractor subclass
    python -m src.eval_variants src.extractors.example.RegexExtractor

    # Run on a subset of papers
    python -m src.eval_variants src.extractors.example.RegexExtractor --pmcids PMC5508045

    # Evaluate from a saved predictions file instead
    python -m src.eval_variants --predictions results/predictions.jsonl

    # Save results to a custom directory
    python -m src.eval_variants src.extractors.example.RegexExtractor --output-dir results/run1

Example extractor (src/extractors/my_extractor.py):

    from src.variant_extractor import VariantExtractor

    class MyExtractor(VariantExtractor):
        def extract_variants(self, paper_text: str) -> list[str]:
            # your logic here — call an LLM, run regexes, whatever
            return ["rs9923231", "CYP2C19*2"]
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.variant_extractor import VariantExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTICLES_DIR = DATA_DIR / "articles"
VARIANT_BENCH = DATA_DIR / "variant_bench.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Variant normalization
# ---------------------------------------------------------------------------


def normalize_variant(v: str) -> str:
    """Normalize a variant string for fair comparison.

    Handles whitespace, rsID casing, HLA formatting, star allele spacing.
    """
    v = v.strip()

    # rsIDs: always lowercase
    if re.match(r"^rs\d+$", v, re.IGNORECASE):
        return v.lower()

    # HLA alleles: normalize to HLA-X*NN:NN format
    hla_match = re.match(
        r"^(HLA[-\s]?)([A-Za-z]+\d*)\s*\*\s*(\d+):?(\d+)?$", v, re.IGNORECASE
    )
    if hla_match:
        gene = hla_match.group(2).upper()
        field1 = hla_match.group(3)
        field2 = hla_match.group(4)
        if field2:
            return f"HLA-{gene}*{field1}:{field2}"
        return f"HLA-{gene}*{field1}"

    # Star alleles: collapse spaces around *
    star_match = re.match(r"^(\w+)\s*\*\s*(.+)$", v)
    if star_match:
        gene = star_match.group(1).upper()
        allele = star_match.group(2).strip()
        return f"{gene}*{allele}"

    # Metabolizer phenotypes: normalize whitespace
    if "metabolizer" in v.lower():
        return " ".join(v.split())

    return v


def normalize_variant_set(variants: list[str]) -> set[str]:
    return {normalize_variant(v) for v in variants}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_benchmark() -> dict[str, dict[str, Any]]:
    bench = {}
    with open(VARIANT_BENCH) as f:
        for line in f:
            entry = json.loads(line)
            bench[entry["pmcid"]] = entry
    return bench


def load_article(pmcid: str) -> str:
    path = ARTICLES_DIR / f"{pmcid}.md"
    if not path.exists():
        raise FileNotFoundError(f"Article not found: {path}")
    return path.read_text()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class PaperResult:
    pmcid: str
    article_title: str
    gold_variants: set[str]
    predicted_variants: set[str]
    true_positives: set[str] = field(default_factory=set)
    false_positives: set[str] = field(default_factory=set)
    false_negatives: set[str] = field(default_factory=set)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    exact_match: bool = False
    latency_s: float = 0.0
    error: str | None = None

    def compute(self):
        self.true_positives = self.gold_variants & self.predicted_variants
        self.false_positives = self.predicted_variants - self.gold_variants
        self.false_negatives = self.gold_variants - self.predicted_variants

        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)

        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1 = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )
        self.exact_match = not self.false_positives and not self.false_negatives

    def to_dict(self) -> dict:
        return {
            "pmcid": self.pmcid,
            "article_title": self.article_title,
            "gold_variants": sorted(self.gold_variants),
            "predicted_variants": sorted(self.predicted_variants),
            "true_positives": sorted(self.true_positives),
            "false_positives": sorted(self.false_positives),
            "false_negatives": sorted(self.false_negatives),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "exact_match": self.exact_match,
            "latency_s": round(self.latency_s, 2),
            "error": self.error,
        }


@dataclass
class AggregateMetrics:
    num_papers: int = 0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    exact_match_rate: float = 0.0
    total_gold: int = 0
    total_predicted: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    mean_latency_s: float = 0.0
    num_errors: int = 0
    category_recall: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_papers": self.num_papers,
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "total_gold": self.total_gold,
            "total_predicted": self.total_predicted,
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "mean_latency_s": round(self.mean_latency_s, 2),
            "num_errors": self.num_errors,
            "category_recall": {
                k: round(v, 4) for k, v in self.category_recall.items()
            },
        }


def categorize_variant(v: str) -> str:
    if re.match(r"^rs\d+$", v, re.IGNORECASE):
        return "rsid"
    if v.upper().startswith("HLA"):
        return "hla"
    if "*" in v:
        return "star_allele"
    if "metabolizer" in v.lower():
        return "metabolizer"
    return "other"


def compute_aggregate(results: list[PaperResult]) -> AggregateMetrics:
    valid = [r for r in results if r.error is None]
    agg = AggregateMetrics()
    agg.num_papers = len(results)
    agg.num_errors = len(results) - len(valid)

    if not valid:
        return agg

    agg.macro_precision = sum(r.precision for r in valid) / len(valid)
    agg.macro_recall = sum(r.recall for r in valid) / len(valid)
    agg.macro_f1 = sum(r.f1 for r in valid) / len(valid)

    agg.total_tp = sum(len(r.true_positives) for r in valid)
    agg.total_fp = sum(len(r.false_positives) for r in valid)
    agg.total_fn = sum(len(r.false_negatives) for r in valid)
    agg.total_gold = sum(len(r.gold_variants) for r in valid)
    agg.total_predicted = sum(len(r.predicted_variants) for r in valid)

    agg.micro_precision = (
        agg.total_tp / (agg.total_tp + agg.total_fp)
        if (agg.total_tp + agg.total_fp) > 0
        else 0.0
    )
    agg.micro_recall = (
        agg.total_tp / (agg.total_tp + agg.total_fn)
        if (agg.total_tp + agg.total_fn) > 0
        else 0.0
    )
    agg.micro_f1 = (
        2 * agg.micro_precision * agg.micro_recall
        / (agg.micro_precision + agg.micro_recall)
        if (agg.micro_precision + agg.micro_recall) > 0
        else 0.0
    )

    agg.exact_match_rate = sum(1 for r in valid if r.exact_match) / len(valid)
    agg.mean_latency_s = sum(r.latency_s for r in valid) / len(valid)

    # Per-category recall
    category_tp: dict[str, int] = {}
    category_total: dict[str, int] = {}
    for r in valid:
        for v in r.gold_variants:
            cat = categorize_variant(v)
            category_total[cat] = category_total.get(cat, 0) + 1
            if v in r.true_positives:
                category_tp[cat] = category_tp.get(cat, 0) + 1
    for cat in category_total:
        agg.category_recall[cat] = category_tp.get(cat, 0) / category_total[cat]

    return agg


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def evaluate_extractor(
    extractor: VariantExtractor,
    pmcids: list[str] | None = None,
) -> tuple[list[PaperResult], AggregateMetrics]:
    """Run an extractor against the benchmark and return scored results.

    Args:
        extractor: A VariantExtractor implementation to evaluate.
        pmcids: Optional subset of PMCIDs. Defaults to all 32.

    Returns:
        (per_paper_results, aggregate_metrics)
    """
    bench = load_benchmark()

    if pmcids:
        missing = [p for p in pmcids if p not in bench]
        if missing:
            print(f"Warning: PMCIDs not in benchmark: {missing}", file=sys.stderr)
        target_pmcids = [p for p in pmcids if p in bench]
    else:
        target_pmcids = list(bench.keys())

    extractor_name = type(extractor).__qualname__
    print(f"Evaluating {extractor_name} on {len(target_pmcids)} papers")

    results: list[PaperResult] = []
    for pmcid in target_pmcids:
        entry = bench[pmcid]
        gold = normalize_variant_set(entry["variants"])
        result = PaperResult(
            pmcid=pmcid,
            article_title=entry.get("article_title", ""),
            gold_variants=gold,
            predicted_variants=set(),
        )

        try:
            article_text = load_article(pmcid)
            t0 = time.monotonic()
            raw_predictions = extractor.extract_variants(article_text)
            result.latency_s = time.monotonic() - t0
            result.predicted_variants = normalize_variant_set(raw_predictions)
        except Exception as e:
            result.error = str(e)

        result.compute()
        status = "OK" if result.error is None else f"ERR: {result.error[:60]}"
        print(
            f"  {pmcid}: F1={result.f1:.3f} P={result.precision:.3f} "
            f"R={result.recall:.3f} [{status}]"
        )
        results.append(result)

    results.sort(key=lambda r: r.pmcid)
    agg = compute_aggregate(results)
    return results, agg


def evaluate_from_predictions(
    predictions_path: str,
) -> tuple[list[PaperResult], AggregateMetrics]:
    """Evaluate from a pre-saved predictions JSONL file.

    Expected format per line:
        {"pmcid": "PMC...", "predicted_variants": ["rs123", ...]}
    """
    bench = load_benchmark()
    results: list[PaperResult] = []

    with open(predictions_path) as f:
        for line in f:
            pred = json.loads(line)
            pmcid = pred["pmcid"]
            if pmcid not in bench:
                print(f"Warning: {pmcid} not in benchmark, skipping", file=sys.stderr)
                continue

            gold = normalize_variant_set(bench[pmcid]["variants"])
            predicted = normalize_variant_set(pred["predicted_variants"])

            result = PaperResult(
                pmcid=pmcid,
                article_title=bench[pmcid].get("article_title", ""),
                gold_variants=gold,
                predicted_variants=predicted,
            )
            result.compute()
            results.append(result)

    results.sort(key=lambda r: r.pmcid)
    agg = compute_aggregate(results)
    return results, agg


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[PaperResult], agg: AggregateMetrics):
    print("\n" + "=" * 80)
    print("VARIANT EXTRACTION EVALUATION REPORT")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Papers evaluated':<30} {agg.num_papers:>10}")
    print(f"{'Errors':<30} {agg.num_errors:>10}")
    print(f"{'Exact match rate':<30} {agg.exact_match_rate:>10.1%}")
    print(f"{'Macro Precision':<30} {agg.macro_precision:>10.4f}")
    print(f"{'Macro Recall':<30} {agg.macro_recall:>10.4f}")
    print(f"{'Macro F1':<30} {agg.macro_f1:>10.4f}")
    print(f"{'Micro Precision':<30} {agg.micro_precision:>10.4f}")
    print(f"{'Micro Recall':<30} {agg.micro_recall:>10.4f}")
    print(f"{'Micro F1':<30} {agg.micro_f1:>10.4f}")
    print(f"{'Total gold variants':<30} {agg.total_gold:>10}")
    print(f"{'Total predicted variants':<30} {agg.total_predicted:>10}")
    print(f"{'True positives':<30} {agg.total_tp:>10}")
    print(f"{'False positives':<30} {agg.total_fp:>10}")
    print(f"{'False negatives':<30} {agg.total_fn:>10}")
    print(f"{'Mean latency (s)':<30} {agg.mean_latency_s:>10.2f}")

    if agg.category_recall:
        print(f"\n{'Category Recall':<30}")
        print("-" * 42)
        for cat, recall in sorted(agg.category_recall.items()):
            print(f"  {cat:<28} {recall:>10.4f}")

    print(f"\nPer-Paper Results")
    print("-" * 100)
    print(
        f"{'PMCID':<18} {'P':>6} {'R':>6} {'F1':>6} {'EM':>4} "
        f"{'Gold':>5} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}"
    )
    print("-" * 100)
    for r in results:
        em_str = "Y" if r.exact_match else ""
        print(
            f"{r.pmcid:<18} {r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} "
            f"{em_str:>4} {len(r.gold_variants):>5} {len(r.predicted_variants):>5} "
            f"{len(r.true_positives):>4} {len(r.false_positives):>4} "
            f"{len(r.false_negatives):>4}"
        )

    valid = [r for r in results if r.error is None]
    if valid:
        worst = sorted(valid, key=lambda r: r.f1)[:5]
        print(f"\nBottom 5 papers by F1:")
        print("-" * 80)
        for r in worst:
            print(f"\n  {r.pmcid} (F1={r.f1:.3f})")
            if r.false_negatives:
                print(f"    Missed:   {sorted(r.false_negatives)}")
            if r.false_positives:
                print(f"    Spurious: {sorted(r.false_positives)}")


def save_results(
    results: list[PaperResult],
    agg: AggregateMetrics,
    output_dir: Path,
    run_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{run_name}_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    agg_path = output_dir / f"{run_name}_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(agg.to_dict(), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Metrics saved to: {agg_path}")


# ---------------------------------------------------------------------------
# Class loading helper
# ---------------------------------------------------------------------------


def load_extractor_class(dotted_path: str) -> type[VariantExtractor]:
    """Import and return a VariantExtractor subclass from a dotted path.

    Example: "my_extractor.MyExtractor" imports MyExtractor from my_extractor.py
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Expected 'module.ClassName', got '{dotted_path}'. "
            f"Example: src.extractors.example.RegexExtractor"
        )

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not (isinstance(cls, type) and issubclass(cls, VariantExtractor)):
        raise TypeError(
            f"{dotted_path} is not a VariantExtractor subclass. "
            f"It must extend eval_variants.VariantExtractor."
        )
    return cls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a VariantExtractor against variant_bench.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m src.eval_variants src.extractors.example.RegexExtractor
  python -m src.eval_variants src.extractors.example.RegexExtractor --pmcids PMC5508045
  python -m src.eval_variants --predictions results/predictions.jsonl
""",
    )
    parser.add_argument(
        "extractor",
        nargs="?",
        help="Dotted path to a VariantExtractor subclass (e.g. my_extractor.MyExtractor)",
    )
    parser.add_argument(
        "--predictions",
        help="Path to a predictions JSONL file (skips running an extractor)",
    )
    parser.add_argument(
        "--pmcids",
        nargs="+",
        help="Specific PMCIDs to evaluate (default: all 32)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--run-name",
        help="Name for this eval run (default: extractor class name + timestamp)",
    )
    args = parser.parse_args()

    if not args.extractor and not args.predictions:
        parser.error(
            "Provide either an extractor class path or --predictions. "
            "Run with -h for help."
        )

    if args.predictions:
        results, agg = evaluate_from_predictions(args.predictions)
        run_name = args.run_name or f"predictions_{int(time.time())}"
    else:
        # Add project root to sys.path so absolute imports (src.*) resolve
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        cls = load_extractor_class(args.extractor)
        extractor = cls()
        results, agg = evaluate_extractor(extractor, pmcids=args.pmcids)
        run_name = args.run_name or f"{cls.__name__}_{int(time.time())}"

    print_report(results, agg)
    save_results(results, agg, Path(args.output_dir), run_name)


if __name__ == "__main__":
    main()
