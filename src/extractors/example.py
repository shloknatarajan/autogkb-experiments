"""
Example VariantExtractor implementations.

Run with:
    python -m src.eval_variants src.extractors.example.RegexExtractor
    python -m src.eval_variants src.extractors.example.OpenAIExtractor
"""

from __future__ import annotations

import json
import re

from src.variant_extractor import VariantExtractor


class RegexExtractor(VariantExtractor):
    """Baseline extractor using pattern matching — no LLM needed."""

    def extract_variants(self, paper_text: str) -> list[str]:
        variants: set[str] = set()

        # rsIDs
        variants.update(re.findall(r"\brs\d+\b", paper_text))

        # HLA alleles (e.g., HLA-B*58:01)
        variants.update(
            re.findall(
                r"\bHLA-[A-Z]+\d*\*\d+(?::\d+)*\b", paper_text
            )
        )

        # Star alleles (e.g., CYP2C19*2, CYP2B6*6)
        variants.update(
            re.findall(r"\b[A-Z][A-Z0-9]+\*\d+\b", paper_text)
        )

        # Metabolizer phenotypes
        variants.update(
            re.findall(
                r"\b(?:CYP\w+|[A-Z]{2,}\w*)\s+"
                r"(?:poor|intermediate|rapid|ultra-?rapid|normal)"
                r"\s+metabolizer\b",
                paper_text,
                re.IGNORECASE,
            )
        )

        return list(variants)


class OpenAIExtractor(VariantExtractor):
    """Example LLM-based extractor using the OpenAI API.

    Requires OPENAI_API_KEY in the environment.
    Set the model via the OPENAI_MODEL env var (default: gpt-4o).
    """

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        import os

        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    def extract_variants(self, paper_text: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a pharmacogenomics expert. Extract ALL "
                        "pharmacogenomic variants from this paper. Return "
                        "ONLY a JSON array of variant strings.\n\n"
                        f"Paper:\n{paper_text}"
                    ),
                }
            ],
        )

        text = response.choices[0].message.content or ""
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except json.JSONDecodeError:
                pass
        return []
