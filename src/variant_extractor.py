"""
VariantExtractor interface.

Subclass this and implement `extract_variants` to create an extractor
that can be evaluated by eval_variants.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VariantExtractor(ABC):
    """Interface that evaluation subjects must implement.

    Subclass this and implement `extract_variants`. The harness will call it
    once per paper with the full markdown text, and compare the returned
    variant list against ground truth.
    """

    @abstractmethod
    def extract_variants(self, paper_text: str) -> list[str]:
        """Extract pharmacogenomic variants from a paper's markdown text.

        Args:
            paper_text: Full markdown content of the research paper.

        Returns:
            List of variant strings. Accepted formats:
                - rsIDs:                "rs9923231"
                - Star alleles:         "CYP2C19*2"
                - HLA alleles:          "HLA-B*58:01"
                - Metabolizer types:    "CYP2D6 poor metabolizer"
        """
        ...
