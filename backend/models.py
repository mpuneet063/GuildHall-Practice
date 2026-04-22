from pydantic import BaseModel, Field
from typing import List, Literal

# Pydantic Models for Type Safety

# 1. The structure of the data we expect from the frontend
class SymptomRequest(BaseModel):
    """
    expected JSON payload when a user searches by symptoms/biomarkers.
    Example: {"biomarkers": ["cholestrol", "blood sugar", "thyroid"]}
    """
    biomarkers: List[str] = Field(
        ...,
        description="List of biomarkers or symptoms to search for",
    )
    sort_by: Literal["cost", "turnaround"] = Field(
        "cost",
        description="Sort order for results (cost-optimal or fastest)"
    )

# 2. The structure of the data we will return to the frontend
class TestDetail(BaseModel):
    """
    Represents a single test or biomarker in the bundle.
    """
    test_name: str
    price: float
    turnaround_time: str
    description: str

class BundleResult(BaseModel):
    """
    Structure for a single test result.
    """
    bundle_name: str
    tests_included: List[TestDetail]
    total_price: float
    total_turnaround: float
    total_biomarkers_covered: int
    covers: List[str] = Field(default=[], description="Biomarkers successfully covered by this bundle")
    misses: List[str] = Field(default=[], description="Biomarkers not covered by this bundle")
    extra: List[str] = Field(default=[], description="Extra biomarkers included in this bundle but not requested")

class RecommendationResponse(BaseModel):
    """
    The final JSON payload sent back to the frontend.
    Contains the bundles and the Gemini AI clinical comparison.
    """
    bundles: List[BundleResult]
    ai_comparison: str = Field(
        ...,
        description="""
        The clinical comparison written by Gemini.
        This should be a concise, human-readable analysis comparing the bundles.
        Focus on value (cost vs. included tests vs. extra tests).
        """
        )
    status: str = "success"

