from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# import Pydantic models
from models import SymptomRequest, RecommendationResponse
# import services
from services import (
    get_unique_biomarkers,
    find_optimal_bundles,
    generate_clinical_comparison,
)
from typing import List


app = FastAPI(
    title="GuildHall test finder API",
    description="Finds optimal bundles of lab tests to meet clinical needs at the lowest cost and fastest turnaround.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://localhost:3000",
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# creating endpoints
@app.get("/")
def health_check():
    return {"message": "GuildHall test finder API is running"}

@app.get("/metadata", response_model=List[str])
def get_metadata():
    """
    Scans the database and returns a list of all unique biomarkers.
    Used to populate the frontend search bar with suggestions.
    """
    biomarkers = get_unique_biomarkers()
    if not biomarkers:
        raise HTTPException(status_code=500, detail="Could not retrieve biomarkers from database")
    return biomarkers

@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: SymptomRequest):
    """
    The core endpoint. Receives a list of biomarkers and returns:
    1. The mathematically optimal set cover solutions.
    2. The clinical comparison from Gemini.
    """
    # 1. pass the input biomarkers to find_optimal_bundles
    bundles, is_partial = find_optimal_bundles(request.biomarkers, sort_by = request.sort_by)

    if not bundles:
        raise HTTPException(status_code=404, detail="Could not find any bundles that match your requirements.")
    
    # 2. generate the clinical comparison for top 3 bundles
    top_3 = bundles[:3]

    # 3. Call the AI Engine
    try:
        ai_summary = generate_clinical_comparison(
            top_3, 
            request.biomarkers, 
            sort_by=request.sort_by
        ) 
    except Exception as e:
        print(f"AI Failure: {e}")
        ai_summary = "Clinical comparison temporarily unavailable."

    # 4. Return response
    return RecommendationResponse(
        bundles=bundles,
        ai_comparison=ai_summary,
        status="partial match" if is_partial else "perfect match"
    )