from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
import re
from itertools import combinations
from typing import List, Dict, Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# configure meditron
token = "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"
url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# making data better
def convert_to_days(turnaround_str):
    if pd.isna(turnaround_str):
        return None
    
    text = str(turnaround_str).lower().strip()
    
    # case 1: "1-2 days"
    if "days" in text:
        num = re.sub(r"[^0-9\-]", "", text)
        if "-" in num:
            low, high = num.split("-")
            return (int(low) + int(high)) / 2
        else:
            return int(num)
            
    # case 2: "24 hrs"
    if "hrs" in text or "hour" in text:
        num = re.sub(r"[^0-9]", "", text)
        return int(num) / 24
        
    # case 3: "same day"
    if "same day" in text:
        return 0.5
        
    # case 4: "2-3 weeks"
    if "weeks" in text:
        num = re.sub(r"[^0-9\-]", "", text)
        if "-" in num:
            low, high = num.split("-")
            return (int(low) + int(high)) * 7 / 2
        else:
            return int(num) * 7
            
    # case 5: "2-3 months"
    if "months" in text:
        num = re.sub(r"[^0-9\-]", "", text)
        if "-" in num:
            low, high = num.split("-")
            return (int(low) + int(high)) * 30 / 2
        else:
            return int(num) * 30
            
    # fallback
    return None

# load my excel
df = pd.read_excel("2026_NP_Prices_simple_details.xlsx")

# convert turnaround
df['Turnaround_Days'] = df['Turnaround'].apply(convert_to_days)
# lower case test name for regularity
df['Tests'] = df['Tests'].str.lower()
all_insights = sorted(df['Tests'].dropna().str.split(',').explode().str.strip().unique())

# comparison agent
def get_comparison(bundle_tests: List[dict]):
    if len(bundle_tests) < 1:
        return None
    if len(bundle_tests) == 1:
        return bundle_tests[0]['Details']

    # create context for pre-extracted details column
    context = "\n".join([f"Test {t.get('Profile Code')}: {t.get('Details', 'No details available.')}" for t in bundle_tests])
    
    prompt = (
        f"As a medical laboratory specialist, compare the following tests and explain "
        f"their combined diagnostic value in 2-3 sentences:\n\n{context}\n\nComparison:"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        return result[0]['generated_text'].split("Comparison:")[-1].strip()
    except Exception as e:
        return f"Comparison Temporarily Unavailable"

# API endpoints
@app.get("/insights")
def get_insights():
    return {"insights": all_insights}

class RecommendationRequest(BaseModel):
    selected_keywords: List[str]
    sort_by: str = 'cost'   # sorting by cost is default

@app.post("/recommend")
def recommend_tests(req: RecommendationRequest):
    selected = [k.lower() for k in req.selected_keywords]
    if not selected:
        return {"error": "No keywords selected"}
    
    required_set = set(selected)
    relevant_tests = df[df['Tests'].apply(lambda x: any(k in str(x) for k in selected))]
    valid_bundles = []
    
    max_tests = min(3, len(relevant_tests))
    for r in range(1, max_tests + 1):
        for combo in combinations(relevant_tests.to_dict('records'), r):
            combo_insights = set()
            for test in combo:
                combo_insights.update(str(test['Tests']).split(', '))
            
            if required_set.issubset(combo_insights):
                comparison = get_comparison(combo)      # generating comparison using meditron
                
                valid_bundles.append({
                    "options": " + ".join([str(t.get('Profile Code', 'N/A')) for t in combo]),
                    "total_cost": sum([t.get('DO NOT USE!      NP cost', 0) for t in combo]),
                    "insights": ", ".join(sorted(combo_insights)),
                    "agent_comparison": comparison,  # New field for your React frontend [cite: 210]
                    "turnaround_time": max([t.get('Turnaround_Days', 0) for t in combo])
                })
    if req.sort_by == "turnaround":
        # Primary: Turnaround (Fastest), Secondary: Cost (Cheapest)
        valid_bundles.sort(key=lambda x: (x['turnaround_time'], x['total_cost']))
    else:
        # Primary: Cost (Cheapest), Secondary: Turnaround (Fastest)
        valid_bundles.sort(key=lambda x: (x['total_cost'], x['turnaround_time']))
    
    return valid_bundles