import os
import pandas as pd
import google.generativeai as genai
import numpy as np
from typing import List, Tuple
from itertools import combinations
import re

from models import BundleResult, TestDetail
from dotenv import load_dotenv

# ==========================================
# 1. INITIALIZATION & DATA LOADING
# ==========================================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- THE BULLETPROOF PATH FIX ---
# 1. Find exactly where services.py lives on the computer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Join it with the data folder and file name
FILE_PATH = os.path.join(BASE_DIR, "data", "2026_NP_Prices_simple_details.xlsx")
# --------------------------------

# helper functions
def convert_to_days(turnaround_str: str) -> float:
    """Helper function to normalize turnaround times to days"""
    if pd.isna(turnaround_str):
        return None
    text = str(turnaround_str).lower().strip()
    if "same day" in text or "1 day" in text:
        return 1
    if "hrs" in text or "hours" in text:
        num = re.sub(r"[^0-9]", "", text)
        return int(num) / 24 if num.isdigit() else None
    if "days" in text:
        num = re.sub(r"[^0-9\-]", "", text)
        if "-" in num:
            parts = num.split("-")
            return (int(parts[0]) + int(parts[1])) / 2
        return int(num) if num.isdigit() else None
    if "weeks" in text:
        num = re.sub(r"[^0-9\-]", "", text)
        if "-" in num:
            parts = num.split("-")
            return (int(parts[0]) + int(parts[1])) * 7 / 2
        return int(num) * 7 if num.isdigit() else None

    if "months" in text:
        num = re.sub(r"[^0-9\-\"]", "", text)
        if "-" in num:
            parts = num.split("-")
            return (int(parts[0]) + int(parts[1])) * 30 / 2
        return int(num) * 30 if num.isdigit() else None 

    return None


# load dataset
file_path = "data/2026_NP_Prices_simple_details.xlsx"
try:
    df = pd.read_excel(file_path)
    # clean and prepare data 
    # 1. Fix misaligned 'Tests' column
    df['Tests'] = df['Tests'].shift(-1)
    # 2. Convert turnaround times to days
    df['Turnaround_Days'] = df['Turnaround'].apply(convert_to_days)
    # 3. Clean and normalize test names
    df['Tests'] = df['Tests'].str.lower().str.strip()
    # 4. Pre-compute keyword sets for faster searching
    df['_keyword_set'] = df['Tests'].apply(
        lambda x: set(k.strip() for k in x.split(',')) if pd.notna(x) else set()
    )

    print("Data loaded and prepared successfully.")
except Exception as e:
    print(f"Error loading file: {e}")


# ==============    
# 2. Set Cover Implementation
# ==============    

def get_unique_biomarkers():
    if df.empty:
        return []
    # returns the master list of all unique biomarkers for the frontend UI search bar
    all_insights = sorted(
        df['Tests'].dropna().str.split(',').explode().str.strip().unique()
    )
    return [i for i in all_insights if i]

def find_optimal_bundles(target_biomarkers: List[str], sort_by: str = 'cost') -> Tuple[List[BundleResult], bool]:
    """Runs the Set Cover math and returns strictly typed Pydantic objects."""
    if not target_biomarkers or df.empty:
        return [], False

    required_set = set(target_biomarkers)
    n_required = len(required_set)

    # 1. Filter relevant tests
    relevant_tests = df[df['_keyword_set'].apply(lambda s: bool(s & required_set))].copy()
    if relevant_tests.empty:
        return [], False

    records = relevant_tests.to_dict('records')

    # 2. Prune search space for speed
    if len(records) > 45:
        pruned_records = []
        seen_codes = set()
        for req in required_set:
            req_records = [r for r in records if req in r['_keyword_set']]
            cheap_reqs = sorted(req_records, key=lambda x: x.get('Lab Fee', 9999))[:15]
            fast_reqs = sorted(req_records, key=lambda x: x.get('Turnaround_Days', 999) or 999)[:15]
            
            for r in cheap_reqs + fast_reqs:
                code = r.get('test code', id(r))
                if code not in seen_codes:
                    seen_codes.add(code)
                    pruned_records.append(r)
        records = pruned_records

    valid_bundles_raw = []
    is_partial = False

    def make_raw_bundle(combo):
        combo_insights = set().union(*(t['_keyword_set'] for t in combo))
        covered = combo_insights & required_set
        missed = required_set - combo_insights
        extra = combo_insights - required_set
        return {
            "name": " + ".join([str(t.get('test code', 'N/A')) for t in combo]),
            "cost": sum([t.get('Lab Fee', 0) for t in combo]),
            "turnaround": max([t.get('Turnaround_Days', 0) or 0 for t in combo]),
            "combo_records": list(combo),
            "n_covered": len(covered),
            "covers": list(covered),
            "misses": list(missed),
            "extras": list(extra)
        }

    # 3. Combinatorial Math
    solo_full_ids = set()
    for t in records:
        if required_set.issubset(t['_keyword_set']):
            solo_full_ids.add(t.get('test code'))
            valid_bundles_raw.append(make_raw_bundle((t,)))

    non_solo_records = [t for t in records if t.get('test code') not in solo_full_ids]
    for combo in combinations(non_solo_records, 2):
        if required_set.issubset(set().union(*(t['_keyword_set'] for t in combo))):
            valid_bundles_raw.append(make_raw_bundle(combo))

    for combo in combinations(non_solo_records, 3):
        if required_set.issubset(set().union(*(t['_keyword_set'] for t in combo))):
            sub_already_covers = any(
                required_set.issubset(set().union(*(s['_keyword_set'] for s in sub)))
                for sub in combinations(combo, 2)
            )
            if not sub_already_covers:
                valid_bundles_raw.append(make_raw_bundle(combo))

    # 4. Fallback for Partial Matches
    if not valid_bundles_raw:
        is_partial = True
        for r in range(1, 4):
            for combo in combinations(records, r):
                combo_insights = set().union(*(t['_keyword_set'] for t in combo))
                covered = combo_insights & required_set
                if len(covered) / n_required >= 0.8:
                    valid_bundles_raw.append(make_raw_bundle(combo))

    if not valid_bundles_raw:
        return [], False

    # 5. Sorting
    if sort_by == 'turnaround':
        valid_bundles_raw.sort(key=lambda x: (-x['n_covered'], x['turnaround'], x['cost']))
    else:
        valid_bundles_raw.sort(key=lambda x: (-x['n_covered'], x['cost'], x['turnaround']))

    # 6. MAP TO PYDANTIC MODELS (The API Contract)
    final_bundles = []
    for b in valid_bundles_raw[:10]:
        test_details = []
        for t in b['combo_records']:
            test_details.append(TestDetail(
                test_name=str(t.get('Test Name', t.get('test code', 'Unknown Test'))),
                price=float(t.get('Lab Fee', 0.0)),
                turnaround_time=str(t.get('Turnaround', 'N/A')),
                description=str(t.get('Details', 'No description available.'))
            ))
            
        final_bundles.append(BundleResult(
            bundle_name=b['name'],
            tests_included=test_details,
            total_price=float(b['cost']),
            total_turnaround=float(b['turnaround']),
            total_biomarkers_covered=int(b['n_covered']),
            covers=b['covers'],
            misses=b['misses'],
            extra=b['extras']
        ))

    return final_bundles, is_partial
# ==========================================
# 3. THE AI INTEGRATION
# ==========================================

def generate_clinical_comparison(bundles: List[BundleResult], target_biomarkers: List[str], sort_by: str = 'cost') -> str:
    """Takes the calculated bundles and asks Gemini 2.5 to compare them clinically."""
    if not bundles:
        return "No combinations available to compare."
        
    # 1. Build the context blocks exactly like your Streamlit app
    context_blocks = []
    for i, bundle in enumerate(bundles):
        context_blocks.append(
            f"Option {i+1} ({bundle.bundle_name}):\n"
            f"- Cost: £{bundle.total_price} | Turnaround: {bundle.tests_included[0].turnaround_time} days\n"
            f"- Hits: {', '.join(bundle.covers)}\n"
            f"- Misses: {', '.join(bundle.misses) if bundle.misses else 'None'}\n"
            f"- Extra (Bonus Biomarkers): {', '.join(bundle.extra) if bundle.extra else 'None'}"
        )
        
    context_str = "\n\n".join(context_blocks)
    priority = "Turnaround Time (speed of results)" if sort_by == "turnaround" else "Cost-effectiveness (lowest price)"
    
    # 2. Configure the Gemini 2.5 Flash model with your specific system instructions
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction="You are a clinical logistics advisor. Your job is to analyze diagnostic test bundles and help a patient choose the best option based on their specific priorities. Never just define the tests; analyze the trade-offs.",
        generation_config=genai.types.GenerationConfig(
            temperature=0.2, 
            max_output_tokens=3000
        )
    )
    
    prompt = (
        f"The patient needs to test for: {', '.join(target_biomarkers)}.\n"
        f"They have sorted their search to prioritize: **{priority}**.\n\n"
        f"Here is the raw data for the top {len(bundles)} combinations our algorithm generated:\n"
        f"{context_str}\n\n"
        f"Write EXACTLY three bullet points comparing these options. "
        f"Bullet 1: Analyze the top-ranked option and why it wins based on their priority.\n"
        f"Bullet 2: Contrast it with the other options regarding what biomarkers are missed, OR what Extra (bonus) biomarkers are gained for the price difference.\n"
        f"Bullet 3: Provide a definitive clinical/logistical recommendation.\n"
        f"Keep the tone professional, objective, and strictly limit the output to these three bullet points."
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Comparison currently unavailable."