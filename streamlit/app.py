import streamlit as st
import pandas as pd
import requests
import re
from itertools import combinations
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. HuggingFace Configuration
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN")
BASE_URL = "https://api-inference.huggingface.co/models"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_URL = f"{BASE_URL}/{MODEL_ID}"
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


# ==========================================
# 2. Data Loading & Pre-processing (Cached)
# ==========================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_excel("2026_NP_Prices_simple_details.xlsx")

    def convert_to_days(turnaround_str):
        if pd.isna(turnaround_str): 
            return None
        
        text = str(turnaround_str).lower().strip()

        # 1. Handle "Same Day" or "1 day" explicitly
        if "same day" in text or "1 day" in text:
            return 0 # Using 0.5 for same-day parity
        
        # 2. Check for Hours first (smallest unit)
        if "hrs" in text or "hour" in text:
            num = re.sub(r"[^0-9]", "", text)
            return int(num) / 24 if num.isdigit() else None

        # 3. Check for Days (Range or Single)
        if "days" in text:
            num = re.sub(r"[^0-9\-]", "", text)
            if "-" in num:
                parts = num.split("-")
                return (int(parts[0]) + int(parts[1])) / 2
            return int(num) if num.isdigit() else None

        # 4. Check for Weeks
        if "weeks" in text:
            num = re.sub(r"[^0-9\-]", "", text)
            if "-" in num:
                parts = num.split("-")
                return (int(parts[0]) + int(parts[1])) * 7 / 2
            return int(num) * 7 if num.isdigit() else None
            
        # 5. Check for Months
        if "months" in text:
            num = re.sub(r"[^0-9\-]", "", text)
            if "-" in num:
                parts = num.split("-")
                return (int(parts[0]) + int(parts[1])) * 30 / 2
            return int(num) * 30 if num.isdigit() else None
                
        return None

    df['Turnaround_Days'] = df['Turnaround'].apply(convert_to_days)
    df['Tests'] = df['Tests'].str.lower()

    # Pre-compute keyword sets once per row — avoids repeated splitting in search loop
    df['_keyword_set'] = df['Tests'].apply(
        lambda x: set(k.strip() for k in x.split(',')) if pd.notna(x) else set()
    )

    all_insights = sorted(df['Tests'].dropna().str.split(',').explode().str.strip().unique())
    all_insights = [i for i in all_insights if i]

    return df, all_insights


df, all_insights = load_and_clean_data()


# ==========================================
# 3. AI Comparison Agent
# ==========================================
# TODO: Replace this function body with Google ADK call
def get_comparison(bundle_tests: list):
    if not bundle_tests:
        return None

    # Single test — no LLM needed, just return its details directly
    if len(bundle_tests) == 1:
        return bundle_tests[0].get('Details', 'No details available.')

    context = "\n".join([
        f"Test {t.get('test code')}: {t.get('Details', 'No details available.')}"
        for t in bundle_tests
    ])
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
        print(f"DEBUG: Target URL is {HF_URL}")
        response = requests.post(HF_URL, headers=HF_HEADERS, json=payload)
        result = response.json()
        return result[0]['generated_text'].split("Comparison:")[-1].strip()
    except Exception as e:
        return f"Debug — error: {e} | Raw response: {response.text if 'response' in dir() else 'no response'}"


# ==========================================
# 4. Residual-Pruned Set Cover Search
# ==========================================
def find_test_combinations(selected_keywords, sort_by='cost'):
    if not selected_keywords:
        return None, [], False

    required_set = set(selected_keywords)
    n_required = len(required_set)

    # Only keep tests that contribute at least one required keyword
    relevant_tests = df[df['_keyword_set'].apply(lambda s: bool(s & required_set))].copy()

    if relevant_tests.empty:
        return pd.DataFrame(), [], False

    records = relevant_tests.to_dict('records')  # Convert once, outside the loop
    valid_bundles = []
    is_partial = False

    def make_bundle(combo):
        combo_insights = set().union(*(t['_keyword_set'] for t in combo))
        covered = combo_insights & required_set
        missed  = required_set - combo_insights
        extra   = combo_insights - required_set
        return {
            "Available Options":   " + ".join([str(t.get('test code', 'N/A')) for t in combo]),
            "Total Cost (in GBP)": sum([t.get('Lab Fee', 0) for t in combo]),
            "Turnaround (Days)":   max([t.get('Turnaround_Days', 0) or 0 for t in combo]),
            "Covers":              ", ".join(sorted(covered)),
            "Misses":              ", ".join(sorted(missed)) if missed else "—",
            "Extra":               ", ".join(sorted(extra)) if extra else "—",
            "_combo_records":      list(combo),
            "_n_covered":          len(covered)
        }

    # --- r=1: collect all single tests that cover 100% of required biomarkers ---
    solo_full_ids = set()
    for t in records:
        if required_set.issubset(t['_keyword_set']):
            solo_full_ids.add(t.get('test code'))
            valid_bundles.append(make_bundle((t,)))

    # --- r=2: only pair tests where NEITHER is already a solo 100% solution ---
    # Rationale: if NP22 alone covers glucose+cholesterol, pairing NP22 with anything
    # else is redundant and misleading — it implies a second test is needed when it isn't.
    non_solo_records = [t for t in records if t.get('test code') not in solo_full_ids]
    for combo in combinations(non_solo_records, 2):
        combo_insights = set().union(*(t['_keyword_set'] for t in combo))
        if required_set.issubset(combo_insights):
            valid_bundles.append(make_bundle(combo))

    # --- r=3: only form triples from non-solo tests,
    #          and skip any triple whose 2-test sub-combo already covers 100% ---
    for combo in combinations(non_solo_records, 3):
        combo_insights = set().union(*(t['_keyword_set'] for t in combo))
        if required_set.issubset(combo_insights):
            # Prune: if any pair within this triple already covers 100%, skip
            sub_already_covers = any(
                required_set.issubset(set().union(*(s['_keyword_set'] for s in sub))
                )
                for sub in combinations(combo, 2)
            )
            if not sub_already_covers:
                valid_bundles.append(make_bundle(combo))

    # --- Fallback: nothing at 100% — collect partials >= 80% ---
    if not valid_bundles:
        is_partial = True
        for r in range(1, 4):
            for combo in combinations(records, r):
                combo_insights = set().union(*(t['_keyword_set'] for t in combo))
                covered = combo_insights & required_set
                if len(covered) / n_required >= 0.8:
                    valid_bundles.append(make_bundle(combo))

    if not valid_bundles:
        return pd.DataFrame(), [], False

    # Sort: most covered descending as primary, then cost or turnaround as tiebreaker
    if sort_by == 'turnaround':
        valid_bundles.sort(key=lambda x: (-x['_n_covered'], x['Turnaround (Days)'], x['Total Cost (in GBP)']))
    else:
        valid_bundles.sort(key=lambda x: (-x['_n_covered'], x['Total Cost (in GBP)'], x['Turnaround (Days)']))

    combo_records_list = [b.pop('_combo_records') for b in valid_bundles]
    for b in valid_bundles:
        b.pop('_n_covered')

    return pd.DataFrame(valid_bundles).reset_index(drop=True), combo_records_list, is_partial


# ==========================================
# 5. Streamlit UI
# ==========================================
st.set_page_config(page_title="Test finder by Guildhall", layout="wide")

st.title("Find your test")
st.markdown("Search for medical insights, and the system will automatically compute the most cost-effective combinations of tests to cover your query.")

st.divider()

selected_keywords = st.multiselect(
    label="Type to search and select insights:",
    options=all_insights,
    placeholder="e.g., cholesterol, syphilis, vitamin d..."
)

sort_by = st.radio(
    label="Sort results by:",
    options=["Cost", "Turnaround"],
    horizontal=True
)

if selected_keywords:
    with st.spinner('Calculating best combinations...'):
        result_df, combo_records_list, is_partial = find_test_combinations(
            selected_keywords,
            sort_by='turnaround' if sort_by == "Turnaround" else 'cost'
        )

    if result_df is not None and not result_df.empty:
        if is_partial:
            st.warning(
                "No combination of up to 3 tests covers all selected biomarkers. "
                "Showing best partial matches (≥ 80% coverage), sorted by most covered first."
            )
        else:
            st.success(f"Found {len(result_df)} valid test combinations!")

        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🔬 Key Highlights")
        st.caption("AI-generated summary for the top-ranked combination.")
        with st.spinner("Generating key highlights..."):
            top_combo = combo_records_list[0] if combo_records_list else []
            highlight = get_comparison(top_combo)
        st.info(highlight)
    else:
        st.warning("No test combinations found. Try reducing your criteria.")
else:
    st.info("👆 Please select at least one insight from the dropdown above to begin.")