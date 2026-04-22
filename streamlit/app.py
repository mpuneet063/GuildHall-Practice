import streamlit as st
import pandas as pd
import requests
import re
from itertools import combinations
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# ==========================================
# 1. Gemini API Configuration (Debug Mode)
# ==========================================


# 1. Force the app to tell us where it is looking
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    # st.sidebar.success("✅ Key successfully loaded from Streamlit Cloud Secrets!")
else:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # if GEMINI_API_KEY:
    #     # st.sidebar.info("✅ Key successfully loaded from local .env!")
    # else:
    #     st.sidebar.error("❌ CRITICAL: No key found in Secrets OR .env!")

# 2. Stop execution safely if missing
if not GEMINI_API_KEY:
    st.warning("Halting execution. Please fix API key configuration.")
    st.stop()

# 3. Build URL
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
GEMINI_HEADERS = {"Content-Type": "application/json"}

# ==========================================
# 2. Data Loading & Pre-processing (Cached)
# ==========================================
# Load the embedding model ONCE (cached) — this takes 30s on first run, then instant
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def generate_biomarker_embeddings(df_biomarkers):
    """
    df_biomarkers: unique list of all biomarkers in the dataset.
    """
    model = load_model()
    # generate embeddings
    embeddings = model.encode(df_biomarkers)
    return embeddings

@st.cache_data
def load_and_clean_data():
    # 1. Get the directory where app.py currently lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Join that directory with your file name
    file_path = os.path.join(current_dir, "2026_NP_Prices_simple_details.xlsx")
    
    # 3. Read the explicitly located file
    df = pd.read_excel(file_path)

    # Fix misalignment: The 'Tests' column in the Excel file is shifted down by 1 row 
    # relative to the 'test code' and 'Test Name' columns.
    df['Tests'] = df['Tests'].shift(-1)


    def convert_to_days(turnaround_str):
        if pd.isna(turnaround_str):
            return None

        text = str(turnaround_str).lower().strip()

        if "same day" in text or "1 day" in text:
            return 1

        if "hrs" in text or "hour" in text:
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
all_insights_embeddings = generate_biomarker_embeddings(all_insights)


# ==========================================
# 3. AI Comparison Agent (Powered by Gemini)
# ==========================================
def get_comparison(top_bundles: list, sort_by: str, required_insights: list):
    if not top_bundles:
        return "No combinations available to compare."

    # 1. FIXED: Inject the "Extra" data into the LLM's context
    context_blocks = []
    for i, bundle in enumerate(top_bundles):
        context_blocks.append(
            f"Option {i+1} ({bundle.get('Available Options', 'Unknown')}):\n"
            f"- Cost: £{bundle.get('Total Cost (in GBP)', 0)} | Turnaround: {bundle.get('Turnaround (Days)', 0)} days\n"
            f"- Hits: {bundle.get('Covers', '')}\n"
            f"- Misses: {bundle.get('Misses', 'None')}\n"
            f"- Extra (Bonus Biomarkers): {bundle.get('Extra', 'None')}" # <-- Added this line!
        )
    
    context_str = "\n\n".join(context_blocks)
    priority = "Turnaround Time (speed of results)" if sort_by == "Turnaround" else "Cost-effectiveness (lowest price)"

    # 2. UPDATED PROMPT: Tell it how to handle the Extras
    payload = {
        "systemInstruction": {
            "parts": [{"text": "You are a clinical logistics advisor. Your job is to analyze diagnostic test bundles and help a patient choose the best option based on their specific priorities. Never just define the tests; analyze the trade-offs."}]
        },
        "contents": [{
            "parts": [{"text": (
                f"The patient needs to test for: {', '.join(required_insights)}.\n"
                f"They have sorted their search to prioritize: **{priority}**.\n\n"
                f"Here is the raw data for the top {len(top_bundles)} combinations our algorithm generated:\n"
                f"{context_str}\n\n"
                f"Write EXACTLY three bullet points comparing these options. "
                f"Bullet 1: Analyze the top-ranked option and why it wins based on their priority.\n"
                f"Bullet 2: Contrast it with the other options regarding what biomarkers are missed, OR what Extra (bonus) biomarkers are gained for the price difference.\n" # <-- Tweaked this instruction
                f"Bullet 3: Provide a definitive clinical/logistical recommendation.\n"
                f"Keep the tone professional, objective, and strictly limit the output to these three bullet points."
            )}]
        }],
        "generationConfig": {
            "maxOutputTokens": 3000,
            "temperature": 0.2 # Lowered temperature for more analytical, less creative output
        }
    }
    
    import time
    for attempt in range(3):
        try:
            response = requests.post(GEMINI_URL, headers=GEMINI_HEADERS, json=payload, timeout=30)
            if response.status_code == 429:
                time.sleep(5)
                continue 
                
            response.raise_for_status() 
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
            
        except Exception as e:
            if attempt == 2:
                return f"Comparison currently unavailable. Please check the table above."

# ==========================================
# 4. Semantic Search Expansion (k-NN)
# ==========================================
def expand_required_set(user_queries, all_biomarkers, biomarker_embeddings):
    """
    Expands the user's selected biomarkers using k-NN semantic search.
    """
    model = load_model()
    expanded_set = set()

    for query in user_queries:
        # 1. Get embedding of query
        query_vector = model.encode([query])
        
        # 2 & 3. Calculate cosine similarity
        similarities = cosine_similarity(query_vector, biomarker_embeddings)[0]
        
        # 4. Find matches >= 0.85
        matched_indices = np.where(similarities >= 0.85)[0]
        
        # 5. Add to expanded set
        for idx in matched_indices:
            expanded_set.add(all_biomarkers[idx])
            
    return list(expanded_set)
# ==========================================
# 5. Residual-Pruned Set Cover Search
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

    records = relevant_tests.to_dict('records')

    # --- OPTIMIZATION: PREVENT 503 TIMEOUTS ---
    # Prune the search space to only use the most efficient tests
    if len(records) > 45:
        pruned_records = []
        seen_codes = set()
        for req in required_set:
            req_records = [r for r in records if req in r['_keyword_set']]
            
            # Keep top 15 cheapest and top 15 fastest for this specific insight
            cheap_reqs = sorted(req_records, key=lambda x: x.get('Lab Fee', 9999))[:15]
            fast_reqs = sorted(req_records, key=lambda x: x.get('Turnaround_Days', 999) or 999)[:15]
            
            for r in cheap_reqs + fast_reqs:
                code = r.get('test code', id(r)) # use id as fallback if no code
                if code not in seen_codes:
                    seen_codes.add(code)
                    pruned_records.append(r)
        records = pruned_records
    # ------------------------------------------

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
    non_solo_records = [t for t in records if t.get('test code') not in solo_full_ids]
    for combo in combinations(non_solo_records, 2):
        combo_insights = set().union(*(t['_keyword_set'] for t in combo))
        if required_set.issubset(combo_insights):
            valid_bundles.append(make_bundle(combo))

    # --- r=3: only form triples from non-solo tests,
    #          skip any triple whose 2-test sub-combo already covers 100% ---
    for combo in combinations(non_solo_records, 3):
        combo_insights = set().union(*(t['_keyword_set'] for t in combo))
        if required_set.issubset(combo_insights):
            sub_already_covers = any(
                required_set.issubset(set().union(*(s['_keyword_set'] for s in sub)))
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
    # --- NEW: Intercept and Expand ---
    with st.spinner("Semantically expanding search..."):
        expanded_search_terms = expand_required_set(
            user_queries=selected_keywords, 
            all_biomarkers=all_insights, 
            biomarker_embeddings=all_insights_embeddings
        )
        
    st.info(f"**Expanded search to include:** {', '.join(expanded_search_terms)}")
    # ---------------------------------

    with st.spinner('Calculating best combinations...'):
        # 1. Fast mathematical combinations (Runs dynamically)
        # Make sure you are passing `expanded_search_terms` here!
        result_df, combo_records_list, is_partial = find_test_combinations(
            selected_keywords=expanded_search_terms, 
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
        
        # 2. The Gated LLM Call
        st.subheader("🔬 Clinical Comparison")
        st.caption("Generate an AI summary comparing the top options based on your priorities.")
        
        if st.button("Get Comparison", type="primary"):
            with st.spinner("Analyzing bundle trade-offs..."):
                top_3_bundles = result_df.head(3).to_dict('records')
                # Note: Pass the original selected_keywords to the LLM so it knows what the patient ACTUALLY asked for
                highlight = get_comparison(top_3_bundles, sort_by, selected_keywords) 
                
            st.info(highlight)
            
    else:
        st.warning("No test combinations found. Try reducing your criteria.")
else:
    st.info("👆 Please select at least one insight from the dropdown above to begin.")