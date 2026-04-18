import streamlit as st
import pandas as pd
import numpy as np
import re
from itertools import combinations

# ==========================================
# 1. Data Loading & Pre-processing (Cached)
# ==========================================
# @st.cache_data ensures the Excel file is only read once upon startup, 
# preventing lag during user interaction.
@st.cache_data
def load_and_clean_data():
    # Load data
    df = pd.read_excel('2026 NP PRICES.xlsx', sheet_name='All Blood Tests available')
    df = df.drop(index=0).reset_index(drop=True)
    
    # Drop unnamed columns
    columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=columns_to_drop)

    # Turnaround conversion logic
    def convert_to_days(turnaround_str):
        if pd.isna(turnaround_str):
            return None
        turnaround_str = str(turnaround_str).lower()

        if 'same day' in turnaround_str: return 1 
        
        match_weeks = re.search(r'(\d+)\s*weeks?', turnaround_str)
        if match_weeks: return int(match_weeks.group(1)) * 7

        match_days = re.search(r'(\d+)\s*(working\s*)?days?', turnaround_str)
        if match_days: return int(match_days.group(1))

        match_range = re.search(r'(\d+)\s*-\s*(\d+)\s*days?', turnaround_str)
        if match_range: return int(match_range.group(2))

        return None

    # Apply turnaround logic
    df['NP Turnaround (Days)'] = df['NP Turnaround'].apply(convert_to_days)
    df['Turnaround (Days)'] = df['GUILD  Turnaround'].apply(convert_to_days)
    df = df.drop(columns=['NP Turnaround', 'GUILD  Turnaround'])

    # Normalize text
    def normalize_text(text):
        if pd.isna(text): return None
        return str(text).lower()

    df['Tests'] = df['Tests'].apply(normalize_text)

    # Extract unique keywords
    all_insights = df['Tests'].dropna().str.split(',').explode().str.strip().unique().tolist()
    # Filter out any empty strings
    all_insights = sorted([i for i in all_insights if i]) 

    return df, all_insights

# Initialize data
df, all_insights = load_and_clean_data()


# ==========================================
# 2. Combinatorial Search Logic
# ==========================================
def find_test_combinations(selected_keywords):
    if not selected_keywords:
        return None

    required_set = set(selected_keywords)
    valid_bundles = []

    # Filter tests that contain at least one requested keyword
    relevant_tests = df[df['Tests'].apply(lambda x: any(k in str(x) for k in selected_keywords))]

    # Generate combinations (limiting to max 3 tests to prevent memory crashes)
    max_tests_in_bundle = min(3, len(relevant_tests))

    for r in range(1, max_tests_in_bundle + 1):
        for combo in combinations(relevant_tests.to_dict('records'), r):

            # Aggregate insights in the current combination
            combo_insights = set()
            for test in combo:
                combo_insights.update(str(test['Tests']).split(', '))

            # Check if the combination covers all selected keywords
            if required_set.issubset(combo_insights):
                valid_bundles.append({
                    "Available Options": " + ".join([str(t.get('Profile Code', 'N/A')) for t in combo]),
                    "Total Cost (in GBP)": sum([t.get('DO NOT USE!     NP cost', 0) for t in combo]),
                    "Information Provided": ", ".join(sorted(combo_insights))
                })

    # Format the output
    if valid_bundles:
        results_df = pd.DataFrame(valid_bundles)
        return results_df.sort_values(by="Total Cost (in GBP)").reset_index(drop=True)
    else:
        return pd.DataFrame() # Return empty dataframe if nothing matches


# ==========================================
# 3. Streamlit UI
# ==========================================
# Page configuration
st.set_page_config(page_title="Medical Test Search", layout="wide")

st.title("Find your test")
st.markdown("Search for medical insights, and the system will automatically compute the most cost-effective combinations of tests to cover your query.")

st.divider()

# Streamlit's native multiselect acts as both the text filter AND the dropdown
selected_keywords = st.multiselect(
    label="Type to search and select insights:",
    options=all_insights,
    placeholder="e.g., cholesterol, syphilis, vitamin d..."
)

# Output Table Generation
if selected_keywords:
    with st.spinner('Calculating best combinations...'):
        result_df = find_test_combinations(selected_keywords)
        
        if result_df is not None and not result_df.empty:
            st.success(f"Found {len(result_df)} valid test combinations!")
            # Render as an interactive, Excel-like table
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No test combinations found to cover all selected insights. Try reducing your criteria.")
else:
    st.info("👆 Please select at least one insight from the dropdown above to begin.")