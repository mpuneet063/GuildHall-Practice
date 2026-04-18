import pandas as pd
import time
from groq import Groq
from concurrent.futures import ThreadPoolExecutor

# initialize groq client
client = Groq(api_key= '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')

def get_medical_info(test_name, biomarkers):
    prompt = f"""
    Act as a friendly medical educator. Explain the test '{test_name}' 
    using these biomarkers: {biomarkers}.
    
    Structure your answer for a layman (simple, clear):
    - What is it? (1-2 sentences)
    - Normal range: (Briefly explain or provide standard context)
    - Related symptoms/diseases: (List 3 common ones)
    
    Keep it under 100 words total. Provide source URLs for your information.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# load data
df = pd.read_excel("2026 NP Prices.xlsx", sheet_name='Test_List')

# parallel processing function
def process_row(row):
    time.sleep(2.1)     # prevents hitting 30 RPM limit
    raw_output = get_medical_info(row['Test Name'], row['Tests'])

    # parsing
    try:
        # Looking for the 'Sources' marker in the output
        if "Sources" in raw_output:
            parts = raw_output.split("Sources")
            details = parts[0].replace("Details:", "").strip()
            sources = parts[1].replace(":", "").replace("-", "").strip()
        else:
            details = raw_output.strip()
            sources = "N/A"
    except Exception:
        details, sources = raw_output, "N/A"
        
    return details, sources

# Execution block to finish the pipeline
# Use ThreadPoolExecutor to run the function in parallel (max_workers=5 is safe for 30 RPM)
with ThreadPoolExecutor(max_workers=5) as executor:
    # map returns a generator; we convert to list to trigger the processing
    results = list(executor.map(process_row, [row for _, row in df.iterrows()]))

# Unzip results into two separate lists
details_list, sources_list = zip(*results)

# Assign back to your dataframe
df['Details'] = details_list
df['Sources'] = sources_list

# Save to your final Excel file
df.to_excel("2026_NP_Prices_simple_details.xlsx", index=False)
print("Processing complete. File saved as 2026_NP_Prices_simple_details.xlsx")