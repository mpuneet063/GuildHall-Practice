import pandas as pd
import time
import re
from huggingface_hub import InferenceClient
from concurrent.futures import ThreadPoolExecutor

# Initialize the Hugging Face client
# Replace with your actual HF token
client = InferenceClient(model="meta-llama/Llama-3.3-70B-Instruct", token="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")

def get_medical_info(test_name, biomarkers):
    """Generates a professional, succinct clinical summary using Chat Completion."""
    messages = [
        {"role": "system", "content": "You are a clinical laboratory specialist. Provide succinct, technical medical summaries."},
        {"role": "user", "content": f"Provide a clinical summary for Test: {test_name}. Biomarkers: {biomarkers}.\n\nStructure exactly as: Details: [Concise clinical purpose and pathophysiology. Max 75 words.] Sources: [Reputable reference URL]"}
    ]
    
    try:
        # Use chat_completion instead of text_generation
        response = client.chat_completion(
            messages, 
            max_tokens=250, 
            temperature=0.3
        )
        # Extract the content from the chat response
        return response.choices[0].message.content
    except Exception as e:
        return f"Details: Error generating info - {e}\nSources: N/A"

def process_row(row):
    """Processes a single row and prints status."""
    print(f"Processing: {row['Test Name']}...")
    raw_output = get_medical_info(row['Test Name'], row['Tests'])
    
    # Parsing logic
    try:
        # Normalize the raw output
        raw_output = raw_output.replace("Details:", "Details").replace("Sources:", "Sources")
        
        if "Sources" in raw_output:
            parts = raw_output.split("Sources")
            details = parts[0].replace("Details", "").strip()
            # Extract URL using Regex for reliability
            url_match = re.search(r'(https?://\S+)', parts[1])
            sources = url_match.group(0) if url_match else "N/A"
        else:
            details = raw_output.replace("Details", "").strip()
            sources = "N/A"
            
    except Exception:
        details, sources = raw_output, "N/A"
        
    return details, sources

def run_pipeline():
    # 1. Load Data
    print("Loading Excel file...")
    df = pd.read_excel("2026 NP PRICES.xlsx", sheet_name='Test_List')
    
    # 2. Run in parallel (max_workers=3 to respect HF rate limits)
    print(f"Starting batch processing for {len(df)} tests...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_row, [row for _, row in df.iterrows()]))
    
    # 3. Save to DataFrame
    details_list, sources_list = zip(*results)
    df['Details'] = details_list
    df['Sources'] = sources_list
    
    # 4. Export
    output_file = "2026_NP_Prices_simple_details.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Pipeline complete. Saved to {output_file}")

if __name__ == "__main__":
    run_pipeline()