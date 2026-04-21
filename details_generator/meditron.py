<<<<<<< HEAD
import pandas as pd
import requests
import time
import json

# configure API
url = "https://fh7sehkgzbk80qse.eu-west-1.aws.endpoints.huggingface.cloud"
token = "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"

def get_details(test_name, biomarker):
    "fetches details of a test with citations"
    prompt = f"""
    Instruct: Act as a medical clinical specialist. 
    Summarize this test in 2-3 sentences and provide 2 sources.
    Test: {test_name}
    Biomarkers: {biomarker}
    
    Output Format:
    Details: [Summary]
    Sources: [Links]
    """

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code}"

# load excel
df = pd.read_excel("2026 NP PRICES.xlsx", sheet_name='Test_List')

details_list = []
sources_list = []

# Process rows with rate limiting for free-tier stability
for index, row in df.iterrows():
    print(f"Processing: {row['Test Name']}")
    raw_output = get_details(row['Test Name'], row['Tests'])
    
    # Basic parsing logic for the two columns
    try:
        details = raw_output.split("Details:")[1].split("Sources:")[0].strip()
        sources = raw_output.split("Sources:")[1].strip()
    except:
        details, sources = raw_output, "N/A"
        
    details_list.append(details)
    sources_list.append(sources)
    
    # Rate limiting for free API tier
    time.sleep(1) 

# Save the enriched file
df['Details'] = details_list
df['Sources'] = sources_list
df.to_excel("NP_2026_with_details.xlsx", index=False)
=======
import pandas as pd
import requests
import time
import json

# configure API
url = "https://fh7sehkgzbk80qse.eu-west-1.aws.endpoints.huggingface.cloud"
token = "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"

def get_details(test_name, biomarker):
    "fetches details of a test with citations"
    prompt = f"""
    Instruct: Act as a medical clinical specialist. 
    Summarize this test in 2-3 sentences and provide 2 sources.
    Test: {test_name}
    Biomarkers: {biomarker}
    
    Output Format:
    Details: [Summary]
    Sources: [Links]
    """

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code}"

# load excel
df = pd.read_excel("2026 NP PRICES.xlsx", sheet_name='Test_List')

details_list = []
sources_list = []

# Process rows with rate limiting for free-tier stability
for index, row in df.iterrows():
    print(f"Processing: {row['Test Name']}")
    raw_output = get_details(row['Test Name'], row['Tests'])
    
    # Basic parsing logic for the two columns
    try:
        details = raw_output.split("Details:")[1].split("Sources:")[0].strip()
        sources = raw_output.split("Sources:")[1].strip()
    except:
        details, sources = raw_output, "N/A"
        
    details_list.append(details)
    sources_list.append(sources)
    
    # Rate limiting for free API tier
    time.sleep(1) 

# Save the enriched file
df['Details'] = details_list
df['Sources'] = sources_list
df.to_excel("NP_2026_with_details.xlsx", index=False)
>>>>>>> 0ad4836c5671e960eb832e7122cb58e99245d387
print("Finished! File saved as NP_2026_with_details.xlsx")