## SEMANTIC EXPANSION OF SEARCH

For each query term, we find its most similar biomarkers in the database as follows:
1. get the embedding of the query term
2. get the embedding of all biomarkers in the database
3. calculate the cosine similarity between the query term and all biomarkers
4. find all the biomarkers with cosine similarity >= 0.85
5. add them to the required_set and run set cover on the expanded set.

## GLOBAL COMPARISON AGENT
Expand the agent's outlook to include all possible combinations of biomarkers that cover the input biomarkers and similar biomarkers (similarity >= 0.85). Gives insights on:

1. Cheapest option/most readily available option that covers all the input biomarkers.

2. Another combination that is near in price/turnaround time (±20%) but covers more similar biomarkers.

3. Comparison of those two.