## SEMANTIC EXPANSION OF SEARCH

For each query term, we find its most similar biomarkers in the database as follows:
1. get the embedding of the query term
2. get the embedding of all biomarkers in the database
3. calculate the cosine similarity between the query term and all biomarkers
4. find all the biomarkers with cosine similarity >= 0.85
5. add them to the required_set and run set cover on the expanded set.