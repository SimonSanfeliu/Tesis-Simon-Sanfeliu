"""
This is a file for running tests on the pipeline with different queries
"""

import pandas as pd
from pprint import pprint
from main import run_pipeline, engine
engine.begin()

# Query to process
query = "Get the object identifiers, probabilities in the stamp classifier and light curves (only detections) for objects whose highest probability in the stamp classifier is obtained for class SN, that had their first detection in the first 2 days of september, and that qualify as fast risers."

# Corresponding request ID for the query
df = pd.read_csv("txt2sql_alerce_train_v2.csv")
req_id = df.loc[df["request"] == query].reset_index(drop=True)["req_id"][0]

# Model to use
#model = "claude-3-5-sonnet-20240620"
model = "gpt-4o-2024-08-06"

# Format for the pipeline
format = "python"

# RAG parameters
max_tokens = 1000
size = 700
overlap = 300
quantity = 10

# Running the pipeline
result, total_usage, prompts = run_pipeline(query, model, max_tokens, size, 
                                            overlap, quantity, format, engine, 
                                            True, True)
print("Resulting table:")
print(result)
print("Total usage of the pipeline:")
pprint(total_usage)

# The prompts used will be saved in this file
with open(f"prompts/examples/prompts_query_{req_id}_{model}.txt", "w") as f:
    f.write(str(prompts))