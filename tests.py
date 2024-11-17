"""
This is a file for running tests on the pipeline with different queries
"""

# TODO: Dar request ID o una query x

import pandas as pd
from pprint import pprint
from main import run_pipeline, engine
engine.begin()

# Query to process
query = """Get the object identifier, candidate identifier, psf magnitudes, magnitude errors, and band identifiers as a function of time of the objects classified as SN II with probability larger than 0.6, number of detections greater than 50 and difference between minimum and maximum magnitudes in ZTF g-band greater than 2 mag."""

# # Corresponding request ID for the query (if it exists)
# df = pd.read_csv("txt2sql_alerce_train_v2.csv")
# req_id = df.loc[df["request"] == query].reset_index(drop=True)["req_id"]

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
                                            overlap, quantity, format, False, 
                                            engine, rag_pipe=False, 
                                            self_corr=True)
print("Resulting table:")
print(result)
print("Total usage of the pipeline:")
pprint(total_usage)

# The prompts used will be saved in this file
with open(f"prompts/examples/prompts_query_{model}.txt", "w") as f:
    f.write(str(prompts))