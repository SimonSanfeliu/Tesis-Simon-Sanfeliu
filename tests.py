"""
This is a file for running tests on the pipeline with different queries
"""

# TODO: Dar request ID o una query x

import pandas as pd
from pprint import pprint
from main import run_pipeline, engine
engine.begin()

# Query to process
query = """Query objects within 10 degress of the next positions: ('source_1',
    160.18301441363647, 33.0164673528409), ('source_2', 174.21524897555543, 
    44.83789535222221), that have their first detection the first 7 days of
    February 2023, with SN probabilities > 0.4, and ranking=1 in the stamp 
    classifier"""

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
                                            engine, rag_pipe=True, 
                                            self_corr=True)
print("Resulting table:")
print(result)
print("Total usage of the pipeline:")
pprint(total_usage)

# The prompts used will be saved in this file
with open(f"prompts/examples/prompts_query_{model}.txt", "w") as f:
    f.write(str(prompts))