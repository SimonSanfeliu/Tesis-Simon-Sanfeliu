import requests
import sqlalchemy as sa
import pandas as pd

from pipeline.process import api_call, format_response, schema_linking, classify, decomposition
from pipeline.ragStep import rag_step
from secret.config import SQL_URL

# Setup params for query engine
params = requests.get(SQL_URL).json()['params']
engine = sa.create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}")
engine.begin()

# TODO: Update the entire pipeline here


def pipeline(query, model, max_tokens, size, overlap, quantity, format):
    """Pipeline of the LLM process using RAG to get the SQL query.

    Args:
        query (str): Natural language query for the database
        model (str): LLM to use
        max_tokens (int): Maximum amount of output tokens of the LLM
        size (int): Size of the chunks for the character text splitter used in
        the RAG process
        overlap (int): Size of the overlap of the chunks used in the RAG 
        process
        quantity (int): The amount of most similar chunks to consider in the
        RAG process
        format (str): The type of formatting to use. It can be 'singular' for
        a singular query string or 'var' for the decomposition in variables
        
    Returns:
        table (str): Resulting SQL query
        usage (dict): API usage after RAG process
    """
    # TODO: Add self correction
    
    # Context for the RAG process
    with open('prompts/schema_linking/DBSchema.txt', 'r') as file:
        context = file.read()
        
    # Creating the RAG instruction
    ragInstruction = f" Given the user request, search for the tables needed
    to generate a SQL query in the ALeRCE database (PostgreSQL).
    \n Request: {query}"
    
    # Initiating RAG
    rag_info = rag_step(size, overlap, context, ragInstruction, quantity)
    
    # Creating the prompt
    prompt = f"""Given the user request, select the tables needed to 
    generate a SQL query. Give the answer in the following format: [table1, 
    table2, ...]. For example, if the answer is table object and table 
    taxonomy, then you should type: [object, taxonomy].
    
    Consider that these tables are necessary to execute the query: {rag_info}"""
    
    # Calling the LLM
    tables, schema_usage = api_call(model, max_tokens, prompt)
    
    # Classify the query
    to_classify = query + f"\n The following tables are needed to generate the
    query: {tables}"
    label, classify_usage = classify(to_classify, model)
    
    # Creating the prompt based on the difficulty of the query
    prompt, decomp_usage = decomposition(label, to_classify)    
    
    # Obtaining the SQL query
    response, usage = api_call(model, max_tokens, prompt)
    
    # Formatting the response
    table = format_response(format, response)
    
    # Obtaining the total usage of the pipeline
    total_usage = {
        "Schema Linking": schema_usage,
        "Classification": classify_usage,
        "Decompostion": decomp_usage,
        "Query generation": usage
    }
    
    return table, total_usage


def recreated_pipeline(query, model, max_tokens, format):
    """Recreated pipeline from the original work

    Args:
        query (str): Natural language query for the database
        model (str): LLM to use
        max_tokens (int): Maximum amount of output tokens of the LLM
        format (str): The type of formatting to use. It can be 'singular' for
        a singular query string or 'var' for the decomposition in variables
        
    Returns:
        table (str): Resulting SQL query
        total_usage (dict): API usage after the whole process
    """
    # Schema linking to obtain the tables needed for the query
    tables, schema_usage = schema_linking(query, model)
    
    # Classify the query
    to_classify = query + f"\n The following tables are needed: {tables}"
    label, classify_usage = classify(to_classify, model)
    
    # Creating the prompt based on the difficulty of the query
    prompt, decomp_usage = decomposition(label, to_classify)
    
    # Obtaining the SQL query
    response, usage = api_call(model, max_tokens, prompt)
    
    # Formatting the response
    table = format_response(format, response)
    
    # Obtaining the total usage of the pipeline
    total_usage = {
        "Schema Linking": schema_usage,
        "Classification": classify_usage,
        "Decompostion": decomp_usage,
        "Query generation": usage
    }
    
    return table, total_usage
    
        

if __name__ == '__main__':
    query = "Get the object identifier, candidate identifier, magnitudes, magnitude errors, and band identifiers as a function of time of the objects classified as SN II in the year 2019-2022, with probability larger than 0.6, initial rise rate greater than 0.5 in ZTF g and r-band and number of detections greater than 50."
    model = "gpt-4o"
    max_tokens = 500
    size = 50
    overlap = 10
    quantity = 3