import requests
import sqlalchemy as sa
import pandas as pd

from pipeline.process import api_call, format_response, schema_linking, classify
from pipeline.ragStep import rag_step
from secret.config import SQL_URL

# Setup params for query engine
params = requests.get(SQL_URL).json()['params']
engine = sa.create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}")
engine.begin()

# TODO: Update the entire pipeline here
# TODO: Create different pipeline for original process


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
    # Schema linking to obtain the tables needed for the query
    tables, schema_usage = schema_linking(query, model)
    
    # Classify the query
    to_classify = query + " The following tables are needed: {tables}"
    label, classify_usage = classify(to_classify, model)
    
    # TODO: Add different if's depending on label and create different prompts
    # TODO: Add self correction
    
    # Context for the RAG process
    with open('your_file.txt', 'r') as file:
        context = file.read()
        
    # Creating the RAG instruction
    ragInstruction = f"Search for the information more relevant to the next 
    natural language query. Take in mind that the database you'll be asking 
    is a SQL database (PostgreSQL): {query}"
    
    # Initiating RAG
    rag_info = rag_step(size, overlap, context, ragInstruction, quantity)
    
    # Creating the prompt
    prompt = f"Relevant info xD
    Take this in consideration: {rag_info}"
    
    # Calling the LLM
    response, usage = api_call(model, max_tokens, prompt)
    
    # Formatting the response
    table = format_response(format, response)
    
    return table, usage


if __name__ == '__main__':
    query = "Get the object identifier, candidate identifier, magnitudes, magnitude errors, and band identifiers as a function of time of the objects classified as SN II in the year 2019-2022, with probability larger than 0.6, initial rise rate greater than 0.5 in ZTF g and r-band and number of detections greater than 50."
    model = "gpt-4o"
    max_tokens = 500
    size = 50
    overlap = 10
    quantity = 3