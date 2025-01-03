import sqlalchemy as sa
import pandas as pd

from pipeline.process import api_call, format_response, schema_linking, \
    classify, decomposition_v2, pricing, direct_prompts, astro_context
from pipeline.ragStep import rag_step


def pipeline(query: str, model: str, max_tokens: int, size: int, overlap: int, 
             quantity: int, format: str, direct: bool) -> tuple[str, dict, dict, str, str]:
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
        direct (bool): If True, use direct approach for query generation. If 
        False, use step-by-step approach
        
    Returns:
        table (str): Resulting SQL query
        usage (dict): API usage after RAG process
        prompts (dict): Dictonary with the prompts used in every step of the 
        pipeline
        true_tables (str): Table schema used for the query
        label (str): Difficulty label detected for the query
    """    
    # Context for the RAG process
    with open('prompts/schema_linking/DBSchema.txt', 'r') as file:
        context = file.read()
        
    # Creating the RAG instruction
    ragInstruction = f"""
    Given the user request, select the tables needed to generate a SQL query. 
    Give the answer in the following format: [table1, table2, ...]. For 
    example, if the answer is table object and table taxonomy, then you should 
    type: [object, taxonomy].
    
    User request: {query}
    
    {astro_context}
    """
    
    # Initiating RAG
    rag_info, schema_usage = rag_step(size, overlap, context, ragInstruction, 
                                      quantity)
    content = rag_info.split("[")[1].split("]")[0]
    true_tables = f"[{content}]"
    print(f"Tables needed: {true_tables}", flush=True)
    
    # Classify the query
    label, to_classify, classify_usage = classify(query, true_tables, model)
    print(f"Difficulty: {label}", flush=True)
    
    # If the direct approach is chosen, do not use the decomposition process
    if direct:
        # Creating the prompt based on the difficulty of the query
        prompt = direct_prompts(label, query, true_tables)
        
        # Obtaining the SQL query
        response, usage = api_call(model, max_tokens, prompt)
        
        # Formatting the response
        table = format_response(format, response)
        print(f"Resulting {format} query: {table}", flush=True)
        
        # Obtaining the total usage of the pipeline
        total_usage = {
            "Schema Linking": schema_usage,
            "Classification": classify_usage,
            "Query generation": usage
        }
        
        # Obtaining its costs
        total_usage = pricing(total_usage, model)
        
        # Adding up the prompts used
        prompts = {
            "Schema Linking": ragInstruction,
            "Classification": to_classify,
            "Query generation": prompt 
        }
        
        return table, total_usage, prompts, true_tables, label
    
    # Creating the prompt based on the difficulty of the query
    prompt, decomp_plan, decomp_usage = decomposition_v2(label, 
                                                         query, 
                                                         true_tables, 
                                                         model, 
                                                         format)    
    
    # Obtaining the SQL query
    response, usage = api_call(model, max_tokens, prompt)
    print(f"Raw response: {response}", flush=True)
    
    # Catching borderline cases
    if format == "python" and label == "simple":
        format = "sql"
    
    # Formatting the response
    table = format_response(format, response)
    print(f"Resulting {format} query: {table}", flush=True)
    
    # Obtaining the total usage of the pipeline
    total_usage = {
        "Schema Linking": schema_usage,
        "Classification": classify_usage,
        "Decompostion": decomp_usage,
        "Query generation": usage
    }
    
    # Obtaining its costs
    total_usage = pricing(total_usage, model)
    
    # Adding up the prompts used
    prompts = {
        "Schema Linking": ragInstruction,
        "Classification": to_classify,
        "Decomposition": decomp_plan,
        "Query generation": prompt 
    }
    
    return table, total_usage, prompts, true_tables, label


def recreated_pipeline(query: str, model: str, max_tokens: int, 
                       format: str, direct: bool) -> tuple[str, dict, dict, str, str]:
    """Recreated pipeline from the original work

    Args:
        query (str): Natural language query for the database
        model (str): LLM to use
        max_tokens (int): Maximum amount of output tokens of the LLM
        format (str): The type of formatting to use. It can be 'singular' for
        a singular query string or 'var' for the decomposition in variables
        direct (bool): If True, use direct approach for query generation. If 
        False, use step-by-step approach
        
    Returns:
        table (str): Resulting SQL query
        total_usage (dict): API usage after the whole process
        prompts (dict): Dictonary with the prompts used in every step of the 
        pipeline
        tables (str): Table schema used for the query
        label (str): Difficulty label detected for the query
    """
    # Schema linking to obtain the tables needed for the query
    tables, schema_usage = schema_linking(query, model)
    
    # Classify the query
    label, to_classify, classify_usage = classify(query, tables, model)
    
    # If the direct approach is chosen, do not use the decomposition process
    if direct:
        # Creating the prompt based on the difficulty of the query
        prompt = direct_prompts(label, query, tables)
        
        # Obtaining the SQL query
        response, usage = api_call(model, max_tokens, prompt)
        
        # Formatting the response
        table = format_response(format, response)
        print(f"Resulting {format} query: {table}", flush=True)
        
        # Obtaining the total usage of the pipeline
        total_usage = {
            "Schema Linking": schema_usage,
            "Classification": classify_usage,
            "Query generation": usage
        }
        
        # Obtaining its costs
        total_usage = pricing(total_usage, model)
        
        # Adding up the prompts used
        prompts = {
            "Schema Linking": tables,
            "Classification": to_classify,
            "Query generation": prompt 
        }
        
        return table, total_usage, prompts, tables, label
    
    # Creating the prompt based on the difficulty of the query
    prompt, decomp_plan, decomp_usage = decomposition_v2(label, 
                                                         query,
                                                         tables, 
                                                         model, 
                                                         format)
    
    # Obtaining the SQL query
    response, usage = api_call(model, max_tokens, prompt)
    print(f"Raw response: {response}", flush=True)
    
    # Catching borderline cases
    if format == "python" and label == "simple":
        format = "sql"
    
    # Formatting the response
    table = format_response(format, response)
    
    # Obtaining the total usage of the pipeline
    total_usage = {
        "Schema Linking": schema_usage,
        "Classification": classify_usage,
        "Decompostion": decomp_usage,
        "Query generation": usage
    }
    
    # Obtaining its costs
    total_usage = pricing(total_usage, model)
    
    # Adding up the prompts used
    prompts = {
        "Schema Linking": tables,
        "Classification": to_classify,
        "Decomposition": decomp_plan,
        "Query generation": prompt
    }
    
    return table, total_usage, prompts, tables, label
