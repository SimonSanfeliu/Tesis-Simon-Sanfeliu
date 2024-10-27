import requests
import sqlalchemy as sa

from pipeline.process import api_call, format_response, schema_linking, \
    classify, decomposition_v2, run_query
from pipeline.ragStep import rag_step
from prompts.correction.SelfCorrection import prompt_self_correction_v2, \
    general_context_selfcorr_v1, general_context_selfcorr_v1_python
from secret.config import SQL_URL

# Setup params for query engine
params = requests.get(SQL_URL).json()['params']
engine = sa.create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}")
engine.begin()


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
        prompts (dict): Dictonary with the prompts used in every step of the 
        pipeline
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
    
    ## Astronomical context:
    There are two main types of variable objects: those that have persistent 
    variability and those that have a transient nature. In the case of 
    persistent variability sources (Periodic or Stochastic), the relevant light 
    curve magnitude is the corrected magnitude (magpsf_corr). In the case of 
    transient sources (Transient), the relevant light curve is the uncorrected 
    magnitude (magpsf). Objects that are transient are considered to be fast 
    risers if dmd_dt < -0.25 mag per day (in magstats) in any band. Note that 
    when the user refers to the first detection of a given object, you should 
    use the firstmjd indexed column (in object). When possible, avoid adding 
    restrictions on the mjd column in the detection table, try putting them in 
    the object table first. Note that all the rows in the detection table are 
    by definition detections, you don't need to ask for additional constraints.
    """
    
    # Initiating RAG
    rag_info, schema_usage = rag_step(size, overlap, context, ragInstruction, 
                                      quantity)
    content = rag_info.split("[")[1].split("]")[0]
    true_tables = f"[{content}]"
    print(f"Tables needed: {true_tables}")
    
    # Classify the query
    to_classify = query + f"""\n The following tables are needed to generate 
    the query: {true_tables}"""
    label, classify_usage = classify(to_classify, model)
    print(f"Difficulty: {label}")
    
    # Creating the prompt based on the difficulty of the query
    prompt, decomp_plan, decomp_usage = decomposition_v2(label, 
                                                         query, 
                                                        true_tables, 
                                                        model, 
                                                        format)    
    
    # Obtaining the SQL query
    response, usage = api_call(model, max_tokens, prompt)
    
    # Formatting the response
    table = format_response(format, response)
    print(f"Resulting {format} query: {table}")
    
    # Obtaining the total usage of the pipeline
    total_usage = {
        "Schema Linking": schema_usage,
        "Classification": classify_usage,
        "Decompostion": decomp_usage,
        "Query generation": usage
    }
    
    # Adding up the prompts used
    prompts = {
        "Schema Linking": ragInstruction,
        "Classification": to_classify,
        "Decomposition": decomp_plan,
        "Generation": prompt 
    }
    
    return table, total_usage, prompts


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
        prompts (dict): Dictonary with the prompts used in every step of the 
        pipeline
    """
    # Schema linking to obtain the tables needed for the query
    tables, schema_usage = schema_linking(query, model)
    
    # Classify the query
    to_classify = query + f"\n The following tables are needed: {tables}"
    label, classify_usage = classify(to_classify, model)
    
    # Creating the prompt based on the difficulty of the query
    prompt, decomp_usage = decomposition_v2(label, 
                                            query, 
                                            tables, 
                                            model, 
                                            format)
    
    # Obtaining the SQL query
    table, usage = api_call(model, max_tokens, prompt)
    
    # Formatting the response
    table = format_response(format, table)
    
    # Obtaining the total usage of the pipeline
    total_usage = {
        "Schema Linking": schema_usage,
        "Classification": classify_usage,
        "Decompostion": decomp_usage,
        "Query generation": usage
    }
    
    # Adding up the prompts used
    prompts = {
        "Schema Linking": tables,
        "Classification": to_classify,
        "Decomposition": prompt
    }
    
    return table, total_usage, prompts


def run_pipeline(query, model, max_tokens, size, overlap, quantity, format, 
                 engine, new_pipe, self_corr):
    """Function to run the entire pipeline. This pipeline could be the 
       original one or the new one. Here the self-correction is applied.

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
        engine (sqlalchemy.engine.base.Engine): SQL database engine
        new_pipe (bool): Condition to use the new pipeline
        self_corr (bool): Condition to use self-correction
        
    Returns:
        result (pandas.DataFrame): Dataframe with the resulting table
        total_usage (dict): API usage after the whole process
        prompts (dict): Dictonary with the prompts used in every step of the 
        pipeline
    """
    # Check if the new pipeline is being used
    if new_pipe:
        table, total_usage, prompts = pipeline(query, model, max_tokens, size, 
                                               overlap, quantity, format)
        # If self-correction is enabled, use the respective prompts to correct
        if self_corr:
            try:
                result = run_query(format, table, engine)
            except Exception as e:
                print(f"Raised exception: {e}")
                print("Start retry with self-correction")
                
                tab_schema = prompts["Classification"]\
                    .split("\n The following tables are needed to generate \
                        the query: ")[0]
                    
                if format == "sql":
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1, 
                        tab_schema=tab_schema, 
                        req=query, 
                        sql_pred=table, 
                        error=str(e))
                    new, new_usage = api_call(model, max_tokens, corr_prompt)
                    new = format_response(format, new)
                    total_usage["Self-correction"] = new_usage
                    prompts["Self-correction"] = corr_prompt
                    
                    try:
                        result = run_query(format, new, engine)
                    except Exception as e:
                        raise Exception(f"Failed again: {e}")
                    
                else:
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1_python, 
                        tab_schema=tab_schema, 
                        req=query, 
                        sql_pred=table, 
                        error=str(e))
                    new, new_usage = api_call(model, max_tokens, corr_prompt)
                    new = format_response(format, new)
                    total_usage["Self-correction"] = new_usage
                    prompts["Self-correction"] = corr_prompt
                    
                    try:
                        result = run_query(format, new, engine)
                    except Exception as e:
                        raise Exception(f"Failed again: {e}")

        # W/o self-correction
        else:
            try:
                result = run_query(format, table, engine)
            except Exception as e:
                raise Exception(f"Raised exception: {e}")

    # Using the recreated pipeline
    else:
        table, total_usage, prompts = recreated_pipeline(query, model, 
                                                         max_tokens, format)
        # If self-correction is enabled, use the respective prompts to correct
        if self_corr:
            try:
                result = run_query(format, table, engine)
            except Exception as e:
                print(f"Raised exception: {e}")
                print("Start retry with self-correction")
                
                tab_schema = prompts["Classification"]\
                    .split("\n The following tables are needed to generate \
                        the query: ")[0]
                    
                if format == "sql":
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1, 
                        tab_schema=tab_schema, 
                        req=query, 
                        sql_pred=table, 
                        error=str(e))
                    new, new_usage = api_call(model, max_tokens, corr_prompt)
                    new = format_response(format, new)
                    total_usage["Self-correction"] = new_usage
                    prompts["Self-correction"] = corr_prompt
                    
                    try:
                        result = run_query(format, new, engine)
                    except Exception as e:
                        raise Exception(f"Failed again: {e}")
                    
                else:
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1_python, 
                        tab_schema=tab_schema, 
                        req=query, 
                        sql_pred=table, 
                        error=str(e))
                    new, new_usage = api_call(model, max_tokens, corr_prompt)
                    new = format_response(format, new)
                    total_usage["Self-correction"] = new_usage
                    prompts["Self-correction"] = corr_prompt
                    
                    try:
                        result = run_query(format, new, engine)
                    except Exception as e:
                        raise Exception(f"Failed again: {e}")

        # W/o self-correction
        else:
            try:
                result = run_query(format, table, engine)
            except Exception as e:
                raise Exception(f"Raised exception: {e}")
            
    return result, total_usage, prompts
    