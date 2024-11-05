import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import openai
import anthropic
import google.generativeai as genai
import sqlalchemy

from secret.config import OPENAI_KEY, ANTHROPIC_KEY, GOOGLE_KEY
from prompts.classification.Classification import diff_class_prompt
from prompts.schema_linking.SchemaLinking import tables_linking_prompt_V2
from prompts.decomposition.Decomposition import final_prompt_simple_vf, \
    simple_query_task_vf, simple_query_cntx_vf, simple_query_instructions_vf
from prompts.decomposition.Decomposition import medium_decomp_prompt_vf, \
    medium_decomp_gen_vf, medium_decomp_gen_vf_python, medium_query_task_vf, \
    medium_query_cntx_vf, medium_query_instructions_1_vf, \
    medium_query_instructions_2_vf, medium_decomp_task_vf
from prompts.decomposition.Decomposition import adv_decomp_prompt_vf, \
    adv_decomp_gen_vf, adv_decomp_gen_vf_python, adv_query_task_vf, \
    adv_query_cntx_vf, adv_query_instructions_1_vf, \
    adv_query_instructions_2_vf, adv_decomp_task_vf
from final_prompts.final_prompts import *

# Setting up astronomical context
with open("final_prompts/astrocontext.txt", "r") as f:
    astro_context = f.read()


def api_call(model: str, max_tokens: int, prompt: str) -> tuple[str, dict]:
    """Create the API calls for the LLM to use.

    Args:
        model (str): Name of the model (LLM)
        max_tokens (int): The maximum number of tokens used for the response 
        of the API
        prompt (str): Prompt for the model
    
    Returns:
        response (str): The response from the API
        usage (dict): LLM API usage
    """
    if "gpt" in model:
        try:
            client = openai.OpenAI(api_key=OPENAI_KEY)
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            usage = {"input_tokens": response.usage.prompt_tokens,
                     "output_tokens": response.usage.completion_tokens,
                     "total_tokens": response.usage.total_tokens}
            response = response.choices[0].message.content
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)
        
    elif "claude" in model:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            response = client.messages.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            usage = response.usage.to_dict()
            usage["total_tokens"] = usage["input_tokens"] + \
                                    usage["output_tokens"]
            response = response.content[0].text
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)
    
    elif "gemini" in model:
        try:
            genai.configure(api_key=GOOGLE_KEY)
            generation_config = {
            "temperature": 0,
            "max_output_tokens": max_tokens
            }
            model2use = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
            )
            chat_session = model2use.start_chat(history=[])
            response = chat_session.send_message(prompt)
            usage = {"input_tokens": response.usage_metadata.prompt_token_count,
                     "output_tokens": response.usage_metadata.candidates_token_count,
                     "total_tokens": response.usage_metadata.total_token_count}
            response = response.text
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)
    else:
        raise Exception("No valid model")
    
    return response, usage


def format_response(specified_format: str, response: str) -> str:
    """Format the response accordingly

    Args:
        specified_format (str): The type of formatting to use. It can be 
        'singular' for a singular query string or 'var' for the 
        decomposition in variables
        response (str): The response from the LLM
        
    Returns:
        formatted_response (str): The response ready to be used in the database
    """
    if specified_format == "sql":
        formatted_response = response.split("```sql")[1].split("```")[0] \
        .replace("```", "").replace("```sql", "")
    elif specified_format == "python":
        formatted_response = response.split("```python")[1].split("```")[0] \
        .replace("```", "").replace("```python", "")
    else:
        raise Exception("No valid format specified")
    
    return formatted_response


def run_query(specified_format: str, formatted_response: str, 
              engine: sqlalchemy.engine.base.Engine) -> pd.DataFrame:
    """Function to run the SQL query in the database

    Args:
        specified_format (str): The type of formatting to use. It can be 
        'singular' for a singular query string or 'var' for the 
        decomposition in variables
        formatted_response (str): The response ready to be used in the database
        engine (sqlalchemy.engine.base.Engine): The engine to access the 
        database
        
    Returns:
        results (pandas.DataFrame): Pandas DataFrame with the results of the 
        query
    """
    if specified_format == "sql":
        try: 
            results = pd.read_sql_query(formatted_response, con=engine)
        except Exception as e:
            raise Exception(f"Running SQL exception: {e}")
    elif specified_format == "python":
        try:
            exec(formatted_response, globals())
            results = pd.read_sql_query(full_query, con=engine)
        except Exception as e:
           raise Exception(f"Running SQL exception: {e}")
    else:
        raise Exception("No valid format specified")
    
    return results


def classify(query: str, model: str) -> tuple[str, dict]:
    """Function to classify the difficulty of a NL query

    Args:
        query (str): NL query
        model (str): LLM to classify the query
        
    Returns:
        label (str): Label of the difficulty level of the query. It can be
        'simple', 'medium' or 'advanced'.
        usage (dict): LLM API usage
    """
    # Make the difficulty classification prompt
    prompt = diff_class_prompt + \
    f"\nThe request to classify is the following: {query}"
    
    # Obtain the difficulty label
    label, usage = api_call(model, 20, prompt)
    labels = ["simple", "medium", "advanced"]
    true_label = [l for l in labels if l in label]
    label = true_label[0]
    return label, usage


def schema_linking(query: str, model: str) -> tuple[str, dict]:
    """Function to make the schema linking of a NL query. This means it will
    obtain the tables necessary to create the corresponding SQL query 

    Args:
        query (str): NL query
        model (str): LLM to obtain the necessary tables
        
    Returns:
        tables (str): A string of a list of the tables needed to create the 
        query
        usage (dict): LLM API usage
    """
    # Make the schema linking prompt
    prompt = tables_linking_prompt_V2 + \
        f"\nThe user request is the following: {query}"
        
    # Obtain the tables necessary for the SQL query
    tables, usage = api_call(model, 100, prompt)
    content = tables.split("[")[1].split("]")[0]
    true_tables = f"[{content}]"
    return true_tables, usage


def schema_linking_v2(query: str, model: str) -> tuple[str, dict]:
    """Function to make the schema linking of a NL query. This means it will
    obtain the tables necessary to create the corresponding SQL query 

    Args:
        query (str): NL query
        model (str): LLM to obtain the necessary tables
        
    Returns:
        tables (str): A string of a list of the tables needed to create the 
        query
        usage (dict): LLM API usage
    """
    # Make the schema linking prompt
    prompt = sch_linking.format(
        ur = query,
        astro_context = astro_context
    )
        
    # Obtain the tables necessary for the SQL query
    tables, usage = api_call(model, 100, prompt)
    content = tables.split("[")[1].split("]")[0]
    true_tables = f"[{content}]"
    return true_tables, usage


def decomposition(label: str, ur_w_tables: str, model: str, 
                  format: str) -> tuple[str, dict]:
    """Function to create the decomposition prompts

    Args:
        label (str): Difficulty label
        ur_w_tables (str): User request with the needed tables from the DB
        model (str): Name of the model (LLM)
        format (str): The type of formatting to use. It can be 
        'sql' for a singular query string or 'python' for the 
        decomposition in Python variables
        
    Returns:
        prompt (str): Prompt to use in the decomposition task of a NL query
        usage (dict): LLM API usage
    """
    if label == "simple":
        # Simple queries don't need decomposition
        prompt = final_prompt_simple_vf.format(
                simple_query_task = simple_query_task_vf, 
                simple_query_cntx = simple_query_cntx_vf,
                simple_query_instructions = simple_query_instructions_vf,
                request = ur_w_tables
        )
        # No usage needed for the simple query. There is no decomposition
        usage = None
        
    elif label == "medium":
        # Getting the decomposition plan
        decomp_plan = medium_decomp_prompt_vf.format(
                medium_decomp_task = medium_decomp_task_vf,
                medium_query_cntx = medium_query_cntx_vf,
                user_request_with_tables = ur_w_tables,
                medium_query_instructions_1 = medium_query_instructions_1_vf
            )
        decomp_plan_true, usage = api_call(model, 1000, decomp_plan)
        # Creating the final prompt with the decomposition plan
        if format == "sql":
            # Through SQL queries
            prompt = medium_decomp_gen_vf.format(
                medium_query_task = medium_query_task_vf,
                user_request_with_tables = ur_w_tables,
                medium_query_instructions_2 = medium_query_instructions_2_vf,
                decomp_plan = decomp_plan_true
            )
        else:
            # Through Python variables
            prompt = medium_decomp_gen_vf_python.format(
                medium_query_task = medium_query_task_vf,
                user_request_with_tables = ur_w_tables,
                medium_query_instructions_2 = medium_query_instructions_2_vf,
                decomp_plan = decomp_plan_true
            )
            
    elif label == "advanced":
        # Getting the decomposition plan
        decomp_plan = adv_decomp_prompt_vf.format(
            adv_decomp_task = adv_decomp_task_vf,
            adv_query_cntx = adv_query_cntx_vf,
            user_request_with_tables = ur_w_tables,
            adv_query_instructions_1 = adv_query_instructions_1_vf
        )
        decomp_plan_true, usage = api_call(model, 1000, decomp_plan)
        # Creating the final prompt with the decomposition plan
        if format == "sql":
            # Through SQL queries
            prompt = adv_decomp_gen_vf.format(
                adv_query_task = adv_query_task_vf,
                user_request_with_tables = ur_w_tables,
                adv_query_instructions_2 = adv_query_instructions_2_vf,
                decomp_plan = decomp_plan_true
            )
        else:
            # Through Python variables
            prompt = adv_decomp_gen_vf_python.format(
                adv_query_task = adv_query_task_vf,
                user_request_with_tables = ur_w_tables,
                adv_query_instructions_2 = adv_query_instructions_2_vf,
                decomp_plan = decomp_plan_true
            )
        
    else:
        raise Exception("No valid label difficulty")
    
    return prompt, usage


def decomposition_v2(label: str, ur: str, tables: str, model: str, 
                     format: str) -> tuple[str, dict]:
    """Function to create the decomposition prompts

    Args:
        label (str): Difficulty label
        ur (str): User request
        tables (str): Tables from the DB needed for the request
        model (str): Name of the model (LLM)
        format (str): The type of formatting to use. It can be 
        'sql' for a singular query string or 'python' for the 
        decomposition in Python variables
        
    Returns:
        prompt (str): Prompt to use in the query generation task of a NL query
        usage (dict): LLM API usage
    """
    if label == "simple":
        # Simple queries don't need decomposition
        prompt = query_sql_simple.format(
            ur = ur,
            tables = tables
        )
        # No usage needed for the simple query. There is no decomposition
        usage = None
        decomp_plan = None
        
    elif label == "medium":
        # Getting the decomposition plan
        decomp_plan = decomp_medium.format(
            ur = ur,
            tables = tables,
            astro_context = astro_context
        )
        decomp_plan_true, usage = api_call(model, 1000, decomp_plan)
        # Creating the final prompt with the decomposition plan
        if format == "sql":
            # Through SQL queries
            prompt = query_sql_medium.format(
                ur = ur,
                tables = tables,
                decomp_plan = decomp_plan_true
            )
        else:
            # Through Python variables
            prompt = query_python_medium.format(
                ur = ur,
                tables = tables,
                decomp_plan = decomp_plan_true
            )
            
    elif label == "advanced":
        # Getting the decomposition plan
        decomp_plan = decomp_advanced.format(
            ur = ur,
            tables = tables,
            astro_context = astro_context
        )
        decomp_plan_true, usage = api_call(model, 1000, decomp_plan)
        # Creating the final prompt with the decomposition plan
        if format == "sql":
            # Through SQL queries
            prompt = query_sql_advanced.format(
                ur = ur,
                tables = tables,
                decomp_plan = decomp_plan_true
            )
        else:
            # Through Python variables
            prompt = query_python_advanced.format(
                ur = ur,
                tables = tables,
                decomp_plan = decomp_plan_true
            )
        
    else:
        raise Exception("No valid label difficulty")
    
    return prompt, decomp_plan, usage


def pricing(usage: dict, model: str) -> dict:
    """Function to obtain the cost of the usage of the LLMs in the pipeline

    Args:
        usage (dict): Dictionary with all the tokens used in the pipeline
        model (str): Name of the model (LLM)
        
    Returns:
        usage (dict): Augmented the token dictionary with the respective costs
    """
    # Prices dictionary (hard-coded)
    # The prices are in US dollars and for every 1M tokens
    prices = {
        "gpt-4o": {
            "input": 2.50,
            "output": 10
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.6
        },
        "o1-preview": {
            "input": 15,
            "output": 60
        },
        "o1-mini": {
            "input": 3,
            "output": 12
        },
        "claude-3-5-sonnet": {
            "input": 3,
            "output": 15
        }
    }
    
    # Checking the corresponding model
    m = [key for key in prices.keys() if key in model][0]
    
    for key in usage.keys():
        # Obtaining the respective costs
        input_cost = prices[m]["input"] * usage[key]["input_tokens"] / 1e6
        output_cost = prices[m]["output"] * usage[key]["output_tokens"] / 1e6
        total_cost = input_cost + output_cost
                
        # Augmenting the usage dictionary
        usage[key]["input_cost"] = input_cost
        usage[key]["output_cost"] = output_cost
        if "total_cost" in usage[key].keys():
            usage[key]["new_total_cost"] = total_cost
        else:
            usage[key]["total_cost"] = total_cost
                    
    return usage


def direct_prompts(label: str, ur: str, tables: str) -> str:
    """Creating simple direct prompts for query generation

    Args:
        label (str): Difficulty label
        ur (str): User request
        tables (str): Tables from the DB needed for the request
        
    Returns:
        direct_prompt (str): Prompt for query generation (direct approach)
    """
    if label == "simple":
        direct_prompt = query_sql_simple.format(
            ur = ur,
            tables = tables
        )
    elif label == "medium":
        direct_prompt = query_direct_sql_medium.format(
            ur = ur,
            tables = tables
        )
    elif label == "medium":
        direct_prompt = query_direct_sql_advanced.format(
            ur = ur,
            tables = tables
        )
    else:
        raise Exception("No valid label difficulty")
    
    return direct_prompt