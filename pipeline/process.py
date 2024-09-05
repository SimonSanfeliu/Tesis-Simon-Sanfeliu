import pandas as pd
import openai
import anthropic
import google.generativeai as genai

from secret.config import OPENAI_KEY, ANTHROPIC_KEY, GOOGLE_KEY
from prompts.classification.Classification import diff_class_prompt


def api_call(model, max_tokens, prompt):
    """Create the API calls for the LLM to use.

    Args:
        model (str): Name of the model
        max_tokens (int): The maximum number of tokens used for the response 
        of the API
        prompt (str): Prompt for the model
    
    Returns:
        response (str): The response from the API
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
            usage = response.usage
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
            usage = response.usage_metadata  # 'prompt_token_count', 'candidates_token_count' and 'total_token_count'
            response = response.text
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)
    else:
        raise Exception("No valid model")
    
    return response, usage


def format_response(specified_format, response):
    """Format the response accordingly

    Args:
        specified_format (str): The type of formatting to use. It can be 
        'singular' for a singular query string or 'var' for the 
        decomposition in variables
        response (str): The response from the LLM
        
    Returns:
        formatted_response (str or list): The response ready to be used in the
        database. A string if the specified format is 'singular', list of the 
        sub-queries if the format is 'var'
    """
    if specified_format == "singular":
        formatted_response = response.split("```sql")[1].split("```")[0] \
        .replace("```", "").replace("```sql", "")
    elif specified_format == "var":
        formatted_response = response.split("```python")[1].split("```")[0] \
        .replace("```", "").replace("```python", "")
        formatted_response = formatted_response.split("\n\n")
    else:
        raise Exception("No valid format specified")
    
    return formatted_response


def run_query(specified_format, formatted_response, engine):
    """Function to run the SQL query in the database

    Args:
        specified_format (str): The type of formatting to use. It can be 
        'singular' for a singular query string or 'var' for the 
        decomposition in variables
        formatted_response (str or list): The response ready to be used in the
        database. A string if the specified format is 'singular', list of the 
        sub-queries if the format is 'var'
        engine (SQL object): The engine to access the database
        
    Returns:
        results (pd.DataFrame or list): Pandas DataFrame with the results of 
        the query if specified format is 'singular'. List of DataFrames with 
        the results of the subqueries and total query if specified format is 
        'var'
    """
    if specified_format == "singular":
        results = pd.read_sql_query(formatted_response, con=engine)
    elif specified_format == "var":
        results = []
        for query in formatted_response:
            exec(query)
            try:
                results.append(pd.read_sql_query(full_query, con=engine))
            except:
                Exception("No 'full_query' variable generated")
    else:
        raise Exception("No valid format specified")
    
    return results


def classify(query, model):
    """Function to classify the difficulty of a NL query

    Args:
        query (str): NL query
        model (str): LLM to classify the query
        
    Returns:
        label (str): Label of the difficulty level of the query. It can be
        'simple', 'medium' or 'advanced'.
    """
    # Make the difficulty classification prompt
    prompt = diff_class_prompt + \
    f"\nThe request to classify is the following: {query}"
    
    # Obtain the difficulty label
    label = api_call(model, 20, prompt)
    labels = ["simple", "medium", "advanced"]
    true_label = [l for l in labels if l in label]
    return true_label[0]