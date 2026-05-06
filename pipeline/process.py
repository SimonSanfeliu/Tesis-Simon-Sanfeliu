import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import openai
import anthropic
import re
import requests
import sqlalchemy
#from google import genai

from secret.config import OPENAI_KEY, ANTHROPIC_KEY, GOOGLE_KEY
from prompts.classification.Classification import diff_class_prompt_v7, \
    final_instructions_diff_v2
from prompts.schema_linking.SchemaLinking import tables_linking_prompt_V2
from prompts.decomposition.Decomposition import final_prompt_simple_vf, \
    simple_query_task_vf, simple_query_cntx_vf, simple_query_instructions_vf
from prompts.decomposition.Decomposition import *
from prompts.decomposition.Decomposition import adv_decomp_prompt_vf, \
    adv_decomp_gen_vf, adv_decomp_gen_vf_python, adv_query_task_vf, \
    adv_query_cntx_vf, adv_query_instructions_1_vf, \
    adv_query_instructions_2_vf, adv_decomp_task_vf
from final_prompts.final_prompts import *
from prompts.schema_linking.DBSchema import schema_all_cntxV1, schema_all_cntxV2_indx, schema_all_cntxV2

# Setting up astronomical context
with open("final_prompts/astrocontext.txt", "r") as f:
    astro_context = f.read()


MODEL_PRICES = {
    "gpt-4o": {
        "input": 2.50,
        "output": 10
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.6
    },
    "gpt-5.2-codex": {
        "input": 1.50,
        "output": 6
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
    },
    "claude-4.6-opus": {
        "input": 15,
        "output": 75
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10
    }
}


def get_model_provider(model: str) -> str:
    """Resolve the provider family from a model name."""
    model = model.lower().strip()

    if model.startswith("gpt-") or "codex" in model or model.startswith("o1"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"

    raise Exception(f"No valid model: {model}")


def get_price_key(model: str) -> str:
    """Map a model name to the configured pricing key."""
    model = model.lower().strip()
    matches = [key for key in MODEL_PRICES.keys() if key in model]
    if not matches:
        raise Exception(f"No pricing configured for model: {model}")
    return max(matches, key=len)


def _extract_responses_api_text(payload: dict) -> str:
    """Extract plain text from a raw Responses API payload."""
    output_items = payload.get("output", [])
    text_chunks = []

    for item in output_items:
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text_chunks.append(content.get("text", ""))

    if text_chunks:
        return "".join(text_chunks).strip()

    if payload.get("output_text"):
        return str(payload["output_text"]).strip()

    raise ValueError("Responses API payload did not contain output text.")


def _responses_api_call(model: str, max_tokens: int, prompt: str,
                        text_format: dict | None = None) -> tuple[str, dict]:
    """Call OpenAI Responses API directly.

    This keeps support for models such as gpt-5.2-codex even when the local
    OpenAI SDK is older and does not expose client.responses yet.
    """
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens
    }
    if text_format is not None:
        payload["text"] = {"format": text_format}

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=180
    )
    response.raise_for_status()
    payload = response.json()
    usage_payload = payload.get("usage", {})
    usage = {
        "input_tokens": usage_payload.get("input_tokens", 0),
        "output_tokens": usage_payload.get("output_tokens", 0),
        "total_tokens": usage_payload.get("total_tokens", 0)
    }
    return _extract_responses_api_text(payload), usage


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
    provider = get_model_provider(model)
    model_lower = model.lower().strip()

    if provider == "openai":
        try:
            if model_lower == "gpt-5.2-codex":
                return _responses_api_call(model, max_tokens, prompt)

            client = openai.OpenAI(api_key=OPENAI_KEY)
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            if model_lower.startswith("o1") or model_lower.startswith("gpt-5") or "codex" in model_lower:
                request_kwargs["max_completion_tokens"] = max_tokens
            else:
                request_kwargs["temperature"] = 0
                request_kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**request_kwargs)
            usage = {"input_tokens": response.usage.prompt_tokens,
                     "output_tokens": response.usage.completion_tokens,
                     "total_tokens": response.usage.total_tokens}
            response = response.choices[0].message.content
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)

    elif provider == "anthropic":
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
    
    elif provider == "google":
        try:
            client = genai.Client(api_key=GOOGLE_KEY)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": 0,
                    "max_output_tokens": max_tokens
                }
            )
            usage = {"input_tokens": response.usage_metadata.prompt_token_count,
                     "output_tokens": response.usage_metadata.candidates_token_count,
                     "total_tokens": response.usage_metadata.total_token_count}
            response = response.text
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)
    
    return response, usage


def parse_schema_linking_tables(raw_response: str, valid_table_names) -> list[str]:
    """Extract valid schema table names from a model response.

    The older pipeline expects a literal Python-like list such as
    ``['object', 'probability']``. Newer models sometimes prepend free-form
    reasoning or explanations, so we recover the table names defensively.
    """
    if raw_response is None:
        raise ValueError("Schema linking returned an empty response.")

    valid_names = {str(name).strip(): str(name).strip() for name in valid_table_names}
    lowered_lookup = {name.lower(): name for name in valid_names}
    raw_text = str(raw_response).strip()

    def _clean_token(token: str) -> str:
        return token.strip().strip("'\"`").strip()

    ordered_matches = []
    seen = set()

    bracket_match = re.search(r"\[(.*?)\]", raw_text, flags=re.DOTALL)
    if bracket_match:
        candidates = [_clean_token(part) for part in bracket_match.group(1).split(",")]
        for candidate in candidates:
            resolved = lowered_lookup.get(candidate.lower())
            if resolved and resolved not in seen:
                ordered_matches.append(resolved)
                seen.add(resolved)

    if ordered_matches:
        return ordered_matches

    text_lower = raw_text.lower()
    indexed_matches = []
    for lowered_name, original_name in lowered_lookup.items():
        pattern = rf"(?<![\w]){re.escape(lowered_name)}(?![\w])"
        match = re.search(pattern, text_lower)
        if match:
            indexed_matches.append((match.start(), original_name))

    indexed_matches.sort(key=lambda item: item[0])
    for _, original_name in indexed_matches:
        if original_name not in seen:
            ordered_matches.append(original_name)
            seen.add(original_name)

    if ordered_matches:
        return ordered_matches

    preview = raw_text[:250].replace("\n", "\\n")
    raise ValueError(
        "Schema linking response did not contain recognizable table names. "
        f"Raw response preview: {preview}"
    )


def parse_classification_label(raw_response: str) -> str:
    """Extract one difficulty label from a free-form classifier response."""
    if raw_response is None:
        raise ValueError("Classification returned an empty response.")

    matches = list(re.finditer(r"\b(simple|medium|advanced)\b", str(raw_response).lower()))
    if not matches:
        preview = str(raw_response).strip()[:250].replace("\n", "\\n")
        raise ValueError(
            "Classification response did not contain a valid difficulty label. "
            f"Raw response preview: {preview}"
        )
    return matches[0].group(1)


def unify_decomposition_steps(steps: list[str]) -> str:
    """Join structured decomposition steps into the plain-text format used by
    the downstream prompts.

    Args:
        steps (list[str]): Ordered decomposition steps

    Returns:
        str: Numbered decomposition plan as a single string
    """
    clean_steps = [step.strip() for step in steps if step and step.strip()]
    return "\n\n".join(
        f"{idx}. {step}" for idx, step in enumerate(clean_steps, start=1)
    )


def decomposition_api_call(model: str, max_tokens: int, prompt: str) -> tuple[str, dict]:
    """Create a structured-output API call for the decomposition stage.

    The response is normalized back into the numbered plain-text format already
    expected by the query-generation prompts.

    Args:
        model (str): Name of the model (LLM)
        max_tokens (int): Maximum output tokens
        prompt (str): Prompt for the decomposition step

    Returns:
        tuple[str, dict]: Unified numbered decomposition plan and API usage
    """
    if "gpt" not in model:
        return api_call(model, max_tokens, prompt)

    schema_prompt = prompt + """

# Return the decomposition as structured data.
# The JSON root must contain a "steps" array.
# Each item in "steps" must be an object with a single key "step".
# The "step" value must contain one complete decomposition step, including any
# relevant labels such as [sub-query], [join], [condition], etc.
# Do not return SQL code.
"""
    schema_definition = {
        "name": "decomposition_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string"
                            }
                        },
                        "required": ["step"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["steps"],
            "additionalProperties": False
        }
    }

    try:
        if model.lower().strip() == "gpt-5.2-codex":
            content, usage = _responses_api_call(
                model,
                max_tokens,
                schema_prompt,
                text_format={
                    "type": "json_schema",
                    **schema_definition
                }
            )
            parsed = json.loads(content)
            decomp_plan = unify_decomposition_steps(
                [item["step"] for item in parsed["steps"]]
            )
            return decomp_plan, usage

        client = openai.OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": schema_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema_definition
            }
        )
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        content = response.choices[0].message.content
        parsed = json.loads(content)
        decomp_plan = unify_decomposition_steps(
            [item["step"] for item in parsed["steps"]]
        )
        return decomp_plan, usage
    except Exception as e:
        print(f"The following exception occured in decomposition_api_call: {e}")
        raise Exception(e)


def format_response(specified_format: str, response: str) -> str:
    """Format the response accordingly

    Args:
        specified_format (str): The type of formatting to use. It can be 
        'sql' for a singular query string or 'python' for the 
        decomposition in Python variables
        response (str): The response from the LLM
        
    Returns:
        formatted_response (str): The response ready to be used in the database
    """
    response = str(response).strip()

    if specified_format == "sql":
        sql_match = re.search(r"```sql\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        generic_match = re.search(r"```\s*(.*?)```", response, flags=re.DOTALL)
        if sql_match:
            formatted_response = sql_match.group(1)
        elif generic_match:
            formatted_response = generic_match.group(1)
        else:
            formatted_response = response.removeprefix("sql").strip()

    elif specified_format == "python":
        py_match = re.search(r"```python\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
        generic_match = re.search(r"```\s*(.*?)```", response, flags=re.DOTALL)
        if py_match:
            formatted_response = py_match.group(1)
        elif generic_match:
            formatted_response = generic_match.group(1)
        else:
            formatted_response = response.removeprefix("python").strip()
        formatted_response = formatted_response.replace('"""""', '"""')

    else:
        raise Exception("No valid format specified")
    
    # Adding more formatting
    formatted_response = formatted_response.replace(";", "")  # .replace("\n", "")
    
    return formatted_response


def run_query(specified_format: str, formatted_response: str, 
              engine: sqlalchemy.engine.base.Engine) -> pd.DataFrame:
    """Function to run the SQL query in the database

    Args:
        specified_format (str): The type of formatting to use. It can be 
        'sql' for a singular query string or 'python' for the 
        decomposition in variables
        formatted_response (str): The response ready to be used in the database
        engine (sqlalchemy.engine.base.Engine): The engine to access the 
        database
        
    Returns:
        results (pandas.DataFrame): Pandas DataFrame with the results of the 
        query
    """
    results = None
    error = ""
    if specified_format == "sql":
        try: 
            results = pd.read_sql_query(formatted_response, con=engine)
        except Exception as e:
            error = e
            print(f"Running SQL exception in run_query: {e}", flush=True)
    elif specified_format == "python":
        try:
            exec(formatted_response, globals())
            results = pd.read_sql_query(full_query, con=engine)
        except Exception as e:
            error = e
            print(f"Running SQL exception in run_query: {e}", flush=True)
    else:
        error = "No valid format specified"
    
    return results, error


def classify(query: str, table_schema: str, model: str) -> tuple[str, str, dict]:
    """Function to classify the difficulty of a NL query

    Args:
        query (str): NL query
        table_schema (str): Tables needed for the query
        model (str): LLM to classify the query
        
    Returns:
        label (str): Label of the difficulty level of the query. It can be
        'simple', 'medium' or 'advanced'.
        prompt (str): Prompt used to classify the query
        usage (dict): LLM API usage
    """
    # Make the difficulty classification prompt
    diff_class_prompt = diff_class_prompt_v7.format(
        table_schema = table_schema,
        final_instructions_diff = final_instructions_diff_v2
    )
    prompt = diff_class_prompt + \
    f"\nThe request to classify is the following: {query}"
    
    # Obtain the difficulty label
    raw_label, usage = api_call(model, 1000, prompt)
    label = parse_classification_label(raw_label)
    return label, prompt, usage


def schema_linking(query: str, model: str) -> tuple[str, dict]:
    """Function to make the schema linking of a NL query. This means it will
    obtain the tables necessary to create the corresponding SQL query 

    Args:
        query (str): NL query
        model (str): LLM to obtain the necessary tables
        
    Returns:
        tables (str): A string of a list of the tables needed to create the 
        query with their respective information
        usage (dict): LLM API usage
    """
    # Make the schema linking prompt
    prompt = tables_linking_prompt_V2 + \
        f"\nThe user request is the following: {query}"
        
    # Obtain the tables necessary for the SQL query
    tables, usage = api_call(model, 1000, prompt)
    content = parse_schema_linking_tables(tables, schema_all_cntxV1.keys())
    true_tables = f"{[schema_all_cntxV1[c] for c in content]}"
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
        ur = query
    )
        
    # Obtain the tables necessary for the SQL query
    tables, usage = api_call(model, 1000, prompt)
    content = parse_schema_linking_tables(tables, schema_all_cntxV1.keys())
    true_tables = f"{[schema_all_cntxV1[c] for c in content]}"
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
        usage = {"input_tokens": 0, "output_tokens": 0}
        
    elif label == "medium":
        # Getting the decomposition plan
        decomp_plan = medium_decomp_prompt_vf.format(
                medium_decomp_task = medium_decomp_task_vf,
                medium_query_cntx = medium_query_cntx_vf,
                user_request_with_tables = ur_w_tables,
                medium_query_instructions_1 = medium_query_instructions_1_vf
            )
        decomp_plan_true, usage = api_call(model, 5000, decomp_plan)
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
        decomp_plan_true, usage = api_call(model, 5000, decomp_plan)
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
        usage = {"input_tokens": 0, "output_tokens": 0}
        decomp_plan = ""
        
    elif label == "medium":
        # Getting the decomposition plan
        decomp_plan = decomp_medium.format(
            ur = ur,
            tables = tables,
            astro_context = astro_context
        )
        decomp_plan_true, usage = api_call(model, 5000, decomp_plan)
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
        decomp_plan_true, usage = api_call(model, 5000, decomp_plan)
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
    m = get_price_key(model)
    
    for key in usage.keys():
        # Obtaining the respective costs
        input_cost = MODEL_PRICES[m]["input"] * usage[key]["input_tokens"] / 1e6
        output_cost = MODEL_PRICES[m]["output"] * usage[key]["output_tokens"] / 1e6
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
            tables = tables,
            astro_context = astro_context
        )
    elif label == "advanced":
        direct_prompt = query_direct_sql_advanced.format(
            ur = ur,
            tables = tables,
            astro_context = astro_context
        )
    else:
        raise Exception("No valid label difficulty")
    
    return direct_prompt
