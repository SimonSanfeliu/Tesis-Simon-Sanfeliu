import openai
import anthropic


def api_call(api_key, model, max_tokens, prompt):
    """Create the API calls for the LLM to use.

    Args:
        api_key (str): Key of the API
        model (str): Name of the model
        max_tokens (int): The maximum number of tokens used for the response of the API
        prompt (str): Prompt for the model
    
    Return:
        response (str): The response from the API
    """
    if "gpt" in model:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response = response.choices[0].message.content
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)
        
    elif "claude" in model:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response = response.content[0].text
        except Exception as e:
            print(f"The following exception occured: {e}")
            raise Exception(e)
        
    else:
        raise Exception("No valid model")
    
    return response


def format_response(specified_format, response):
    """Format the response accordingly

    Args:
        specified_format (str): The type of formatting to use. It can be 'singular' for a singular query string or 'var' for the decomposition in variables
        response (str): The response from the LLM
        
    Return:
        formatted_response: The response ready to be used in the database
    """
    if specified_format == "singular":
        formatted_response = response.split("```sql")[1].split("```")[0].replace("```", "").replace("```sql", "")
    elif specified_format == "var":
        formatted_response = response
    else:
        raise Exception("No valid format specified")