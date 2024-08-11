import openai
import anthropic 

from ragStep import rag_step
from config import OPENAI_KEY, ANTHROPIC_KEY


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
            print(f"The following excpetion occured: {e}")
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
            print(f"The following excpetion occured: {e}")
            raise Exception(e)
        
    else:
        raise Exception("No valid model")
    
    return response