# NL query -> SL -> classify -> decomp -> SQL query
#                             -> direct
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.process import *
from prompts.base.prompts import *


class queryPipeline():
    """
    Pipeline class
    """
    def __init__(self, model, format, max_tokens):
        """
        Class generator
        """
        self.model = model
        self.format = format
        self.max_tokens = max_tokens
        
    def schema_linking(self, query):
        """Function to make the schema linking of a NL query. This means it will
        obtain the tables necessary to create the corresponding SQL query 

        Args:
            query (str): NL query
            
        Returns:
            tables (str): A string of a list of the tables needed to create the 
            query with their respective information
            usage (dict): LLM API usage
        """
        # Make the schema linking prompt
        prompt = tables_linking_prompt_V2 + \
            f"\nThe user request is the following: {query}"
            
        # Obtain the tables necessary for the SQL query
        tables, usage = api_call(self.model, 1000, prompt)
        content = tables.strip("[]").replace("'", "").split(", ")
        true_tables_1 = f"{[schema_all_cntxV1[c] for c in content]}"
        true_tables_2 = f"{[schema_all_cntxV2_indx[c] for c in content]}"  # Esta usarla para decomposition de Jorge
        true_tables_3 = f"{[schema_all_cntxV2[c] for c in content]}"  # Esta usarla para direct de Jorge
        return true_tables_1, true_tables_2, true_tables_3, usage
    
    def classify(self, query, table_schema):
        """Function to classify the difficulty of a NL query

        Args:
            query (str): NL query
            table_schema (str): Tables needed for the query
            
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
        label, usage = api_call(self.model, 1000, prompt)
        labels = ["simple", "medium", "advanced"]
        true_label = [l for l in labels if l in label]
        label = true_label[0]
        return label, usage
    
    def decomposition(self, label, query, table_schema, ext_kn, dom_kn):
        """Function to create the decomposition prompts

        Args:
            label (str): Difficulty label
            query_w_tables (str): User request with the needed tables from the DB
            model (str): Name of the model (LLM)
            format (str): The type of formatting to use. It can be 
            'sql' for a singular query string or 'python' for the 
            decomposition in Python variables
            
        Returns:
            prompt (str): Prompt to use in the decomposition task of a NL query
            usage (dict): LLM API usage
        """
        # defining query_w_tables
        query_w_tables = query + "\n" + table_schema
        if label == "simple":
            # Simple queries don't need decomposition
            prompt = prompt_inference(simple_query_task_v2, table_schema, simple_query_cntx, ext_kn, dom_kn, simple_query_instructions_v2)
            prompt += f"\nThe user request is the following: {query}"
            # No usage needed for the simple query. There is no decomposition
            usage = {"input_tokens": 0, "output_tokens": 0}
            
        elif label == "medium":
            # Getting the decomposition plan
            decomp_plan = medium_decomp_prompt.format(
                    medium_decomp_task = medium_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2,
                    medium_query_cntx = medium_query_cntx,
                    user_request_with_tables = query_w_tables,
                    medium_query_instructions_1 = medium_query_instructions_1_v2
                )
            decomp_plan_true, usage = api_call(self.model, 5000, decomp_plan)
            # Creating the final prompt with the decomposition plan
            if format == "sql":
                # Through SQL queries
                prompt = medium_decomp_gen.format(
                    medium_query_task = medium_query_task_v2,
                    user_request_with_tables = query_w_tables,
                    medium_query_instructions_2 = medium_query_instructions_2_v2,
                    decomp_plan = decomp_plan_true
                )
            else:
                # TODO: Asimilar al de SQL corregido
                
                # Through Python variables
                prompt = medium_decomp_gen_vf_python.format(
                    medium_query_task = medium_query_task_vf,
                    user_request_with_tables = query_w_tables,
                    medium_query_instructions_2 = medium_query_instructions_2_vf,
                    decomp_plan = decomp_plan_true
                )
                
        elif label == "advanced":
            # Getting the decomposition plan
            decomp_plan = adv_decomp_prompt.format(
                adv_decomp_task = adv_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2,
                adv_query_cntx = adv_query_cntx,
                user_request_with_tables = query_w_tables,
                adv_query_instructions_1 = adv_query_instructions_1_v3
            )
            decomp_plan_true, usage = api_call(self.model, 5000, decomp_plan)
            # Creating the final prompt with the decomposition plan
            if format == "sql":
                # Through SQL queries
                prompt = adv_decomp_gen.format(
                    adv_query_task = adv_query_task_v2,
                    user_request_with_tables = query_w_tables,
                    adv_query_instructions_2 = adv_query_instructions_2_v3,
                    decomp_plan = decomp_plan_true
                )
            else:
                # TODO: Asimilar al de SQL corregido
                
                # Through Python variables
                prompt = adv_decomp_gen_vf_python.format(
                    adv_query_task = adv_query_task_vf,
                    user_request_with_tables = query_w_tables,
                    adv_query_instructions_2 = adv_query_instructions_2_vf,
                    decomp_plan = decomp_plan_true
                )
            
        else:
            raise Exception("No valid label difficulty")
        
        return prompt, usage
    
    def direct(self, query, table_schema, ext_kn, dom_kn):
        """TODO: Docstring apropiado
        """
        # The same direct approach is used for every label
        
        # Base prompt for the direct approach
        base = base_prompt(general_taskv18, general_contextv15, final_instructions_v19)
        # Request prompt for the specific query
        req = prompt_request(table_schema, ext_kn, dom_kn, query)
        # Final prompt
        prompt = base + "\n" + req
        
        return prompt
    
    def  direct_v2(self, label, query, table_schema):
        """Creating simple direct prompts for query generation

        Args:
            label (str): Difficulty label
            query (str): User request
            tables (str): Tables from the DB needed for the request
            
        Returns:
            direct_prompt (str): Prompt for query generation (direct approach)
        """
        if label == "simple":
            direct_prompt = query_sql_simple.format(
                query = query,
                tables = table_schema
            )
        elif label == "medium":
            direct_prompt = query_direct_sql_medium.format(
                query = query,
                tables = table_schema,
                astro_context = astro_context
            )
        elif label == "advanced":
            direct_prompt = query_direct_sql_advanced.format(
                query = query,
                tables = table_schema,
                astro_context = astro_context
            )
        else:
            raise Exception("No valid label difficulty")
        
        return direct_prompt
    
    def query_generation(self, label, prompt):
        """_summary_

        Args:
            prompt (_type_): _description_
        """
        # Obtaining the SQL query
        response, usage = api_call(self.model, self.max_tokens, prompt)
        # print(f"Raw response: {response}", flush=True)
        
        # Catching borderline cases
        if self.format == "python" and label == "simple":
            format = "sql"
        
        # Formatting the response
        gen_query = format_response(self.format, response)
        
        return gen_query, usage