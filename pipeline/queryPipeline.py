import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.process import *
from prompts.base.prompts import *


class queryPipeline():
    """
    Pipeline class
    """
    def __init__(self, model, lang_type, max_tokens, prompts):
        """
        Class generator
        """
        self.model = model
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.prompts = prompts
        self.tab_schema_class = ""
        self.tab_schema_decomp = ""
        self.tab_schema_direct = ""
        self.label = ""
        self.final_prompt = ""  # ¿Dejar aquí el prompt final directo o de descomposición?
        self.usage = {}  # Lo mismo con usage
        
    def schema_linking(self, query):
        """Function to make the schema linking of a NL query. This means it will
        obtain the tables necessary to create the corresponding SQL query 

        Args:
            query (str): NL query
            
        Returns:
            usage (dict): LLM API usage
        """
        # Make the schema linking prompt
        prompt = self.prompts["Schema Linking"]["base_prompt"] + \
            f"\nThe user request is the following: {query}"
            
        # Obtain the tables necessary for the SQL query
        tables, usage = api_call(self.model, 1000, prompt)
        content = tables.strip("[]").replace("'", "").split(", ")
        self.tab_schema_class = f"{[self.prompts['Schema Linking']['context1'][c] for c in content]}"
        self.tab_schema_decomp = f"{[self.prompts['Schema Linking']['context2'][c] for c in content]}"  # Esta usarla para decomposition de Jorge
        self.tab_schema_direct = f"{[self.prompts['Schema Linking']['context3'][c] for c in content]}"  # Esta usarla para direct de Jorge
        return usage
    
    def classify(self, query):
        """Function to classify the difficulty of a NL query

        Args:
            query (str): NL query
            
        Returns:
            usage (dict): LLM API usage
        """
        # Make the difficulty classification prompt
        diff_class_prompt = self.prompts["Classify"]["base_prompt"].format(
            table_schema = self.tab_schema_class,
            final_instructions_diff = self.prompts["Classify"]["final_instructions"]
        )
        prompt = diff_class_prompt + \
        f"\nThe request to classify is the following: {query}"
        
        # Obtain the difficulty label
        label, usage = api_call(self.model, 1000, prompt)
        labels = ["simple", "medium", "advanced"]
        true_label = [l for l in labels if l in label]
        self.label = true_label[0]
        return usage
    
    def decomposition(self, query):
        """Function to create the decomposition prompts

        Args:
            model (str): Name of the model (LLM)
            
        Returns:
            usage (dict): LLM API usage
        """
        # defining query_w_tables
        query_w_tables = query + "\n" + self.tab_schema_decomp
        if self.label == "simple":
            # Simple queries don't need decomposition
            prompt = prompt_inference(self.prompts["Decomposition"]["simple"]["query_task"], 
                                      self.tab_schema_decomp, 
                                      self.prompts["Decomposition"]["simple"]["query_context"], 
                                      self.prompts["Decomposition"]["simple"]["external_knowledge"], 
                                      self.prompts["Decomposition"]["simple"]["domain_knowledge"], 
                                      self.prompts["Decomposition"]["simple"]["query_instructions"])
            prompt += f"\nThe user request is the following: {query}"
            # No usage needed for the simple query. There is no decomposition
            usage = {"input_tokens": 0, "output_tokens": 0}
            
        elif self.label == "medium":
            # Getting the decomposition plan
            decomp_plan = self.prompts["Decomposition"]["medium"]["decomp_plan"]["base_prompt"].format(
                medium_decomp_task = self.prompts["Decomposition"]["medium"]["decomp_plan"]["decomp_task"],
                medium_query_cntx = self.prompts["Decomposition"]["medium"]["decomp_plan"]["query_context"],
                user_request_with_tables = query_w_tables,
                medium_query_instructions_1 = self.prompts["Decomposition"]["medium"]["decomp_plan"]["query_instructions"]
            )
            decomp_plan_true, usage = api_call(self.model, 5000, decomp_plan)
            # Creating the final prompt with the decomposition plan
            if self.lang_type == "sql":
                # Through SQL queries
                prompt = self.prompts["Decomposition"]["medium"]["decomp_gen"]["sql"]["base_prompt"].format(
                    medium_query_task = self.prompts["Decomposition"]["medium"]["decomp_gen"]["sql"]["query_task"],
                    user_request_with_tables = query_w_tables,
                    medium_query_instructions_2 = self.prompts["Decomposition"]["medium"]["decomp_gen"]["sql"]["query_instructions"],
                    decomp_plan = decomp_plan_true
                )
            else:
                # TODO: Asimilar al de SQL corregido
                
                # Through Python variables
                prompt = self.prompts["Decomposition"]["medium"]["decomp_gen"]["python"]["base_prompt"].format(
                    medium_query_task = self.prompts["Decomposition"]["medium"]["decomp_gen"]["python"]["query_task"],
                    user_request_with_tables = query_w_tables,
                    medium_query_instructions_2 = self.prompts["Decomposition"]["medium"]["decomp_gen"]["python"]["query_instructions"],
                    decomp_plan = decomp_plan_true
                )
                
        elif self.label == "advanced":
            # Getting the decomposition plan
            decomp_plan = self.prompts["Decomposition"]["advanced"]["decomp_plan"]["base_prompt"].format(
                adv_decomp_task = self.prompts["Decomposition"]["advanced"]["decomp_plan"]["decomp_task"],
                adv_query_cntx = self.prompts["Decomposition"]["advanced"]["decomp_plan"]["query_context"],
                user_request_with_tables = query_w_tables,
                adv_query_instructions_1 = self.prompts["Decomposition"]["advanced"]["decomp_plan"]["query_instructions"]
            )
            decomp_plan_true, usage = api_call(self.model, 5000, decomp_plan)
            # Creating the final prompt with the decomposition plan
            if self.lang_type == "sql":
                # Through SQL queries
                prompt = self.prompts["Decomposition"]["advanced"]["decomp_gen"]["sql"]["base_prompt"].format(
                    adv_query_task = self.prompts["Decomposition"]["advanced"]["decomp_gen"]["sql"]["query_task"],
                    user_request_with_tables = query_w_tables,
                    adv_query_instructions_2 = self.prompts["Decomposition"]["advanced"]["decomp_gen"]["sql"]["query_instructions"],
                    decomp_plan = decomp_plan_true
                )
            else:
                # TODO: Asimilar al de SQL corregido
                
                # Through Python variables
                prompt = self.prompts["Decomposition"]["advanced"]["decomp_gen"]["python"]["base_prompt"].format(
                    adv_query_task = self.prompts["Decomposition"]["advanced"]["decomp_gen"]["python"]["query_task"],
                    user_request_with_tables = query_w_tables,
                    adv_query_instructions_2 = self.prompts["Decomposition"]["advanced"]["decomp_gen"]["python"]["query_instructions"],
                    decomp_plan = decomp_plan_true
                )
            
        else:
            raise Exception("No valid label difficulty")
        
        return prompt, usage
    
    def direct(self, query):
        """TODO: Docstring apropiado
        """
        # The same direct approach is used for every label
        
        # Base prompt for the direct approach
        base = base_prompt(self.prompts["Direct"]["base_prompt"]["general_task"], 
                           self.prompts["Direct"]["base_prompt"]["general_context"], 
                           self.prompts["Direct"]["base_prompt"]["final_instructions"]
        )
        # Request prompt for the specific query
        req = prompt_request(self.tab_schema_direct, 
                             self.prompts["Direct"]["request_prompt"]["external_knowledge"], 
                             self.prompts["Direct"]["request_prompt"]["domain_knowledge"], 
                             query)
        # Final prompt
        prompt = base + "\n" + req
        
        return prompt
    
    def query_generation(self, prompt):
        """_summary_

        Args:
            prompt (_type_): _description_
        """
        # Obtaining the SQL query
        response, usage = api_call(self.model, self.max_tokens, prompt)
        # print(f"Raw response: {response}", flush=True)
        
        # Catching borderline cases
        if self.lang_type == "python" and self.label == "simple":
            self.lang_type = "sql"
        
        # Formatting the response
        gen_query = format_response(self.lang_type, response)
        
        return gen_query, usage