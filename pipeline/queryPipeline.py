import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import csv
import time
from datetime import datetime

from pipeline.process import *
from pipeline.eval import *
from prompts.base.prompts import *


class queryPipeline():
    def __init__(self, query: str, llm: str, lang_type: str, max_tokens: int, 
                 prompts: dict):
        """Query pipeline class. It has all the components to generate the
        predicted SQL queries from the natural language ones

        Args:
            query (str): Natural language (NL) query
            llm (str): LLM used for the pipeline
            lang_type (str): Programming language used for the queries ('sql' 
            or 'python')
            max_tokens (int): Maximum token output for the LLM
            prompts (dict): Dictionary with all the corresponding prompts
        """
        # Build attributes
        self.query = query
        self.llm = llm
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.prompts = prompts
        
        # Fill-in attributes
        self.tab_schema_class = ""
        self.tab_schema_decomp = ""
        self.tab_schema_direct = ""
        self.label = ""
        self.final_prompt = ""
        self.usage = {}
        
    def schema_linking(self, query: str = None):
        """Function to make the schema linking of a NL query. This means it will
        obtain the tables necessary to create the corresponding SQL query 

        Args:
            query (str, optional): NL query. Defaults to None.
            
        Returns:
            None
        """
        # Check what query to use
        if query is None:
            query = self.query
            
        # Make the schema linking prompt
        prompt = self.prompts["Schema Linking"]["base_prompt"] + \
            f"\nThe user request is the following: {query}"
            
        # Obtain the tables necessary for the SQL query
        tables, usage = api_call(self.llm, 1000, prompt)
        content = tables.strip("[]").replace("'", "").split(", ")
        self.tab_schema_class = f"{[self.prompts['Schema Linking']['context1'][c] for c in content]}"
        self.tab_schema_decomp = f"{[self.prompts['Schema Linking']['context2'][c] for c in content]}"  # Esta usarla para decomposition de Jorge
        self.tab_schema_direct = f"{[self.prompts['Schema Linking']['context3'][c] for c in content]}"  # Esta usarla para direct de Jorge
        
        # Saving the usage
        self.usage["Schema Linking"] = usage
    
    def classify(self, query: str = None):
        """Function to classify the difficulty of a NL query

        Args:
            query (str, optional): NL query. Defaults to None.
            
        Returns:
            None
        """
        # Check what query to use
        if query is None:
            query = self.query
        
        # Make the difficulty classification prompt
        diff_class_prompt = self.prompts["Classify"]["base_prompt"].format(
            table_schema = self.tab_schema_class,
            final_instructions_diff = self.prompts["Classify"]["final_instructions"]
        )
        prompt = diff_class_prompt + \
        f"\nThe request to classify is the following: {query}"
        
        # Obtain the difficulty label
        label, usage = api_call(self.llm, 1000, prompt)
        labels = ["simple", "medium", "advanced"]
        true_label = [l for l in labels if l in label]
        self.label = true_label[0]
        
        # Saving the usage
        self.usage["Classify"] = usage
    
    def decomposition(self, query: str = None):
        """Function to create the decomposition prompts

        Args:
            query (str, optional): NL query. Defaults to None.
            
        Returns:
            None
        """
        # Check what query to use
        if query is None:
            query = self.query
            
        # Defining query_w_tables
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
            decomp_plan_true, usage = api_call(self.llm, 5000, decomp_plan)
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
            decomp_plan_true, usage = api_call(self.llm, 5000, decomp_plan)
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
        
        # Saving the prompt and usage
        self.final_prompt = prompt
        self.usage["Decomposition"] = usage
    
    def direct(self, query: str = None):
        """Function to create the direct prompts

        Args:
            query (str, optional): NL query. Defaults to None.
            
        Returns:
            None
        """
        # Check what query to use
        if query is None:
            query = self.query
        
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
        
        # Saving the prompt
        self.final_prompt = prompt
    
    def query_generation(self):
        """Function to generate the SQL query based on the saved prompts

        Args:
            None
            
        Returns:
            gen_query (str): Generated SQL query 
        """
        # Obtaining the SQL query
        response, usage = api_call(self.llm, self.max_tokens, self.final_prompt)
        
        # Catching borderline cases
        if self.lang_type == "python" and self.label == "simple":
            self.lang_type = "sql"
        
        # Formatting the response
        gen_query = format_response(self.lang_type, response)
        
        # Saving the usage
        self.usage["Query Generation"] = usage
        
        return gen_query
    
    def pricing(self):
        """Function to obtain the cost of the usage of the LLMs in the pipeline

        Args:
            None
            
        Returns:
            None
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
        m = [key for key in prices.keys() if key in self.llm][0]
        
        for key in self.usage.keys():
            # Obtaining the respective costs
            input_cost = prices[m]["input"] * self.usage[key]["input_tokens"] / 1e6
            output_cost = prices[m]["output"] * self.usage[key]["output_tokens"] / 1e6
            total_cost = input_cost + output_cost
                    
            # Augmenting the usage dictionary
            self.usage[key]["input_cost"] = input_cost
            self.usage[key]["output_cost"] = output_cost
            if "total_cost" in self.usage[key].keys():
                self.usage[key]["new_total_cost"] = total_cost
            else:
                self.usage[key]["total_cost"] = total_cost
                 
    def create_conn(self) -> sa.engine.base.Engine:
        """Function to create a connection with ALeRCE's SQL database

        Args:
            None
        Returns:
            sqlalchemy.engine.base.Engine or None: The engine to access the 
            database or prints an error message
        """
        # At the moment, there are only the options of 2 or 10 minutes
        if self.t_conn==2:
            engine = sa.create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}", poolclass=sa.pool.NullPool)
            return engine
            
        elif self.t_conn==10:
            engine = sa.create_engine(f"postgresql+psycopg2://{USER_10}:{PASS_10}@{params['host']}/{params['dbname']}", poolclass=sa.pool.NullPool)
            return engine
        
        else:
            print('Time not avalaible')
            
    def run_query(self, specified_format: str, formatted_response: str, 
              engine: sqlalchemy.engine.base.Engine) -> tuple[pd.DataFrame, str]:
        """Function to run the SQL query in the database

        Args:
            specified_format (str): The type of formatting to use. It can be 
            'sql' for a singular query string or 'python' for the 
            decomposition in variables
            formatted_response (str): The response ready to be used in the 
            database engine (sqlalchemy.engine.base.Engine): The engine to 
            access the database
            
        Returns:
            results (pandas.DataFrame): Pandas DataFrame with the results of 
            the query
            error (str): If there is an error, it is described here as a 
            string. If there is none, then None is returned (pun intended)
        """
        results = None
        error = None
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

    def run_sql_alerce(self, sql: str) -> tuple[pd.DataFrame, str]:
        """Execute the SQL query at the ALeRCE database and return the result
        
        Args:
            sql (str): SQL query to execute
            
        Returns:
            query (pandas.DataFrame): The result of the query 
            error (str): Error message if the query could not be executed. None 
            if there is no error
        """
        # Create the instance
        engine = self.create_conn()
        query = None
        with engine.connect() as conn:
            # Try the query a number of times
            for n_ in range(0, self.n_tries):
                error = None
                with engine.begin() as conn:
                    try:
                        query, error = self.run_query(self.lang_type, sql, 
                                                      conn)
                        break
                    except Exception as e:
                        error = e
                        continue
        engine.dispose()
        return query, error
    
    def run_pipeline(self, query: str, use_rag: bool = False, 
                     use_direct_prompts: bool = False, 
                     self_corr: bool = False) -> tuple[pd.DataFrame, str, str]:
        """Function to run the whole SQL prediction pipeline

        Args:
            df (pandas.DataFrame): DataFrame with all the NL queries and their
            context
            use_rag (bool): Indicates if the pipeline will be using RAG. 
            Defaults to False
            use_direct_prompts (bool): Indicates if the direct prompts are 
            going to be used or the decomposition ones. Defaults to False
            self_corr (bool): Indicates if the self-correction step is going to 
            be used. Defaults to False
        
        Returns:
            result (pandas.DataFrame): Table with the results of the generated 
            SQL query
            error (str): Error message if the query could not be executed. None 
            if there is no error
            sql_pred (str): Predicted SQL query by the pipeline
        """
        # Using the recreated pipeline
        if not use_rag:
            # Creating the pipeline
            pipe = queryPipeline(
                query,
                self.llm, 
                self.lang_type, 
                self.max_tokens, 
                self.prompts
            )
            
            # Schema Linking
            pipe.schema_linking(query)

            # Classification
            pipe.classify(query)

            if use_direct_prompts:
                # Direct prompt
                pipe.direct(query)
                tables = pipe.tab_schema_direct
            else:
                # Decomposition
                pipe.decomposition(query)
                tables = pipe.tab_schema_decomp
            
            # Generating the queries
            sql_pred = pipe.query_generation()
            
            # If self-correction is enabled, use the respective prompts to correct
            if self_corr:
                # Check if there was an error. If there was, correct it
                result, error = self.run_sql_alerce(sql_pred)
                
                # Correct it in the appropiate format      
                if self.lang_type == "sql":
                    # Correcting the generated SQL
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1, 
                        tab_schema=tables, 
                        req=query, 
                        sql_pred=sql_pred, 
                        error=str(error))
                    new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                    new = format_response(self.lang_type, new)
                    
                    # Adding prices and prompts
                    pipe.usage["Self-correction"] = new_usage
                    pipe.pricing()
                    pipe.prompts["Self-correction"] = corr_prompt
                    
                    # Run the corrected query
                    result, error = self.run_sql_alerce(new)
                    
                    # Standarizing the return variable
                    sql_pred = new
                    
                elif self.lang_type == "python" and pipe.label == "simple":
                    # Border case
                    self.lang_type = "sql"
                    # Correcting the generated SQL
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1, 
                        tab_schema=tables, 
                        req=query, 
                        sql_pred=sql_pred, 
                        error=str(error))
                    new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                    new = format_response(self.lang_type, new)
                    
                    # Adding prices and prompts
                    pipe.usage["Self-correction"] = new_usage
                    pipe.pricing()
                    pipe.prompts["Self-correction"] = corr_prompt
                    
                    # Run the corrected query  
                    result, error = self.run_sql_alerce(new)
                    
                    # Standarizing the return variable
                    sql_pred = new
                            
                else:
                    # Correcting the generated SQL
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1_python, 
                        tab_schema=tables, 
                        req=query, 
                        sql_pred=sql_pred, 
                        error=str(error))
                    new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                    new = format_response(format, new)
                    
                    # Adding prices and prompts
                    pipe.usage["Self-correction"] = new_usage
                    pipe.pricing()
                    pipe.prompts["Self-correction"] = corr_prompt

                    # Run the corrected query
                    result, error = self.run_sql_alerce(new)
                    
                    # Standarizing the return variable
                    sql_pred = new

            # W/o self-correction
            else:
                result, error = self.run_sql_alerce(sql_pred)
                
        # TODO: Add new pipeline
                
        return result, error, sql_pred
    
    def run_experiments(self, df: pd.DataFrame, total_exps: int = 10, 
                        restart: bool = False):
        """Run all the experiments neccessary given a pandas DataFrame with all
        the NL queries and their respective context

        Args:
            df (pd.DataFrame): Pandas DataFrame with all the queries and their 
            context
            total_exps (int, optional): Total number of experiments for each 
            query. Defaults to 10.
            restart (bool): Indicates if the experiment proccess must be 
            restarted. Defaults to False
        """
        # Name of the file to save the predicted queries
        file_path = f"preds_{self.llm}_{datetime.now().isoformat(timespec='seconds')}".replace(":", "-")
        
        # Columns to use
        column_names = ['query_id', 'query_run', 'sql_query', 'query_results', 
                   'query_error', 'query_gen_time', 'code_tag', 'llm_used']
        
        # Generate an empty DataFrame with the corresponding columns and rows
        num_rows = len(df) * total_exps
        new_df = pd.DataFrame([[None]*len(column_names) for _ in range(num_rows)], columns=column_names)
        
        for _, row in df.iterrows():
            pass