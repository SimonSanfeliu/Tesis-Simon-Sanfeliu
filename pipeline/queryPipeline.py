import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import time
from datetime import datetime

from pipeline.process import *
from pipeline.eval import *
from prompts.base.prompts import *

from logger_setup import setup_logger

logger = setup_logger(name="preds", log_file="logs/preds.txt")


class queryPipeline():
    def __init__(self, query: str, llm: str, lang_type: str, max_tokens: int, 
                 prompts_path: str):
        """Query pipeline class. It has all the components to generate the
        predicted SQL queries from the natural language ones

        Args:
            query (str): Natural language (NL) query
            llm (str): LLM used for the pipeline
            lang_type (str): Programming language used for the queries ('sql' 
            or 'python')
            max_tokens (int): Maximum token output for the LLM
            prompts_path (str): Path to the dictionary with all the 
            corresponding prompts
        """
        # Build attributes
        self.query = query
        self.llm = llm
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.prompts_path = prompts_path
        
        # Reading the prompt file
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)
        
        # Fill-in attributes
        self.tab_schema_class = ""
        self.tab_schema_decomp = ""
        self.tab_schema_direct = ""
        self.label = ""
        self.final_prompt = ""
        self.usage = {}
        self.new_df = None
        
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
    
    def run_pipeline(self, query: str, use_rag: bool = False, 
                     use_direct_prompts: bool = False) -> str:
        """Function to run the whole SQL prediction pipeline

        Args:
            query (str): NL query
            use_rag (bool): Indicates if the pipeline will be using RAG. 
            Defaults to False
            use_direct_prompts (bool): Indicates if the direct prompts are 
            going to be used or the decomposition ones. Defaults to False
        
        Returns:
            sql_pred (str): Predicted SQL query by the pipeline
            tables (str): Table schema used for the query generation
        """
        # Using the recreated pipeline
        if not use_rag:
            # Creating the pipeline
                       
            # Schema Linking
            self.schema_linking(query)

            # Classification
            self.classify(query)
            label = self.label

            if use_direct_prompts:
                # Direct prompt
                self.direct(query)
                tables = self.tab_schema_direct
            else:
                # Decomposition
                self.decomposition(query)
                tables = self.tab_schema_decomp
            
            # Generating the query
            sql_pred = self.query_generation()
                
        # TODO: Add new pipeline
                
        return sql_pred, tables, label
    
    def run_experiments(self, df: pd.DataFrame, total_exps: int = 10, 
                        restart: bool = False, use_rag: bool = False, 
                        use_direct_prompts: bool = False):
        """Run all the experiments neccessary given a pandas DataFrame with all
        the NL queries and their respective context

        Args:
            df (pd.DataFrame): Pandas DataFrame with all the queries and their 
            context
            total_exps (int, optional): Total number of experiments for each 
            query. Defaults to 10.
            restart (bool): Indicates if the experiment proccess must be 
            restarted. Defaults to False
            use_rag (bool): Indicates if the pipeline will be using RAG. 
            Defaults to False
            use_direct_prompts (bool): Indicates if the direct prompts are 
            going to be used or the decomposition ones. Defaults to False
            
        Returns:
            None
        """
        # Name of the file to save the predicted queries
        file_path = f"experiments/preds_{self.llm}_{datetime.now().isoformat(timespec='seconds')}.csv".replace(":", "-")
        bkp_path = "experiments/bkp.csv"
        
        # Columns to use
        column_names = ['code_tag', 'llm_used', 'prompt_version', 'query_id', 
                        'query_run', 'sql_query', 'tab_schema', 'label', 
                        'query_gen_time', 'query_gen_date']
        
        # Generate an empty DataFrame with the corresponding columns and rows
        num_rows = len(df) * total_exps
        self.new_df = pd.DataFrame(
            [[None]*len(column_names) for _ in range(num_rows)], 
            columns=column_names
        )
        
        # Reading the tag
        with open("tag.txt", "r") as f:
            tag = f.read().split("v")[1]
            f.close()
            
        # Reading the prompts' version
        prompt_version = self.prompts_path.split("/prompts_")[1].split(".json")[0]

        # Filling up the first columns
        row_count = 0
        for _, row in df.iterrows():
            for exp in range(total_exps):
                to_fill = [tag, self.llm, prompt_version, row['req_id'], exp+1, None,
                           None, None, None, None]
                self.new_df.loc[row_count+exp] = to_fill
            row_count += total_exps
        
        # Check if the process must be restarted
        if os.path.exists(bkp_path) and restart:
            try:
                logger.info("Restarting")
                # Restart the process where there is no query_gen_date
                restarted = pd.read_csv(bkp_path)
                null_indexes = restarted[restarted["query_gen_date"].isna()].index.to_list()
                for index in null_indexes:
                    # Get the NL query from the given DataFrame
                    req_id = restarted.loc[index, "query_id"]
                    temp_df = df[df["req_id"] == req_id]
                    nl_req = temp_df["request"].item()
                    logger.info(f"Query ID: {req_id}, Run ID: {restarted.loc[index, 'query_run']}")
                    
                    # Updating the external and domain knowledge of the prompts for this query
                    self.prompts["Decomposition"]["simple"]["external_knowledge"] = temp_df["external_knowledge"].item()
                    self.prompts["Decomposition"]["simple"]["domain_knowledge"] = temp_df["domain_knowledge"].item()
                    self.prompts["Direct"]["request_prompt"]["external_knowledge"] = temp_df["external_knowledge"].item()
                    self.prompts["Direct"]["request_prompt"]["domain_knowledge"] = temp_df["domain_knowledge"].item()
                    
                    # Run the pipeline and time it
                    pred_start = time.time()
                    logger.info("Running pipeline")
                    sql_pred, tables, label = self.run_pipeline(nl_req, 
                                                                use_rag,
                                                                use_direct_prompts)
                    pred_time = time.time() - pred_start
                    
                    # Fill in the resulting SQL query and the time it took to generate
                    restarted.loc[index, "sql_query"] = sql_pred
                    restarted.loc[index, "tab_schema"] = tables
                    restarted.loc[index, "label"] = label
                    restarted.loc[index, "query_gen_time"] = pred_time
                    restarted.loc[index, "query_gen_date"] = datetime.now().isoformat(timespec='seconds')
                    
                    # Saving the DataFrame as a CSV file backup
                    logger.info("Saving backup")
                    restarted.to_csv(bkp_path)
                    
                # Now save it appropiately
                logger.info("Saving all")
                restarted.to_csv(file_path)
                
            except Exception as e:
                logger.error(f"An error has occurred while restarting the process: {e}")
                logger.info("Progress saved in new_df attribute of this object")
                
        else:
            try:
                logger.info("Running process")
                # Filling up the rest of the DataFrame
                row_count = 0
                for _, row in df.iterrows():
                    # Updating the external and domain knowledge of the prompts for this query                  
                    self.prompts["Decomposition"]["simple"]["external_knowledge"] = row["external_knowledge"].item()
                    self.prompts["Decomposition"]["simple"]["domain_knowledge"] = row["domain_knowledge"].item()
                    self.prompts["Direct"]["request_prompt"]["external_knowledge"] = row["external_knowledge"].item()
                    self.prompts["Direct"]["request_prompt"]["domain_knowledge"] = row["domain_knowledge"].item()
                    
                    for exp in range(total_exps):                                    
                        # Run the pipeline and time it
                        pred_start = time.time()
                        logger.info("Running pipeline")
                        sql_pred, tables, label = self.run_pipeline(row['request'], 
                                                            use_rag, 
                                                            use_direct_prompts)
                        pred_time = time.time() - pred_start
                        
                        # Fill in the resulting SQL query and the time it took to generate
                        self.new_df.loc[row_count+exp, "sql_query"] = sql_pred
                        self.new_df.loc[row_count+exp, "tab_schema"] = tables
                        self.new_df.loc[row_count+exp, "label"] = label
                        self.new_df.loc[row_count+exp, "query_gen_time"] = pred_time
                        self.new_df.loc[row_count+exp, "query_gen_date"] = datetime.now().isoformat(timespec='seconds')
                        
                        # Saving the DataFrame as a CSV file backup
                        logger.info("Saving backup")
                        self.new_df.to_csv(bkp_path)
                    row_count += total_exps   
                    
                # Saving the DataFrame as a CSV file
                logger.info("Saving all")
                self.new_df.to_csv(file_path)
                
            except Exception as e:
                logger.error(f"An error has occurred: {e}")
                logger.info("Progress saved in new_df attribute of this object")