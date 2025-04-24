import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np

from pipeline.process import *
from pipeline.eval import *
from pipeline.queryPipeline import queryPipeline
from prompts.base.prompts import *

# TODO: Recreate the dynamic prompts for queryPipeline for the Self Correction prompts

class metricsPipeline():
    def __init__(self, llm, lang_type, max_tokens, df, t_conn, 
                 n_tries, direct, self_corr, self_corr_prompts):
        """
        Metrics pipeline class
        """
        # metricsPipeline specific attributes
        self.llm = llm
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.df = df
        self.t_conn = t_conn
        self.n_tries = n_tries
        self.direct = direct
        self.self_corr = self_corr
        self.self_corr_prompts = self_corr_prompts
        
        # Fill-in attributes
        self.new_df = None
        
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
            query (str): NL query
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
    
    def run_metrics(self, sql_preds_path: str, total_exps: int = 10, 
                    restart: bool = False):
        """Function to run the experiments

        Args:
            sql_preds_path (str): Path to the CSV with the predicted SQL 
            queries
            total_exps (int, optional): Number of experiments. Defaults to 10
            restart (bool): Indicates if the experiment proccess must be 
            restarted. Defaults to False
            
        Returns:
            None
        """        
        # The metrics must be calculated by row for the new DataFrame/CSV
        # The metrics are:
        # 1. ER and EP for rows and columns in each experiment (r and p)
        # 2. The total number of perfect queries for rows and columns (N_rows (r = p = 1) and N_cols (r = 1))
        # Later, we can obtain the EP and ER for rows and columns with these values, as well as the number of perfect queries we had
        
        # Name of the file to save the predicted queries
        file_path = f"experiments/metrics_{self.llm}_{datetime.now().isoformat(timespec='seconds')}.csv".replace(":", "-")
        
        # Reading the CSV with the predicted SQL queries
        sql_preds = pd.read_csv(sql_preds_path)
        
        # CHECK: If we assume the tag and LLM used dont change for the metrics and the preds, then we could edit
        # only the sql_preds DataFrame. In the mean time, assume that it is not.
        
        # Columns to use
        column_names = ['code_tag', 'llm_used', 'query_id', 'query_run', 
                        'sql_query', 'tab_schema', 'label', 'query_gen_time', 
                        'query_gen_date', 'query_results', 'query_error', 
                        'sql_time', 'sql_date', 'r_row', 'p_row', 'r_col', 
                        'p_col', 'N_perfect_row', 'N_perfect_col']
        
        # Generate an empty DataFrame with the corresponding columns and rows
        num_rows = len(self.df) * total_exps
        self.new_df = pd.DataFrame(
            [[None]*len(column_names) for _ in range(num_rows)], 
            columns=column_names
        )
        
        # Reading the tag
        with open("tag.txt", "r") as f:
            tag = f.read().split("v")[1]
            f.close()
            
        # TODO: Optimize this code
        # Filling up the first columns
        row_count = 0
        for _, row in self.df.iterrows():
            to_fill_0 = [tag, None, row['req_id'], 0, 
                         None, None, None, None, 
                         None, None, None, None,
                         None, None, None, None,
                         None, None, None]
            self.new_df.loc[row_count] = to_fill_0
            for exp in range(total_exps):
                to_fill = [tag, self.llm, row['req_id'], exp+1, 
                           None, None, None, None, 
                           None, None, None, None,
                           None, None, None, None,
                           None, None, None]
                self.new_df.loc[row_count+exp+1] = to_fill
            row_count += total_exps+1
        
        # Check if the process must be restarted
        if self.new_df is not None and restart:
            try:
                # Restart the process where there is no sql_date
                null_indexes = self.new_df[self.new_df["sql_date"].isna()].index.to_list()
                for index in null_indexes:
                    # Get the predicted SQL query
                    restart_row = self.new_df.loc[index]
                    sql_pred = sql_preds[sql_preds["query_id"] == restart_row["query_id"]]
                    
                    # Check if it is a gold query or a pred query
                    if sql_pred["query_run"] == 0:                        
                        # Get output of the expected SQL query
                        gold_query_test = str(sql_pred['sql_query'])
                        gold_start = time.time()  # start time gold_query
                        query_gold, error_gold = self.run_sql_alerce(gold_query_test)
                        
                        # Check if the gold query was executed correctly, if not try again
                        if error_gold is not None:
                            query_gold, error_gold = self.run_sql_alerce(gold_query_test)
                            if error_gold is not None:
                                # If the second run fails, then save it as such
                                print("Failed gold query")
                                gold_time =  time.time() - gold_start
                                gold_date = datetime.now().isoformat(timespec='seconds')
                                self.new_df.loc[row_count] = [tag, None, row["req_id"], 0, 
                                                              row['gold_query'], None, None, None, 
                                                              None, query_gold, error_gold, 
                                                              gold_time, gold_date, 0, 0, 0, 
                                                              0, 0, 0]
                                continue
                
                        gold_time = time.time() - gold_start
                        gold_date = datetime.now().isoformat(timespec='seconds')
                        
                        # Drop duplicated columns
                        query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
                        
                        # Writing the gold values in the CSV
                        self.new_df.loc[row_count] = [tag, None, row['req_id'], 0,
                                                      row['gold_query'], None, None, None, 
                                                      None, query_gold, error_gold, 
                                                      gold_time, gold_date, 1, 1, 1, 
                                                      1, 1, 1]
                        
                        # Obtain the gold values for metric calculation
                        oids_gold = query_gold.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                        n_rows_gold = len(oids_gold)
                        n_cols_gold = query_gold.shape[1]
                        
                    else:
                        # We need to run the gold query
                        gold_row = self.new_df[(self.new_df["query_id"] == restart_row["query_id"]) & (self.new_df["query_run"] == 0)]
                        
                        # Get output of the expected SQL query
                        gold_query_test = str(gold_row['sql_query'])
                        
                        # Check if the gold query was executed correctly, if not try again
                        if error_gold is not None:
                            query_gold, error_gold = self.run_sql_alerce(gold_query_test)
                            if error_gold is not None:
                                # If the second run fails, then save it as such
                                print("Failed gold query")
                                n_rows_gold = 0
                                n_cols_gold = 0
                        
                        # Drop duplicated columns
                        query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
                        
                        # Obtain the gold values for metric calculation
                        oids_gold = query_gold.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                        n_rows_gold = len(oids_gold)
                        n_cols_gold = query_gold.shape[1]
                    
                        # Run the SQL query and time it
                        pred_start = time.time()
                        query_pred, error_pred = self.run_sql_alerce(sql_pred["sql_query"])
                        pred_time = time.time() - pred_start
                        
                        # Fill in the resulting SQL query and the time it took to generate
                        # If self-correction is enabled, use the respective prompts to correct
                        if self.self_corr:
                            # Check if there was an error. If there was, correct it
                            pred_start = time.time()
                            query_pred, error_pred = self.run_sql_alerce(sql_pred["sql_query"])
                            
                            # Correct it in the appropiate format      
                            if self.lang_type == "sql":
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1, 
                                    tab_schema=sql_pred["tab_schema"], 
                                    req=row["request"], 
                                    sql_pred=sql_pred["sql_query"], 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV
                                
                                # Run the corrected query
                                query_pred, error_pred = self.run_sql_alerce(new)
                                
                            # TODO: Review this border case
                            
                            elif self.lang_type == "python" and sql_pred["label"] == "simple":
                                # Border case
                                self.lang_type = "sql"
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1, 
                                    tab_schema=sql_pred["tab_schema"], 
                                    req=row["request"], 
                                    sql_pred=sql_pred["sql_query"], 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV
                                
                                # Run the corrected query  
                                query_pred, error_pred = self.run_sql_alerce(new)
                                        
                            else:
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1_python, 
                                    tab_schema=sql_pred["tab_schema"], 
                                    req=row["request"], 
                                    sql_pred=sql_pred["sql_query"], 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(format, new)
                                
                                # TODO: Add correction prompts to CSV

                                # Run the corrected query
                                query_pred, error_pred = self.run_sql_alerce(new)

                        # W/o self-correction
                        else:
                            pred_start = time.time()
                            query_pred, error_pred = self.run_sql_alerce(sql_pred["sql_query"])
                            
                        pred_time = time.time() - pred_start
                        pred_date = datetime.now().isoformat(timespec='seconds')
                        
                        # Predicted query is valid
                        if query_pred is not None and error_pred is None:
                            # Drop duplicated columns, it is assumed that the column name is exactly 'oid'
                            query_pred = query_pred.loc[:, ~query_pred.columns.duplicated()]
                            n_rows_pred = query_pred.shape[0]
                            n_cols_pred = query_pred.shape[1]
                            
                            ## Metrics for columns
                            
                            # Compare the columns of the predicted and expected SQL queries
                            cols_pred = query_pred.columns.values.tolist()
                            cols_gold = query_gold.columns.values.tolist()
                            true_pred_column = 0
                            false_pred_column = 0
                            true_gold_column = 0
                            false_gold_column = 0
                            # Get the number of columns that match between the predicted and expected SQL queries
                            for col in cols_pred:
                                if col in cols_gold:
                                    true_pred_column += 1
                                else:
                                    false_pred_column += 1
                            for col in cols_gold:
                                if col in cols_pred:
                                    true_gold_column += 1
                                else:
                                    false_gold_column += 1
                                    
                            # Calculating r and p
                            r_col = 0 if n_cols_gold == 0 else true_pred_column / n_cols_gold
                            p_col = 0 if n_cols_pred == 0 else true_gold_column / n_cols_pred
                            
                            # Calculating N_perfect
                            N_perfect_col = 1 if r_col == 1 else 0
                                    
                            ## Metrics for rows
                            
                            # Compare the oids of the predicted and expected SQL queries
                            true_pred_oid = 0
                            false_pred_oid = 0
                            true_gold_oid = 0
                            false_gold_oid = 0
                            try:
                                # Get predicted oid list
                                if 'oid' in query_pred.columns:
                                    oids_pred = query_pred.sort_values(by="oid",axis=0).reset_index(drop=True)['oid'].values.tolist()
                                elif 'oid_catalog' in query_pred.columns:
                                    oids_pred = query_pred.sort_values(by="oid_catalog",axis=0).reset_index(drop=True)['oid_catalog'].values.tolist()
                                elif 'objectidps1' in query_pred.columns:
                                    oids_pred = query_pred.sort_values(by="objectidps1",axis=0).reset_index(drop=True)['objectidps1'].values.tolist()
                                elif 'classifier_name' in query_pred.columns:
                                    oids_pred = query_pred.sort_values(by="classifier_name",axis=0).reset_index(drop=True)['classifier_name'].values.tolist()
                                elif 'count' in query_pred.columns:
                                    oids_pred = query_pred.sort_values(by="count",axis=0).reset_index(drop=True)['count'].values.tolist()
                                
                                # check oids for tipical columns names hallucinations
                                elif 'ztf_identifier' in [col.lower() for col in query_pred.columns.tolist()]:
                                    # change the column name to 'oid'
                                    query_pred.rename(columns={'ztf_identifier': 'oid'}, inplace=True)
                                    oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                                elif 'ztf identifier' in [col.lower() for col in query_pred.columns.tolist()]:
                                    # change the column name to 'oid'
                                    query_pred.rename(columns={'ztf identifier': 'oid'}, inplace=True)
                                    oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                                elif 'ztf_oid' in [col.lower() for col in query_pred.columns.tolist()]:
                                    # change the column name to 'oid'
                                    query_pred.rename(columns={'ztf_oid': 'oid'}, inplace=True)
                                    oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                                elif 'object' in [col.lower() for col in query_pred.columns.tolist()]:
                                    # change the column name to 'oid'
                                    query_pred.rename(columns={'object': 'oid'}, inplace=True)
                                    oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                                elif 'ztf' in [col.lower() for col in query_pred.columns.tolist()]:
                                    # change the column name to 'oid'
                                    query_pred.rename(columns={'ztf': 'oid'}, inplace=True)
                                    oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                                    
                                # Check if the predicted oids are equal to the expected oids list in the same order
                                are_equal = (oids_gold == oids_pred)
                            
                                # If the list of oids are equal, then the number of true and false oids is the same
                                if are_equal:
                                    true_pred_oid = len(oids_pred)
                                    true_gold_oid = len(oids_gold)
                                    false_pred_oid = 0
                                    false_gold_oid = 0
                                # If the list of oids are not equal, then the number of true and false oids is calculated
                                # based on the number of oids that match between the predicted and expected oids
                                else:
                                    # Check the number of pred oids that match the set of gold oids
                                    s = set(oids_gold)
                                    for oids in oids_pred:
                                        if oids in s: true_pred_oid += 1
                                        else: false_pred_oid += 1
                                    # Check the number of gold oids that match the set of pred oids
                                    s = set(oids_pred)
                                    for oids in oids_gold:
                                        if oids in s: true_gold_oid += 1
                                        else: false_gold_oid += 1
                            # If the column name is not exactly 'oid' then the oids are not compared
                            # The number of true and false oids is calculated based on the number of rows in the predicted and expected SQL queries
                            except Exception:
                                if query_pred.shape[0] == query_gold.shape[0]:
                                    true_pred_oid = query_pred.shape[0]
                                    true_gold_oid = query_gold.shape[0]
                                    false_pred_oid = 0
                                    false_gold_oid = 0

                                else:
                                    true_pred_oid = query_pred.shape[0]
                                    true_gold_oid = query_gold.shape[0]
                                    false_pred_oid = query_pred.shape[0]
                                    false_gold_oid = query_gold.shape[0]
                                    
                            # Calculating r and p
                            r_row = 0 if n_rows_gold == 0 else true_pred_oid / n_rows_gold
                            p_row = 0 if n_rows_pred == 0 else true_gold_oid / n_rows_pred
                                
                            # Calculating N_perfect
                            N_perfect_row = 1 if r_row == 1 and p_row == 1 else 0

                        # Predicted query is not valid due to an error in the query execution
                        else:
                            # metrics
                            r_row = 0
                            p_row = 0
                            r_col = 0
                            p_col = 0
                            N_perfect_row = 0
                            N_perfect_col = 0
                            
                        # Writing the pred values in the CSV
                        self.new_df.loc[index] = [tag, self.llm,
                                                  *sql_pred.iloc[0].to_list()[2:], 
                                                  query_pred, error_pred,
                                                  pred_time, pred_date, r_row,
                                                  p_row, r_col, p_col,
                                                  N_perfect_row, N_perfect_col]
                    
                # Now save it appropiately
                self.new_df.to_csv(file_path)
                
            except Exception as e:
                print(f"An error has occurred while restarting the process: {e}")
                print("Progress saved in new_df attribute of this object")
        
        # Filling up the rest of the DataFrame
        try:
            row_count = 0
            for _, row in self.df.iterrows():
                # Working only with the predicted queries for this request
                sql_preds_use = sql_preds[sql_preds["query_id"] == row["req_id"]]
                
                # Get output of the expected SQL query
                gold_query_test = str(row['gold_query'])
                gold_start = time.time()  # start time gold_query
                query_gold, error_gold = self.run_sql_alerce(gold_query_test)
                
                # Check if the gold query was executed correctly, if not try again
                if error_gold is not None:
                    query_gold, error_gold = self.run_sql_alerce(gold_query_test)
                    if error_gold is not None:
                        # If the second run fails, then save it as such
                        print("Failed gold query")
                        gold_time =  time.time() - gold_start
                        gold_date = datetime.now().isoformat(timespec='seconds')
                        self.new_df.loc[row_count] = np.array([tag, None, row["req_id"], 0,
                                                      row['gold_query'], None, None, None,
                                                      None, query_gold, error_gold,
                                                      gold_time, gold_date, 0, 0, 0,
                                                      0, 0, 0]).reshape([1, 19])
                        continue
        
                gold_time = time.time() - gold_start
                gold_date = datetime.now().isoformat(timespec='seconds')
                
                # Drop duplicated columns
                query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
                
                # Writing the gold values in the CSV
                self.new_df.loc[row_count] = np.array([tag, None, row['req_id'], 0,
                                              row['gold_query'], None, None, None,
                                              None, query_gold, error_gold,
                                              gold_time, gold_date, 1, 1, 1,
                                              1, 1, 1]).reshape([1, 19])
                
                # Obtain the gold values for metric calculation
                oids_gold = query_gold.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                n_rows_gold = len(oids_gold)
                n_cols_gold = query_gold.shape[1]
                
                # Number of times a query is predicted (number of experiments)
                for exp in range(total_exps):
                    # Predicted query info for this run
                    sql_pred = sql_preds_use[sql_preds_use["query_run"] == exp+1].reset_index()
                    
                    # Get output of the predicted SQL query
                    # If self-correction is enabled, use the respective prompts to correct
                    if self.self_corr:
                        # Check if there was an error. If there was, correct it
                        pred_start = time.time()
                        query_pred, error_pred = self.run_sql_alerce(sql_pred["sql_query"])
                        
                        # Correct it in the appropiate format      
                        if self.lang_type == "sql":
                            # Correcting the generated SQL
                            corr_prompt = prompt_self_correction_v2(
                                gen_task=general_context_selfcorr_v1, 
                                tab_schema=sql_pred["tab_schema"], 
                                req=row["request"], 
                                sql_pred=sql_pred["sql_query"], 
                                error=str(error_pred))
                            new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                            new = format_response(self.lang_type, new)
                            
                            # TODO: Add correction prompts to CSV
                            
                            # Run the corrected query
                            query_pred, error_pred = self.run_sql_alerce(new)
                            
                        # TODO: Review this border case
                        
                        elif self.lang_type == "python" and sql_pred["label"] == "simple":
                            # Border case
                            self.lang_type = "sql"
                            # Correcting the generated SQL
                            corr_prompt = prompt_self_correction_v2(
                                gen_task=general_context_selfcorr_v1, 
                                tab_schema=sql_pred["tab_schema"], 
                                req=row["request"], 
                                sql_pred=sql_pred["sql_query"], 
                                error=str(error_pred))
                            new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                            new = format_response(self.lang_type, new)
                            
                            # TODO: Add correction prompts to CSV
                            
                            # Run the corrected query  
                            query_pred, error_pred = self.run_sql_alerce(new)
                                    
                        else:
                            # Correcting the generated SQL
                            corr_prompt = prompt_self_correction_v2(
                                gen_task=general_context_selfcorr_v1_python, 
                                tab_schema=sql_pred["tab_schema"], 
                                req=row["request"], 
                                sql_pred=sql_pred["sql_query"], 
                                error=str(error_pred))
                            new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                            new = format_response(format, new)
                            
                            # TODO: Add correction prompts to CSV

                            # Run the corrected query
                            query_pred, error_pred = self.run_sql_alerce(new)

                    # W/o self-correction
                    else:
                        pred_start = time.time()
                        query_pred, error_pred = self.run_sql_alerce(sql_pred["sql_query"])
                        
                    pred_time = time.time() - pred_start
                    pred_date = datetime.now().isoformat(timespec='seconds')
                    
                    # Predicted query is valid
                    if query_pred is not None and error_pred is None:
                        # Drop duplicated columns, it is assumed that the column name is exactly 'oid'
                        query_pred = query_pred.loc[:, ~query_pred.columns.duplicated()]
                        n_rows_pred = query_pred.shape[0]
                        n_cols_pred = query_pred.shape[1]
                        
                        ## Metrics for columns
                        
                        # Compare the columns of the predicted and expected SQL queries
                        cols_pred = query_pred.columns.values.tolist()
                        cols_gold = query_gold.columns.values.tolist()
                        true_pred_column = 0
                        false_pred_column = 0
                        true_gold_column = 0
                        false_gold_column = 0
                        # Get the number of columns that match between the predicted and expected SQL queries
                        for col in cols_pred:
                            if col in cols_gold:
                                true_pred_column += 1
                            else:
                                false_pred_column += 1
                        for col in cols_gold:
                            if col in cols_pred:
                                true_gold_column += 1
                            else:
                                false_gold_column += 1
                                
                        # Calculating r and p
                        r_col = 0 if n_cols_gold == 0 else true_pred_column / n_cols_gold
                        p_col = 0 if n_cols_pred == 0 else true_gold_column / n_cols_pred
                        
                        # Calculating N_perfect
                        N_perfect_col = 1 if r_col == 1 else 0
                                
                        ## Metrics for rows
                        
                        # Compare the oids of the predicted and expected SQL queries
                        true_pred_oid = 0
                        false_pred_oid = 0
                        true_gold_oid = 0
                        false_gold_oid = 0
                        try:
                            # Get predicted oid list
                            if 'oid' in query_pred.columns:
                                oids_pred = query_pred.sort_values(by="oid",axis=0).reset_index(drop=True)['oid'].values.tolist()
                            elif 'oid_catalog' in query_pred.columns:
                                oids_pred = query_pred.sort_values(by="oid_catalog",axis=0).reset_index(drop=True)['oid_catalog'].values.tolist()
                            elif 'objectidps1' in query_pred.columns:
                                oids_pred = query_pred.sort_values(by="objectidps1",axis=0).reset_index(drop=True)['objectidps1'].values.tolist()
                            elif 'classifier_name' in query_pred.columns:
                                oids_pred = query_pred.sort_values(by="classifier_name",axis=0).reset_index(drop=True)['classifier_name'].values.tolist()
                            elif 'count' in query_pred.columns:
                                oids_pred = query_pred.sort_values(by="count",axis=0).reset_index(drop=True)['count'].values.tolist()
                            
                            # check oids for tipical columns names hallucinations
                            elif 'ztf_identifier' in [col.lower() for col in query_pred.columns.tolist()]:
                                # change the column name to 'oid'
                                query_pred.rename(columns={'ztf_identifier': 'oid'}, inplace=True)
                                oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                            elif 'ztf identifier' in [col.lower() for col in query_pred.columns.tolist()]:
                                # change the column name to 'oid'
                                query_pred.rename(columns={'ztf identifier': 'oid'}, inplace=True)
                                oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                            elif 'ztf_oid' in [col.lower() for col in query_pred.columns.tolist()]:
                                # change the column name to 'oid'
                                query_pred.rename(columns={'ztf_oid': 'oid'}, inplace=True)
                                oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                            elif 'object' in [col.lower() for col in query_pred.columns.tolist()]:
                                # change the column name to 'oid'
                                query_pred.rename(columns={'object': 'oid'}, inplace=True)
                                oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                            elif 'ztf' in [col.lower() for col in query_pred.columns.tolist()]:
                                # change the column name to 'oid'
                                query_pred.rename(columns={'ztf': 'oid'}, inplace=True)
                                oids_pred = query_pred.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
                                
                            # Check if the predicted oids are equal to the expected oids list in the same order
                            are_equal = (oids_gold == oids_pred)
                        
                            # If the list of oids are equal, then the number of true and false oids is the same
                            if are_equal:
                                true_pred_oid = len(oids_pred)
                                true_gold_oid = len(oids_gold)
                                false_pred_oid = 0
                                false_gold_oid = 0
                            # If the list of oids are not equal, then the number of true and false oids is calculated
                            # based on the number of oids that match between the predicted and expected oids
                            else:
                                # Check the number of pred oids that match the set of gold oids
                                s = set(oids_gold)
                                for oids in oids_pred:
                                    if oids in s: true_pred_oid += 1
                                    else: false_pred_oid += 1
                                # Check the number of gold oids that match the set of pred oids
                                s = set(oids_pred)
                                for oids in oids_gold:
                                    if oids in s: true_gold_oid += 1
                                    else: false_gold_oid += 1
                        # If the column name is not exactly 'oid' then the oids are not compared
                        # The number of true and false oids is calculated based on the number of rows in the predicted and expected SQL queries
                        except Exception:
                            if query_pred.shape[0] == query_gold.shape[0]:
                                true_pred_oid = query_pred.shape[0]
                                true_gold_oid = query_gold.shape[0]
                                false_pred_oid = 0
                                false_gold_oid = 0

                            else:
                                true_pred_oid = query_pred.shape[0]
                                true_gold_oid = query_gold.shape[0]
                                false_pred_oid = query_pred.shape[0]
                                false_gold_oid = query_gold.shape[0]
                                
                        # Calculating r and p
                        r_row = 0 if n_rows_gold == 0 else true_pred_oid / n_rows_gold
                        p_row = 0 if n_rows_pred == 0 else true_gold_oid / n_rows_pred
                            
                        # Calculating N_perfect
                        N_perfect_row = 1 if r_row == 1 and p_row == 1 else 0

                    # Predicted query is not valid due to an error in the query execution
                    else:
                        # metrics
                        r_row = 0
                        p_row = 0
                        r_col = 0
                        p_col = 0
                        N_perfect_row = 0
                        N_perfect_col = 0
                        
                    # Writing the pred values in the CSV
                    self.new_df.loc[row_count+exp+1] = np.array([tag, self.llm,
                                                        *sql_pred.iloc[0].to_list()[4:], 
                                                        query_pred, error_pred, 
                                                        pred_time, pred_date, 
                                                        r_row, p_row, r_col, p_col, 
                                                        N_perfect_row, N_perfect_col]).reshape([1, 19])
                        
                    print(f"\n\n Evaluation {exp+1} finished. Closing connection \n\n", flush=True)
            
                # Adding up the row count
                row_count += total_exps+1
            
            # Saving the DataFrame as a CSV file
            self.new_df.to_csv(file_path)
        
        except Exception as e:
            print(f"An error has occurred: {e}")
            print("Progress saved in new_df attribute of this object")
                    