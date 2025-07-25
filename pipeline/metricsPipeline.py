import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import tempfile
import shutil
import traceback

from pipeline.process import *
from pipeline.eval import *
from prompts.base.prompts import *

from logger_setup import setup_logger

logger = setup_logger(name="metrics", log_file="logs/metrics.txt")

# TODO: Recreate the dynamic prompts for queryPipeline for the Self Correction prompts

class metricsPipeline():
    def __init__(self, llm, lang_type, max_tokens, t_conn, 
                 n_tries, direct, self_corr, self_corr_prompts, prompts_path):
        """
        Metrics pipeline class
        """
        # metricsPipeline specific attributes
        self.llm = llm
        self.original_lang_type = lang_type
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.t_conn = t_conn
        self.n_tries = n_tries
        self.direct = direct
        self.self_corr = self_corr
        self.self_corr_prompts = self_corr_prompts
        self.prompts_path = prompts_path
        
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
            database 
            engine (sqlalchemy.engine.base.Engine): The engine to 
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
                logger.error(f"Running SQL exception in run_query: {e}")
        elif specified_format == "python":
            try:
                exec(formatted_response, globals())
                if 'full_query' in globals():
                    results = pd.read_sql_query(full_query, con=engine)
                else:
                    error = "No 'full_query' variable in Python code"
                    logger.error(error)
            except Exception as e:
                error = str(e)
                logger.error(f"Running SQL exception in run_query: {e}")
        else:
            error = "No valid format specified"
            logger.error(error)
        
        return results, error            

    def run_sql_alerce(self, sql: str, label: str, gold: bool) -> tuple[pd.DataFrame, str]:
        """Execute the SQL query at the ALeRCE database and return the result
        
        Args:
            sql (str): SQL query to execute
            label (str): Predicted difficulty label for the SQL query
            gold (bool): Flag to tell if the query is the gold one
            
        Returns:
            query (pandas.DataFrame): The result of the query 
            error (str): Error message if the query could not be executed. None 
            if there is no error
        """
        # Create the instance
        engine = self.create_conn()
        query = None
        error = None
        with engine.connect() as conn:
            # Try the query a number of times
            for n_ in range(0, self.n_tries):
                with engine.begin() as conn:
                    try:
                        # Catching borderline cases
                        if gold:
                            self.lang_type = "sql"
                        
                        elif self.lang_type == "python" and label == "simple":
                            self.lang_type = "sql"
                            
                        elif self.original_lang_type == "python" and self.lang_type == "sql" and label != "simple":
                            self.lang_type = "python"

                        query, error = self.run_query(self.lang_type, sql, 
                                                      conn)
                        break
                    except Exception as e:
                        error = e
                        traceback.print_exc()
                        continue
        engine.dispose()
        return query, error

    def safe_to_csv(self, df: pd.DataFrame, path: str):
        """Safely write a DataFrame to CSV using a temporary file."""
        dir_name = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
            df.to_csv(tmp_path, index=False)
        shutil.move(tmp_path, path)

    def run_metrics(self, sql_preds_path: str, df: pd.DataFrame, 
                    total_exps: int = 10, restart: bool = False):
        """Function to run the experiments

        Args:
            sql_preds_path (str): Path to the CSV with the predicted SQL 
            queries
            df (pandas.DataFrame): DataFrame with all the original info
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
        bkp_path = "experiments/bkp_metrics.csv"
        
        # Reading the CSV with the predicted SQL queries
        sql_preds = pd.read_csv(sql_preds_path)
        
        # Filtering the predictions with the DataFrame
        sql_preds = sql_preds[sql_preds["query_id"].isin(df["req_id"])]
        
        # Getting the number of unique queries
        n_unique = len(sql_preds["query_id"].unique())
        
        # Columns to use
        column_names = ['code_tag', 'llm_used', 'prompt_version', 'query_id', 'query_run', 
                        'sql_query', 'tab_schema', 'label', 'query_gen_time', 
                        'query_gen_date', 'query_results', 'query_error', 
                        'sql_time', 'sql_date', 'r_row', 'p_row', 'r_col', 
                        'p_col', 'N_perfect_row', 'N_perfect_col']
        
        # Generate an empty DataFrame with the corresponding columns and rows if it doesn't exist already
        if self.new_df is None and not os.path.exists(bkp_path):
            num_rows = len(sql_preds) + n_unique
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
            for query_id in sql_preds["query_id"].unique():
                # Base row
                to_fill_0 = [tag, None, None, query_id, 0, *[None]*15]
                self.new_df.iloc[row_count] = to_fill_0

                for exp in range(total_exps):
                    to_fill = [tag, self.llm, prompt_version, query_id, exp + 1, *[None]*15]
                    self.new_df.iloc[row_count + exp + 1] = to_fill

                row_count += total_exps + 1
        
        # Check if the process must be restarted
        if os.path.exists(bkp_path) and restart:
            try:
                logger.info("Restarting")
                # Restart the process where there is no sql_date
                restarted = pd.read_csv(bkp_path)
                null_indexes = restarted[restarted["sql_date"].isna()].index.tolist()
                for index in null_indexes:
                    # Get the predicted SQL query
                    restart_row = restarted.loc[index]
                    sql_pred = sql_preds[sql_preds["query_id"] == restart_row["query_id"]]
                    logger.info(f"Query ID: {restart_row['query_id']}, Run ID: {restart_row['query_run']}")
                    
                    # Check if it is a gold query or a pred query
                    if restart_row["query_run"] == 0:            
                        # Get output of the expected SQL query
                        gold_query_test = df[df["req_id"] == restart_row["query_id"]]["gold_query"].item()
                        gold_start = time.time()  # start time gold_query
                        query_gold, error_gold = self.run_sql_alerce(gold_query_test, df[df["req_id"] == restart_row["query_id"]]["difficulty"].item(), True)
                        
                        # Check if the gold query was executed correctly, if not try again
                        if error_gold is not None or query_gold is None:
                            query_gold, error_gold = self.run_sql_alerce(gold_query_test, df[df["req_id"] == restart_row["query_id"]]["difficulty"].item(), True)
                            if error_gold is not None or query_gold is None:
                                # If the second run fails, then save it as such
                                logger.error("Failed gold query")
                                gold_time =  time.time() - gold_start
                                gold_date = datetime.now().isoformat(timespec='seconds')
                                restarted.loc[index, "sql_query"] = gold_query_test
                                restarted.loc[index, "tab_schema"] = None
                                restarted.loc[index, "label"] = None
                                restarted.loc[index, "query_gen_time"] = None
                                restarted.loc[index, "query_gen_date"] = None
                                restarted.loc[index, "query_results"] = query_gold
                                restarted.loc[index, "query_error"] = error_gold
                                restarted.loc[index, "sql_time"] = gold_time
                                restarted.loc[index, "sql_date"] = gold_date
                                restarted.loc[index, "r_row"] = 0
                                restarted.loc[index, "p_row"] = 0
                                restarted.loc[index, "r_col"] = 0
                                restarted.loc[index, "p_col"] = 0
                                restarted.loc[index, "N_perfect_row"] = 0
                                restarted.loc[index, "N_perfect_col"] = 0
                                
                                # Saving the DataFrame as a CSV file backup
                                logger.info("Saving backup")
                                self.safe_to_csv(restarted, bkp_path)
                                continue
                
                        gold_time = time.time() - gold_start
                        gold_date = datetime.now().isoformat(timespec='seconds')
                        
                        # Drop duplicated columns
                        logger.info(f"Query ID: {restart_row['query_id']}, Run ID: {restart_row['query_run']}, Query gold: {query_gold}")
                        query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
                        
                        # Writing the gold values in the CSV
                        restarted.loc[index, "sql_query"] = gold_query_test
                        restarted.loc[index, "tab_schema"] = None
                        restarted.loc[index, "label"] = None
                        restarted.loc[index, "query_gen_time"] = None
                        restarted.loc[index, "query_gen_date"] = None
                        restarted.loc[index, "query_results"] = [query_gold]
                        restarted.loc[index, "query_error"] = error_gold
                        restarted.loc[index, "sql_time"] = gold_time
                        restarted.loc[index, "sql_date"] = gold_date
                        restarted.loc[index, "r_row"] = 1
                        restarted.loc[index, "p_row"] = 1
                        restarted.loc[index, "r_col"] = 1
                        restarted.loc[index, "p_col"] = 1
                        restarted.loc[index, "N_perfect_row"] = 1
                        restarted.loc[index, "N_perfect_col"] = 1
                        
                        # Saving the DataFrame as a CSV file backup
                        logger.info("Saving backup")
                        self.safe_to_csv(restarted, bkp_path)
                        
                        # Obtain the gold values for metric calculation
                        oids_names = ["oid", "oid_catalog", "count", "classifier_name"]
                        check = [name for name in query_gold.columns.tolist() if name in oids_names]
                        oids_gold = query_gold.sort_values(by=check[0],axis=0).reset_index(drop=True)[check[0]].values.tolist()
                        n_rows_gold = len(oids_gold)
                        n_cols_gold = query_gold.shape[1]
                        
                    else:
                        logger.info("Getting the gold values to compare")                    
                        # Get output of the expected SQL query
                        gold_query_test = df[df["req_id"] == restart_row["query_id"]]["gold_query"].item()
                        gold_start = time.time()
                        query_gold, error_gold = self.run_sql_alerce(gold_query_test, df[df["req_id"] == restart_row["query_id"]]["difficulty"].item(), True)
                        
                        # Check if the gold query was executed correctly, if not try again
                        if error_gold is not None:
                            query_gold, error_gold = self.run_sql_alerce(gold_query_test, df[df["req_id"] == restart_row["query_id"]]["difficulty"].item(), True)
                            if error_gold is not None:
                                # If the second run fails, then save it as such
                                logger.error("Failed gold query")
                                gold_time =  time.time() - gold_start
                                gold_date = datetime.now().isoformat(timespec='seconds')
                                restarted.loc[index, "tab_schema"] = None
                                restarted.loc[index, "label"] = None
                                restarted.loc[index, "query_gen_time"] = None
                                restarted.loc[index, "query_gen_date"] = None
                                restarted.loc[index, "sql_time"] = gold_time
                                restarted.loc[index, "sql_date"] = gold_date
                                restarted.loc[index, "r_row"] = 0
                                restarted.loc[index, "p_row"] = 0
                                restarted.loc[index, "r_col"] = 0
                                restarted.loc[index, "p_col"] = 0
                                restarted.loc[index, "N_perfect_row"] = 0
                                restarted.loc[index, "N_perfect_col"] = 0
                                
                                # Saving the DataFrame as a CSV file backup
                                logger.info("Saving backup")
                                self.safe_to_csv(restarted, bkp_path)
                                continue
                        
                        # Drop duplicated columns
                        logger.info(f"Query ID: {restart_row['query_id']}, Run ID: {restart_row['query_run']}, Query gold: {query_gold}")
                        query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
                        
                        # Obtain the gold values for metric calculation
                        oids_names = ["oid", "oid_catalog", "count", "classifier_name"]
                        check = [name for name in query_gold.columns.tolist() if name in oids_names]
                        oids_gold = query_gold.sort_values(by=check[0],axis=0).reset_index(drop=True)[check[0]].values.tolist()
                        n_rows_gold = len(oids_gold)
                        n_cols_gold = query_gold.shape[1]
                        
                        # For self correction
                        logger.info("Getting the request for self correction")
                        request = df[df["req_id"] == restart_row["query_id"]]["request"].item()
                        
                        # Fill in the resulting SQL query and the time it took to generate
                        # If self-correction is enabled, use the respective prompts to correct
                        if self.self_corr:
                            logger.info("Self-correction")
                            # Check if there was an error. If there was, correct it
                            pred_start = time.time()
                            query_pred, error_pred = self.run_sql_alerce(sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["sql_query"].item(),
                                                                         sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["label"].item(),
                                                                         False)
                            # Correct it in the appropiate format      
                            if self.lang_type == "sql":
                                logger.info("Using SQL")
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1, 
                                    tab_schema=sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["tab_schema"].item(), 
                                    req=request, 
                                    sql_pred=sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["sql_query"].item(), 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV
                                
                                # Run the corrected query
                                query_pred, error_pred = self.run_sql_alerce(new, sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["label"].item(), False)
                                
                            # TODO: Review this border case
                            
                            elif self.lang_type == "python" and sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["label"].item() == "simple":
                                logger.info("Using SQL because of Python and simple query")
                                # Border case
                                self.lang_type = "sql"
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1, 
                                    tab_schema=sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["tab_schema"].item(), 
                                    req=request, 
                                    sql_pred=sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["sql_query"].item(), 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV
                                
                                # Run the corrected query  
                                query_pred, error_pred = self.run_sql_alerce(new, sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["label"].item(), False)
                                        
                            else:
                                logger.info("Using Python")
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1_python, 
                                    tab_schema=sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["tab_schema"].item(), 
                                    req=request, 
                                    sql_pred=sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["sql_query"].item(), 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV

                                # Run the corrected query
                                query_pred, error_pred = self.run_sql_alerce(new, sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["label"].item(), False)

                        # W/o self-correction
                        else:
                            logger.info("No self-correction")
                            pred_start = time.time()
                            query_pred, error_pred = self.run_sql_alerce(sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["sql_query"].item(),
                                                                         sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["label"].item(),
                                                                         False)
                            
                        pred_time = time.time() - pred_start
                        pred_date = datetime.now().isoformat(timespec='seconds')
                        
                        # If the DataFrame is None
                        if query_pred is None:
                            # metrics
                            r_row = 0
                            p_row = 0
                            r_col = 0
                            p_col = 0
                            N_perfect_row = 0
                            N_perfect_col = 0
                        
                        # Border case for empty DataFrame (it has columns)
                        elif query_pred.empty:
                            # Drop duplicated columns, it is assumed that the column name is exactly 'oid'
                            query_pred = query_pred.loc[:, ~query_pred.columns.duplicated()]
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
                            
                            r_row = 0
                            p_row = 0
                            N_perfect_row = 0
                        
                        # Predicted query is valid
                        elif query_pred is not None and error_pred is None:
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
                        restarted.loc[index, "sql_query"] = sql_pred[sql_pred["query_run"] == restart_row["query_run"]]['sql_query'].item()
                        restarted.loc[index, "tab_schema"] = sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["tab_schema"].item()
                        restarted.loc[index, "label"] = sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["label"].item()
                        restarted.loc[index, "query_gen_time"] = sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["query_gen_time"].item()
                        restarted.loc[index, "query_gen_date"] = sql_pred[sql_pred["query_run"] == restart_row["query_run"]]["query_gen_date"].item()
                        restarted.loc[index, "query_results"] = [query_pred]
                        restarted.loc[index, "query_error"] = error_pred
                        restarted.loc[index, "sql_time"] = pred_time
                        restarted.loc[index, "sql_date"] = pred_date
                        restarted.loc[index, "r_row"] = r_row
                        restarted.loc[index, "p_row"] = p_row
                        restarted.loc[index, "r_col"] = r_col
                        restarted.loc[index, "p_col"] = p_col
                        restarted.loc[index, "N_perfect_row"] = N_perfect_row
                        restarted.loc[index, "N_perfect_col"] = N_perfect_col
                        
                    # Saving the DataFrame as a CSV file backup
                    logger.info("Saving backup")
                    self.safe_to_csv(restarted, bkp_path)
                           
                # Now save it appropiately
                logger.info("Process ended. Saving it all")
                self.safe_to_csv(restarted, file_path)
                
            except Exception as e:
                traceback.print_exc()
                logger.error(f"An error has occurred while restarting the process: {e}")
                logger.info("Progress saved in new_df attribute of this object")
        
        else:
            # Filling up the rest of the DataFrame
            try:
                row_count = 0
                for _, row in self.new_df.iterrows():
                    # Working only with the predicted queries for this request
                    sql_preds_use = sql_preds[sql_preds["query_id"] == row["query_id"]].reset_index()
                    req_id = sql_preds_use['query_id'][0]
                    logger.info(f"Query ID: {req_id}, Run ID: 0 (gold)")
                    
                    # Get output of the expected SQL query
                    gold_query_test = df[df["req_id"] == req_id]["gold_query"].item()
                    gold_start = time.time()  # start time gold_query
                    query_gold, error_gold = self.run_sql_alerce(gold_query_test, df[df["req_id"] == req_id]["difficulty"].item(), True)
                    
                    # Check if the gold query was executed correctly, if not try again
                    if error_gold is not None:
                        query_gold, error_gold = self.run_sql_alerce(gold_query_test, df[df["req_id"] == req_id]["difficulty"].item(), True)
                        if error_gold is not None:
                            # If the second run fails, then save it as such
                            logger.error("Failed gold query")
                            gold_time =  time.time() - gold_start
                            gold_date = datetime.now().isoformat(timespec='seconds')
                            self.new_df.loc[row_count, "sql_query"] = gold_query_test
                            self.new_df.loc[row_count, "tab_schema"] = None
                            self.new_df.loc[row_count, "label"] = None
                            self.new_df.loc[row_count, "query_gen_time"] = None
                            self.new_df.loc[row_count, "query_gen_date"] = None
                            self.new_df.loc[row_count, "query_results"] = [query_gold]
                            self.new_df.loc[row_count, "query_error"] = error_gold
                            self.new_df.loc[row_count, "sql_time"] = gold_time
                            self.new_df.loc[row_count, "sql_date"] = gold_date
                            self.new_df.loc[row_count, "r_row"] = 0
                            self.new_df.loc[row_count, "p_row"] = 0
                            self.new_df.loc[row_count, "r_col"] = 0
                            self.new_df.loc[row_count, "p_col"] = 0
                            self.new_df.loc[row_count, "N_perfect_row"] = 0
                            self.new_df.loc[row_count, "N_perfect_col"] = 0
                            
                            # Saving the DataFrame as a CSV file backup
                            logger.info("Saving backup")
                            self.safe_to_csv(self.new_df, bkp_path)
                            continue
            
                    gold_time = time.time() - gold_start
                    gold_date = datetime.now().isoformat(timespec='seconds')
                    
                    # Drop duplicated columns
                    query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
                    
                    # Writing the gold values in the CSV
                    self.new_df.loc[row_count, "sql_query"] = gold_query_test
                    self.new_df.loc[row_count, "tab_schema"] = None
                    self.new_df.loc[row_count, "label"] = None
                    self.new_df.loc[row_count, "query_gen_time"] = None
                    self.new_df.loc[row_count, "query_gen_date"] = None
                    self.new_df.loc[row_count, "query_results"] = [query_gold]
                    self.new_df.loc[row_count, "query_error"] = error_gold
                    self.new_df.loc[row_count, "sql_time"] = gold_time
                    self.new_df.loc[row_count, "sql_date"] = gold_date
                    self.new_df.loc[row_count, "r_row"] = 1
                    self.new_df.loc[row_count, "p_row"] = 1
                    self.new_df.loc[row_count, "r_col"] = 1
                    self.new_df.loc[row_count, "p_col"] = 1
                    self.new_df.loc[row_count, "N_perfect_row"] = 1
                    self.new_df.loc[row_count, "N_perfect_col"] = 1
                    
                    # Saving the DataFrame as a CSV file backup
                    logger.info("Saving backup")
                    self.safe_to_csv(self.new_df, bkp_path)
                    
                    # Obtain the gold values for metric calculation
                    oids_names = ["oid", "oid_catalog", "count", "classifier_name"]
                    check = [name for name in query_gold.columns.tolist() if name in oids_names]
                    oids_gold = query_gold.sort_values(by=check[0],axis=0).reset_index(drop=True)[check[0]].values.tolist()
                    n_rows_gold = len(oids_gold)
                    n_cols_gold = query_gold.shape[1]
                    
                    # For self correction
                    request = df[df["req_id"] == req_id]["request"][0]
                    
                    # Number of times a query is predicted (number of experiments)
                    for exp in range(total_exps):
                        # Predicted query info for this run
                        sql_pred = sql_preds_use[sql_preds_use["query_run"] == exp+1]
                        logger.info(f"Query ID: {sql_pred['query_id'].item()}, Run ID: {sql_pred['query_run'].item()}")
                        
                        # Get output of the predicted SQL query
                        # If self-correction is enabled, use the respective prompts to correct
                        if self.self_corr:
                            # Check if there was an error. If there was, correct it
                            pred_start = time.time()
                            query_pred, error_pred = self.run_sql_alerce(sql_pred["sql_query"].item(), sql_pred["label"].item(), False)
                            
                            # Correct it in the appropiate format      
                            if self.lang_type == "sql":
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1, 
                                    tab_schema=sql_pred["tab_schema"].item(), 
                                    req=request, 
                                    sql_pred=sql_pred["sql_query"].item(), 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV
                                
                                # Run the corrected query
                                query_pred, error_pred = self.run_sql_alerce(new, sql_pred["label"].item(), False)
                                
                            # TODO: Review this border case
                            
                            elif self.lang_type == "python" and sql_pred["label"].item() == "simple":
                                # Border case
                                self.lang_type = "sql"
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1, 
                                    tab_schema=sql_pred["tab_schema"].item(), 
                                    req=request, 
                                    sql_pred=sql_pred["sql_query"].item(), 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV
                                
                                # Run the corrected query  
                                query_pred, error_pred = self.run_sql_alerce(new, sql_pred["label"].item(), False)
                                        
                            else:
                                # Correcting the generated SQL
                                corr_prompt = prompt_self_correction_v2(
                                    gen_task=general_context_selfcorr_v1_python, 
                                    tab_schema=sql_pred["tab_schema"].item(), 
                                    req=request, 
                                    sql_pred=sql_pred["sql_query"].item(), 
                                    error=str(error_pred))
                                new, new_usage = api_call(self.llm, self.max_tokens, corr_prompt)
                                new = format_response(self.lang_type, new)
                                
                                # TODO: Add correction prompts to CSV

                                # Run the corrected query
                                query_pred, error_pred = self.run_sql_alerce(new, sql_pred["label"].item(), False)

                        # W/o self-correction
                        else:
                            pred_start = time.time()
                            query_pred, error_pred = self.run_sql_alerce(sql_pred["sql_query"].item(), sql_pred["label"].item(), False)
                            
                        pred_time = time.time() - pred_start
                        pred_date = datetime.now().isoformat(timespec='seconds')
                        
                        # If the DataFrame is None
                        if query_pred is None:
                            # metrics
                            r_row = 0
                            p_row = 0
                            r_col = 0
                            p_col = 0
                            N_perfect_row = 0
                            N_perfect_col = 0
                        
                        # Border case for empty DataFrame (it has columns)
                        elif query_pred.empty:
                            # Drop duplicated columns, it is assumed that the column name is exactly 'oid'
                            query_pred = query_pred.loc[:, ~query_pred.columns.duplicated()]
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
                            
                            r_row = 0
                            p_row = 0
                            N_perfect_row = 0
                        
                        # Predicted query is valid
                        elif query_pred is not None and error_pred is None:
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
                        self.new_df.loc[row_count+exp+1, "query_run"] = exp+1
                        self.new_df.loc[row_count+exp+1, "sql_query"] = sql_pred['sql_query'].item()
                        self.new_df.loc[row_count+exp+1, "tab_schema"] = sql_pred["tab_schema"].item()
                        self.new_df.loc[row_count+exp+1, "label"] = sql_pred["label"].item()
                        self.new_df.loc[row_count+exp+1, "query_gen_time"] = sql_pred["query_gen_time"].item()
                        self.new_df.loc[row_count+exp+1, "query_gen_date"] = sql_pred["query_gen_date"].item()
                        self.new_df.loc[row_count+exp+1, "query_results"] = [query_pred]
                        self.new_df.loc[row_count+exp+1, "query_error"] = error_pred
                        self.new_df.loc[row_count+exp+1, "sql_time"] = pred_time
                        self.new_df.loc[row_count+exp+1, "sql_date"] = pred_date
                        self.new_df.loc[row_count+exp+1, "r_row"] = r_row
                        self.new_df.loc[row_count+exp+1, "p_row"] = p_row
                        self.new_df.loc[row_count+exp+1, "r_col"] = r_col
                        self.new_df.loc[row_count+exp+1, "p_col"] = p_col
                        self.new_df.loc[row_count+exp+1, "N_perfect_row"] = N_perfect_row
                        self.new_df.loc[row_count+exp+1, "N_perfect_col"] = N_perfect_col
                            
                        logger.info(f"Evaluation {exp+1} finished. Closing connection")
                        
                        # Saving the DataFrame as a CSV file backup
                        logger.info("Saving backup")
                        self.safe_to_csv(self.new_df, bkp_path)
                
                    # Adding up the row count
                    row_count += total_exps+1
                
                # Saving the DataFrame as a CSV file
                logger.info("Process ended. Saving it all")
                self.safe_to_csv(self.new_df, file_path)
            
            except Exception as e:
                logger.error(f"An error has occurred: {e}")
                logger.info("Progress saved in new_df attribute of this object")
                    