import time, os, pytz
from datetime import datetime
import requests
import pandas as pd
import multiprocessing as mp
import pickle, json
import sqlalchemy as sa
from typing import Callable

from secret.config import SQL_URL, USER_10, PASS_10
from pipeline.process import run_query, api_call, format_response, pricing
from pipeline.main import pipeline, recreated_pipeline
from prompts.correction.SelfCorrection import prompt_self_correction_v2, \
    general_context_selfcorr_v1, general_context_selfcorr_v1_python

# Setup params for query engine
params = requests.get(SQL_URL).json()['params']


def create_conn(min: int=2) -> sa.engine.base.Engine:
    """Function to create a connection with ALeRCE's SQL database

    Args:
        min (int, optional): Number of minutes the connection to the database 
        is active. Defaults to 2

    Returns:
        sqlalchemy.engine.base.Engine or None: The engine to access the 
        database or prints an error message
    """
    # At the moment, there are only the options of 2 or 10 minutes
    if min==2:
        engine = sa.create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}", poolclass=sa.pool.NullPool)
        return engine
        
    elif min==10:
        engine = sa.create_engine(f"postgresql+psycopg2://{USER_10}:{PASS_10}@{params['host']}/{params['dbname']}", poolclass=sa.pool.NullPool)
        return engine
    
    else:
        print('Time not avalaible')
        

def run_sql_alerce(sql: str, format: str, min: int = 2, 
                   n_tries: int = 3) -> tuple[pd.DataFrame, str]:
  """Execute the SQL query at the ALeRCE database and return the result
  
  Args:
    sql (str): SQL query to execute
    format (str): The type of formatting to use. It can be 'sql' for a singular 
    query string or 'python' for the decomposition in variables
    min (int, optional): Timeout limit for the database connection. Defaults to 2
    n_tries (int, optional): Number of tries to execute the query. Defaults to 3
    
  Returns:
    query (pandas.DataFrame): The result of the query
    error (str): Error message if the query could not be executed
  """
  # Create the instance
  engine = create_conn(min=min)
  query = None
  with engine.connect() as conn:
    # Try the query a number of times
    for n_tries in range(0, n_tries):
      error = None
      with engine.begin() as conn:
        try:
            query, e = run_query(format, sql, conn)
            error = e
            break
        except Exception as e:
          error = e
          continue
  engine.dispose()
  return query, error


def run_pipeline(query: str, model: str, max_tokens: int, size: int, 
                 overlap: int, quantity: int, format: int, 
                 direct: bool = False, rag_pipe: bool = True, 
                 self_corr: bool = True, min: int = 2, 
                 n_tries: int = 3) -> tuple[pd.DataFrame, str, dict, dict]:
    """Function to run the entire pipeline. This pipeline could be the 
       original one or the new one. Here the self-correction is applied.

    Args:
        query (str): Natural language query for the database
        model (str): LLM to use
        max_tokens (int): Maximum amount of output tokens of the LLM
        size (int): Size of the chunks for the character text splitter used in
        the RAG process
        overlap (int): Size of the overlap of the chunks used in the RAG 
        process
        quantity (int): The amount of most similar chunks to consider in the
        RAG process
        format (str): The type of formatting to use. It can be 'singular' for
        a singular query string or 'var' for the decomposition in variables
        direct (bool): If True, use direct approach for query generation. If 
        False, use step-by-step approach
        rag_pipe (bool): Condition to use the new pipeline (uses RAG)
        self_corr (bool): Condition to use self-correction
        min (int): Time to make the query. Defaults to 2
        n_tries (int): Number of times to try excuting the query. Defaults to 3
        
    Returns:
        result (pandas.DataFrame): Dataframe with the resulting table
        error (str or None): Error message of the query (None if there wasn't one)
        total_usage (dict): API usage after the whole process
        prompts (dict): Dictonary with the prompts used in every step of the 
        pipeline
    """
    # Check if the new pipeline is being used
    if rag_pipe:
        table, total_usage, prompts = pipeline(query, model, max_tokens, size, 
                                               overlap, quantity, format, 
                                               direct)
        print(table, flush=True)
        # If self-correction is enabled, use the respective prompts to correct
        if self_corr:
          # TODO: Agregar caso borde de queries simples cuando el formato es python
          result, error = run_sql_alerce(table, format, min, n_tries)
          # Check if there was an error. If there was, correct it
          if error is not None:
            print(f"Raised exception: {error}", flush=True)
            print("Start retry with self-correction", flush=True)
                
            tab_schema = prompts["Classification"].split("\n The following \
              tables are needed to generate the query: ")[0]
            
            # Correct it in the appropiate format
            if format == "sql":
              corr_prompt = prompt_self_correction_v2(
                gen_task=general_context_selfcorr_v1, 
                tab_schema=tab_schema, 
                req=query, 
                sql_pred=table, 
                error=str(error))
              new, new_usage = api_call(model, max_tokens, corr_prompt)
              new = format_response(format, new)
              print("Corrected query:", flush=True)
              print(new, flush=True)
              total_usage["Self-correction"] = new_usage
              total_usage = pricing(total_usage, model)
              prompts["Self-correction"] = corr_prompt
              
              # Run the corrected query  
              result, error = run_sql_alerce(table, format, min, n_tries)
                    
            else:
              corr_prompt = prompt_self_correction_v2(
                gen_task=general_context_selfcorr_v1_python, 
                tab_schema=tab_schema, 
                req=query, 
                sql_pred=table, 
                error=str(error))
              new, new_usage = api_call(model, max_tokens, corr_prompt)
              print("Corrected query:", flush=True)
              print(new, flush=True)
              new = format_response(format, new)
              total_usage["Self-correction"] = new_usage
              total_usage = pricing(total_usage, model)
              prompts["Self-correction"] = corr_prompt
              
              # Run the corrected query
              result, error = run_sql_alerce(table, format, min, n_tries)

        # W/o self-correction
        else:
          result, error = run_sql_alerce(table, format, min, n_tries)

    # Using the recreated pipeline
    else:
        table, total_usage, prompts = recreated_pipeline(query, model, 
                                                         max_tokens, format, 
                                                         direct)
        print(table)
        # If self-correction is enabled, use the respective prompts to correct
        if self_corr:
          # Check if there was an error. If there was, correct it
          result, error = run_sql_alerce(table, format, min, n_tries)
          print(f"Raised exception: {error}", flush=True)
          print("Start retry with self-correction", flush=True)
                
          tab_schema = prompts["Classification"].split("\n The following \
            tables are needed to generate the query: ")[0]
          
          # Correct it in the appropiate format      
          if format == "sql":
            corr_prompt = prompt_self_correction_v2(
              gen_task=general_context_selfcorr_v1, 
              tab_schema=tab_schema, 
              req=query, 
              sql_pred=table, 
              error=str(error))
            new, new_usage = api_call(model, max_tokens, corr_prompt)
            new = format_response(format, new)
            print("Corrected query:", flush=True)
            print(new, flush=True)
            total_usage["Self-correction"] = new_usage
            total_usage = pricing(total_usage, model)
            prompts["Self-correction"] = corr_prompt
            
            # Run the corrected query
            result, error = run_sql_alerce(table, format, min, n_tries)
                    
          else:
            corr_prompt = prompt_self_correction_v2(
              gen_task=general_context_selfcorr_v1_python, 
              tab_schema=tab_schema, 
              req=query, 
              sql_pred=table, 
              error=str(error))
            new, new_usage = api_call(model, max_tokens, corr_prompt)
            print("Corrected query:", flush=True)
            print(new, flush=True)
            new = format_response(format, new)
            total_usage["Self-correction"] = new_usage
            total_usage = pricing(total_usage, model)
            prompts["Self-correction"] = corr_prompt

            # Run the corrected query
            result, error = run_sql_alerce(table, format, min, n_tries)

        # W/o self-correction
        else:
          result, error = run_sql_alerce(table, format, min, n_tries)
            
    return result, error, total_usage, prompts


def compare_oids(df_: pd.DataFrame, sql_pred_list: list[str], n_exp: int, 
                 format: str, min: int = 2) -> list[dict]:
  """Evaluate the generated SQL queries with the true/gold SQL queries
  
  Args:
    df_ (pandas.DataFrame): Dataframe with the true/gold SQL queries
    sql_pred_list (list[str]): List of SQL queries to evaluate
    n_exp (int): Number of the experiment, used to save the results
    format (str): The type of formatting to use. It can be 'sql' for a singular 
    query string or 'python' for the decomposition in variables
    min (int, optional): Timeout limit for the database connection. Defaults to 2
    
  Returns:
    results_list (list[dict]): List of dictionaries with the evaluation results
  """

  results_list = []
  indx = 0
  # iterate over the rows of the dataset
  for _, row in df_.iterrows():

    query_pred = None
    error_pred = None
    query_gold = None
    error_gold = None

    # Get output of the predicted SQL query
    pred_start = time.time()
    query_pred, error_pred = run_sql_alerce(sql_pred_list[indx], format, min=min, n_tries=3)    
    pred_end = time.time()
    pred_time = pred_end - pred_start

    # Get output of the expected SQL query
    gold_query_test = str(row['gold_query'])
    gold_start = time.time() # start time gold_query
    query_gold, error_gold = run_sql_alerce(gold_query_test, format, min=min, n_tries=3)
    
    # Check if the gold query was executed correctly, if not try again
    if error_gold is not None:
      query_gold, error_gold = run_sql_alerce(gold_query_test, format, min=min, n_tries=3)
      if error_gold is not None:
        print(f"Gold query {row['req_id']} could not be executed for experiment {n_exp}")
        results_list.append({ "req_id": row['req_id'], "n_exp": n_exp, "query_diff": row['difficulty'], "query_type": row['type'], 
                         "n_rows_gold": 0, "n_rows_pred": 0, "rows_oid_pred_true": 0, "rows_oid_pred_false": 0, "rows_oid_gold_true": 0, "rows_oid_gold_false": 0,
                         "n_cols_gold": 0, "n_cols_pred": 0, "cols_pred_true": 0, "cols_pred_false": 0, "cols_gold_true": 0, "cols_gold_false": 0,
                         "gold_time": 0, "pred_time": 0, 'error_pred': None, "are_equal": False, "is_oid": False,
                         "cols_pred": 0, "cols_gold": 0, "columns": 1})
        indx += 1    
        continue
    gold_end = time.time()
    gold_time =  gold_end - gold_start

    # Drop duplicated columns
    query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
    
    # Obtain the gold values
    oids_gold = query_gold.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
    n_rows_gold = len(oids_gold)
    n_cols_gold = query_gold.shape[1]
    is_oid = True
    are_equal = False
    
    # Predicted query is valid
    if query_pred is not None and error_pred is None:
      # Drop duplicated columns, it is assumed that the column name is exactly 'oid'
      query_pred = query_pred.loc[:, ~query_pred.columns.duplicated()]
      n_rows_pred = query_pred.shape[0]
      n_cols_pred = query_pred.shape[1]
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

    # Predicted query is not valid due to an error in the query execution
    else:
      # pred
      n_rows_pred = 0
      n_cols_pred = 0
      true_pred_oid = 0
      false_pred_oid = 0
      true_pred_column = 0
      false_pred_column = 0
      # gold
      true_gold_oid = 0
      false_gold_oid = 0
      true_gold_column = 0
      false_gold_column = 0
      cols_pred = 0
      cols_gold = 0
    
    indx+=1
    results_list.append({ "req_id": row['req_id'], "n_exp": n_exp, "query_diff": row['difficulty'], "query_type": row['type'], 
                         "n_rows_gold": n_rows_gold, "n_rows_pred": n_rows_pred, "rows_oid_pred_true": true_pred_oid, "rows_oid_pred_false": false_pred_oid, "rows_oid_gold_true": true_gold_oid, "rows_oid_gold_false": false_gold_oid,
                         "n_cols_gold": n_cols_gold, "n_cols_pred": n_cols_pred, "cols_pred_true": true_pred_column, "cols_pred_false": false_pred_column, "cols_gold_true": true_gold_column, "cols_gold_false": false_gold_column,
                         "gold_time": gold_time, "pred_time": pred_time, 'error_pred': error_pred, "are_equal": are_equal, "is_oid": is_oid, 
                         "cols_pred": cols_pred, "cols_gold": cols_gold, "columns": cols_pred})
    
  print(f"\n\n Evaluation {n_exp} Finished, closing connection \n\n", flush=True)
  
  return results_list


def new_compare_oids(df_: pd.DataFrame, n_exp: int, model: str, 
                     max_tokens: int, format: str, path: str, min: int = 2, 
                     n_tries: int = 3, self_corr: bool = False, 
                     rag_pipe: bool = False, direct: bool = False, 
                     size: int = 0, overlap: int = 0, 
                     quantity: int = 0) -> list[dict]:
  """Evaluate the generated SQL queries with the true/gold SQL queries
  
  Args:
    df_ (pandas.DataFrame): Dataframe with the true/gold SQL queries
    n_exp (int): Number of the experiment, used to save the results
    model (str): LLM model to use
    max_tokens (int): Maximum output tokens of the LLM.
    format (str): The format for SQL queries ('singular' or 'var')
    path (str): Path to save the usage and prompts
    min (int, optional): Timeout limit for the database connection. Defaults to 2
    n_tries (int, optional): Number of times to try excuting the query. Defaults to 3
    self_corr (bool, optional): Enable self-correction. Defaults to False
    rag_pipe (bool, optional): Use the RAG pipeline. Defaults to False
    direct (bool, optional): Use direct query generation. Defaults to False
    size (int, optional): Chunk size for RAG. Defaults to 0
    overlap (int, optional): Overlap size for RAG chunks. Defaults to 0
    quantity (int, optional): Number of similar chunks for RAG. Defaults to 0
    
  Returns:
    results_list (list[dict]): List of dictionaries with the evaluation results
  """
  # Path to save files
  if not os.path.exists(path):
    os.makedirs(path)
  
  # iterate over the rows of the dataset
  results_list = []
  indx = 0
  for _, row in df_.iterrows():

    query_pred = None
    error_pred = None
    query_gold = None
    error_gold = None

    # Get output of the predicted SQL query
    pred_start = time.time()
    query_pred, error_pred, usage, prompts = run_pipeline(row["request"], 
                                                          model, max_tokens, 
                                                          size, overlap, 
                                                          quantity, format, 
                                                          direct, rag_pipe, 
                                                          self_corr, min, 
                                                          n_tries)    
    pred_end = time.time()
    pred_time = pred_end - pred_start
    
    # Saving the usage and prompts
    current_time = datetime.now(pytz.timezone('Chile/Continental'))
    with open(f"{path}/usage_req_{row['req_id']}_{current_time.strftime('%H-%M-%S')}.pkl", "wb") as fp:
      pickle.dump(usage, fp)
    with open(f"{path}/prompts_req_{row['req_id']}_{current_time.strftime('%H-%M-%S')}.pkl", "w") as fp:
      json.dump(prompts, fp)

    # Get output of the expected SQL query
    gold_query_test = str(row['gold_query'])
    gold_start = time.time() # start time gold_query
    query_gold, error_gold = run_sql_alerce(gold_query_test, "sql", min=min, 
                                            n_tries=n_tries)
    
    # Check if the gold query was executed correctly, if not try again
    if error_gold is not None:
      query_gold, error_gold = run_sql_alerce(gold_query_test, "sql", min=min, 
                                              n_tries=n_tries)
      if error_gold is not None:
        print(f"Gold query {row['req_id']} could not be executed for experiment {n_exp}", flush=True)
        results_list.append({"req_id": row['req_id'], "n_exp": n_exp, 
                             "query_diff": row['difficulty'], 
                             "query_type": row['type'], "n_rows_gold": 0, 
                             "n_rows_pred": 0, "rows_oid_pred_true": 0, 
                             "rows_oid_pred_false": 0, "rows_oid_gold_true": 0,
                             "rows_oid_gold_false": 0, "n_cols_gold": 0, 
                             "n_cols_pred": 0, "cols_pred_true": 0, 
                             "cols_pred_false": 0, "cols_gold_true": 0, 
                             "cols_gold_false": 0, "gold_time": 0, 
                             "pred_time": 0, 'error_pred': None, 
                             "are_equal": False, "is_oid": False, 
                             "cols_pred": 0, "cols_gold": 0, "columns": 1})
        indx += 1    
        continue
    gold_end = time.time()
    gold_time =  gold_end - gold_start

    # Drop duplicated columns
    query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]
    
    # Obtain the gold values
    oids_gold = query_gold.sort_values(by='oid',axis=0).reset_index(drop=True)['oid'].values.tolist()
    n_rows_gold = len(oids_gold)
    n_cols_gold = query_gold.shape[1]
    is_oid = True
    are_equal = False
    
    # Predicted query is valid
    if query_pred is not None and error_pred is None:
      # Drop duplicated columns, it is assumed that the column name is exactly 'oid'
      query_pred = query_pred.loc[:, ~query_pred.columns.duplicated()]
      n_rows_pred = query_pred.shape[0]
      n_cols_pred = query_pred.shape[1]
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

    # Predicted query is not valid due to an error in the query execution
    else:
      # pred
      n_rows_pred = 0
      n_cols_pred = 0
      true_pred_oid = 0
      false_pred_oid = 0
      true_pred_column = 0
      false_pred_column = 0
      # gold
      true_gold_oid = 0
      false_gold_oid = 0
      true_gold_column = 0
      false_gold_column = 0
      cols_pred = 0
      cols_gold = 0
    
    indx+=1
    results_list.append({"req_id": row['req_id'], "n_exp": n_exp, 
                         "query_diff": row['difficulty'], 
                         "query_type": row['type'], "n_rows_gold": n_rows_gold, 
                         "n_rows_pred": n_rows_pred, 
                         "rows_oid_pred_true": true_pred_oid, 
                         "rows_oid_pred_false": false_pred_oid, 
                         "rows_oid_gold_true": true_gold_oid, 
                         "rows_oid_gold_false": false_gold_oid,
                         "n_cols_gold": n_cols_gold, 
                         "n_cols_pred": n_cols_pred, 
                         "cols_pred_true": true_pred_column, 
                         "cols_pred_false": false_pred_column, 
                         "cols_gold_true": true_gold_column, 
                         "cols_gold_false": false_gold_column,
                         "gold_time": gold_time, "pred_time": pred_time, 
                         "error_pred": error_pred, "are_equal": are_equal, 
                         "is_oid": is_oid, "cols_pred": cols_pred, 
                         "cols_gold": cols_gold, "columns": cols_pred})
    
  print(f"\n\n Evaluation {n_exp} Finished, closing connection \n\n", flush=True)
  
  return results_list


def safe_new_compare_oids(*args):
    try:
        # Call the actual function
        return new_compare_oids(*args)
    except Exception as e:
        # Serialize error details for safe return
        arg_dict = {*args}
        return {"error": str(e), "args": arg_dict}


def error_handler(e: str) -> None:
  """Error handler for the parallel execution
  
  Args:
    e (str): Error message
    
  Returns:
    None
  """
  print('error', flush=True)
  print(dir(e), "\n", flush=True)
  print("-->{}<--".format(e.__cause__), flush=True)
  

def run_sqls_parallel(sqls_list: list[list], db_: pd.DataFrame, 
                      result_funct: Callable, format: str, num_cpus: int = 1, 
                      min: int = 2) -> None:
  """Run the SQL queries in parallel
  
  Args:
    sqls_list (list[list]): List of lists with the SQL queries to evaluate
    db_ (pandas.Dataframe): Dataframe with the true/gold SQL queries
    result_funct (func): Function to save the results during the parallel execution
    format (str): The type of formatting to use. It can be 'sql' for a singular 
    query string or 'python' for the decomposition in variables
    num_cpus (int, optional): Number of CPUs to use for multiprocessing. Defaults to 1
    min (int, optional): Timeout limit for the database connection. Defaults to 2
    
  Returns:
    None
  """
  pool = mp.Pool(processes=num_cpus)
  for i,sql_pred in enumerate(sqls_list):
      print(f"Running evaluation n°{i}", flush=True)
      pool.apply_async(compare_oids, args=(db_, sql_pred, i, format, min), callback=result_funct, error_callback=error_handler)
  pool.close()
  pool.join()
  
  
def new_run_sqls_parallel(df_: pd.DataFrame, model: str, max_tokens: int, 
                          format: str, path: str, min: int = 2, 
                          n_tries: int = 3, self_corr: bool = False, 
                          rag_pipe: bool = False, direct: bool = False, 
                          size: int = 0, overlap: int = 0, quantity: int = 0, 
                          result_funct: Callable = None, num_cpus: int = 1, 
                          exps: int = 5) -> None:
  """Run the SQL queries in parallel
  
  Args:
    df_ (pandas.Dataframe): Dataframe with the true/gold SQL queries
    model (str): LLM model to use
    max_tokens (int): Maximum output tokens of the LLM
    format (str): The format for SQL queries ('singular' or 'var')
    path (str): Path to save the usage and prompts
    min (int, optional): Timeout limit for the database connection. Defaults to 2
    n_tries (int, optional): Number of times to try excuting the query. Defaults to 3
    self_corr (bool, optional): Enable self-correction. Defaults to False
    rag_pipe (bool, optional): Use the RAG pipeline. Defaults to False
    direct (bool, optional): Use direct query generation. Defaults to False
    size (int, optional): Chunk size for RAG. Defaults to 0
    overlap (int, optional): Overlap size for RAG chunks. Defaults to 0
    quantity (int, optional): Number of similar chunks for RAG. Defaults to 0
    result_funct (func, optional): Function to save the results during the parallel execution
    num_cpus (int, optional): Number of CPUs to use for multiprocessing. Defaults to 1
    exps (int, optional): Number of experiments to do. Defaults to 5
    
  Returns:
    None
  """
  pool = mp.Pool(processes=num_cpus)
  sqls_list = [df_ for e in range(exps)]
  for i,sql_pred in enumerate(sqls_list):
      print(f"Running evaluation n°{i}", flush=True)
      pool.apply_async(safe_new_compare_oids, 
                       args=(sql_pred, i, model, max_tokens, format, path, min, 
                             n_tries, self_corr, rag_pipe, direct, size, 
                             overlap, quantity), 
                       callback=result_funct, 
                       error_callback=error_handler)
  pool.close()
  pool.join()


def run_eval_fcn(db_eval, experiment_path, save_path,
                n_exps=10, num_cpus=10, db_min=2, selfcorr=False):
    ''' Evaluate the generated SQL queries with the true/gold SQL queries
    Args:
      db_eval: pandas dataframe with the true/gold SQL queries
      experiment_path: path to the predictions in pickle format
      save_path: path to save the evaluations in pickle format
      n_exps: number of experiments to evaluate
      num_cpus: number of CPUs to use for multiprocessing
      db_min: Timeout limit for the database connection
    Returns:
      exec_result: list of dictionaries with the evaluation results
    '''

    # load the predictions
    sql_pred_list_temp = []
    for i in range(n_exps):
        with open(experiment_path+f"_{i}", "rb") as fp:   # Unpickling
            if selfcorr:
              # TODO: fix try except to know if the pickle file is in the new format or the old one
              # check keys in the pickle file to know if it is the new format or the old one
              
              try: 
                sql_pred_list_temp.append([eval["original_pred_query"] if eval["selfcorr_query"] is None else eval["selfcorr_query"] for eval in pickle.load(fp)])
              except:
                sql_pred_list_temp.append([eval["pred_query"] if eval["selfcorr_query"] is None else eval["selfcorr_query"] for eval in pickle.load(fp)])
                
            else: sql_pred_list_temp.append([eval["pred_query"] for eval in pickle.load(fp)])
        
    print(db_eval.req_id.values)
    # define function to save the results during the parallel execution
    exec_result = []
    def result_callback(result):
        exec_result.append(result)

    # Run Evaluation
    run_sqls_parallel(sql_pred_list_temp_gptformat, db_eval, result_callback, num_cpus=num_cpus, min=db_min)
    print(f"Finished evaluation, proceding to save the results for experiment {experiment_path}", flush=True)
    
    with open(os.path.join(os.getcwd(), save_path)+'.pkl', "wb") as fp: pickle.dump(exec_result, fp) # save the evaluations in pickle format

    # check if the number of experiments is the same as the number of evaluations and the number of predictions
    if (len(exec_result) == n_exps) and (len(exec_result) == len(sql_pred_list_temp)): print("All evaluations finished correctly")
    elif len(exec_result) == 1: print("Evaluation finished correctly")
    else: raise ValueError(f'Error in the number of experiments: {len(exec_result)} and {len(sql_pred_list_temp)}')
    
    exec_result_json = []
    for n_exp, eval_i in enumerate(exec_result):
      # change the error message to string and save the results in a json file
      print([q['req_id'] for q in eval_i])
      for q_i in eval_i:
        if q_i["error_pred"] is not None:
          q_i["error_pred"] = str(q_i["error_pred"])

      # save the evaluation of each experiment in a json
      if db_min==2:
        with open(os.path.join(os.getcwd(), save_path)+f'_{n_exp}.json', 'w') as fp:
          json.dump(eval_i, fp)
      else:
        with open(os.path.join(os.getcwd(), save_path)+f'_{n_exp}_{db_min}.json', 'w') as fp:
          json.dump(eval_i, fp)

      exec_result_json.append({'n_exp': n_exp, 'eval': eval_i} )
      
    # save all evaluations in a json file
    if db_min==2:
      with open(os.path.join(os.getcwd(), save_path)+'.json', 'w') as fp:
        json.dump(exec_result_json, fp)
    else:
      with open(os.path.join(os.getcwd(), save_path)+f'_{db_min}.json', 'w') as fp:
        json.dump(exec_result_json, fp)

    return exec_result


def new_run_eval_fcn(db_eval: pd.DataFrame, experiment_path: str, model: str, 
                     max_tokens: int, format: str, num_cpus: int = 1, 
                     db_min: int = 2, n_tries: int = 3, 
                     self_corr: bool = False, rag_pipe: bool = False, 
                     direct: bool = False, size: int = 0, overlap: int = 0, 
                     quantity: int = 0, exps: int = 5) -> list[dict]:
    """
    Evaluate the generated SQL queries with the true/gold SQL queries.

    Args:
        db_eval (pd.DataFrame): Dataframe with the true/gold SQL queries
        experiment_path (str): Path to save the predictions and evaluations in 
        pickle and JSON format
        model (str): LLM model to use.
        max_tokens (int): Maximum output tokens of the LLM.
        format (str): The format for SQL queries ('singular' or 'var').
        num_cpus (int): Number of CPUs for parallel processing. Defaults to 1.
        db_min (int): Timeout limit for the database connection. Defaults to 2.
        n_tries (int, optional): Number of times to try excuting the query. Defaults to 3.
        self_corr (bool, optional): Enable self-correction. Defaults to False.
        rag_pipe (bool, optional): Use the RAG pipeline. Defaults to False.
        direct (bool, optional): Use direct query generation. Defaults to False.
        size (int, optional): Chunk size for RAG. Defaults to 0.
        overlap (int, optional): Overlap size for RAG chunks. Defaults to 0.
        quantity (int, optional): Number of similar chunks for RAG. Defaults to 0.
        exps (int, optional): Number of experiments to do. Defaults to 5

    Returns:
        exec_result (list[dict]): List of dictionaries with evaluation results.
    """
    # Path to save files
    current_time = datetime.now(pytz.timezone('Chile/Continental'))
    experiment_path += f"/{current_time.strftime('%Y-%m-%d')}"
    if not os.path.exists(experiment_path):
      os.makedirs(experiment_path)
    
    # Callback to collect results
    exec_result = []
    def result_callback(result):
        exec_result.extend(result)

    # Run evaluations in parallel
    print("Starting parallel evaluation using run_sqls_parallel...")
    new_run_sqls_parallel(db_eval, model, max_tokens, format, experiment_path, 
                          db_min, n_tries, self_corr, rag_pipe, direct, size, 
                          overlap, quantity, result_callback, num_cpus, exps)
      
    with open(f"{experiment_path}/results_{current_time.strftime('%H-%M-%S')}.pkl", "wb") as fp:
        pickle.dump(exec_result, fp)
    with open(f"{experiment_path}/results_{current_time.strftime('%H-%M-%S')}.json", "w") as fp:
        json.dump(exec_result, fp)

    print("Evaluation completed and results saved.")
    return exec_result
