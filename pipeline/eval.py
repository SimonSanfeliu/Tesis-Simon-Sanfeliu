import os, time, psutil
import requests, psycopg2
import pandas as pd, numpy as np
import multiprocessing as mp
import pickle, json
import sqlalchemy as sa

from secret.config import SQL_URL, USER_10, PASS_10
from pipeline.process import run_query

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
        

def run_sql_alerce(sql: str, format: str, min: int = 2, n_tries: int = 3) -> tuple[pd.DataFrame, str]:
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
            query = run_query(format, sql, conn)
            break
        except Exception as e:
          error = e
          continue
  engine.dispose()
  return query, error


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
                      result_funct: function, format: str, num_cpus: int = 1, 
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


def run_eval_fcn(db_eval: pd.DataFrame, experiment_path: str, save_path: str, 
                 format: str, n_exps: int = 10, num_cpus: int = 10, 
                 db_min: int = 2, selfcorr: bool = False) -> list[dict]:
    """Evaluate the generated SQL queries with the true/gold SQL queries
    
    Args:
      db_eval (pandas.DataFrame): Dataframe with the true/gold SQL queries
      experiment_path (str): Path to the predictions in pickle format
      save_path (str): Path to save the evaluations in pickle format
      format (str): The type of formatting to use. It can be 'sql' for a singular 
      query string or 'python' for the decomposition in variables
      n_exps (int, optional): Number of experiments to evaluate. Defaults to 10
      num_cpus (int, optional): Number of CPUs to use for multiprocessing. Defaults to 10
      db_min (int, optional): Timeout limit for the database connection. Defaults to 2
      
    Returns:
      exec_result (list[dict]): List of dictionaries with the evaluation results
    """

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
    
    # Define function to save the results during the parallel execution
    exec_result = []
    def result_callback(result):
        exec_result.append(result)

    # Run Evaluation
    run_sqls_parallel(sql_pred_list_temp, db_eval, result_callback, format, num_cpus=num_cpus, min=db_min)
    print(f"Finished evaluation, proceding to save the results for experiment {experiment_path}", flush=True)
    
    with open(os.path.join(os.getcwd(), save_path)+'.pkl', "wb") as fp: pickle.dump(exec_result, fp) # save the evaluations in pickle format

    # Check if the number of experiments is the same as the number of evaluations and the number of predictions
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
