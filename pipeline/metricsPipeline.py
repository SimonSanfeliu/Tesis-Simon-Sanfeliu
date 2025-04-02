import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.process import *
from pipeline.eval import *
from pipeline.queryPipeline import queryPipeline
from prompts.base.prompts import *


class metricsPipeline():
    """
    Metrics pipeline class
    """
    def __init__(self, model, lang_type, max_tokens, prompts, df, t_conn, size,
                 overlap, quantity, direct, rag_pipe, self_corr):
        """
        Class generator
        """
        # queryPipeline attributes
        self.model = model
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.prompts = prompts
        
        # metricsPipeline specific attributes
        self.df = df
        self.t_conn = t_conn
        self.size = size
        self.overlap = overlap
        self.quantity = quantity
        self.direct = direct
        self.rag_pipe = rag_pipe
        self.self_corr = self_corr
        
    def create_conn(self) -> sa.engine.base.Engine:
        """Function to create a connection with ALeRCE's SQL database

        Args:

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
              engine: sqlalchemy.engine.base.Engine) -> pd.DataFrame:
        """Function to run the SQL query in the database

        Args:
            specified_format (str): The type of formatting to use. It can be 
            'sql' for a singular query string or 'python' for the 
            decomposition in variables
            formatted_response (str): The response ready to be used in the database
            engine (sqlalchemy.engine.base.Engine): The engine to access the 
            database
            
        Returns:
            results (pandas.DataFrame): Pandas DataFrame with the results of the 
            query
        """
        results = None
        error = ""
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

    def run_sql_alerce(self, sql: str, n_tries: int = 3) -> tuple[pd.DataFrame, str]:
        """Execute the SQL query at the ALeRCE database and return the result
        
        Args:
            sql (str): SQL query to execute
            format (str): The type of formatting to use. It can be 'sql' for a singular 
            query string or 'python' for the decomposition in variables
            n_tries (int, optional): Number of tries to execute the query. Defaults to 3
            
        Returns:
            query (pandas.DataFrame): The result of the query
            error (str): Error message if the query could not be executed
        """
        # Create the instance
        engine = self.create_conn()
        query = None
        with engine.connect() as conn:
            # Try the query a number of times
            for n_ in range(0, n_tries):
                error = None
                with engine.begin() as conn:
                    try:
                        query, e = self.run_query(self.lang_type, sql, conn)
                        error = e
                        break
                    except Exception as e:
                        error = e
                        continue
        engine.dispose()
        return query, error
    
    def run_pipeline(self, query: str, n_tries: int = 3) -> tuple[pd.DataFrame, str, dict, dict, str]:
        """Function to run the entire pipeline. This pipeline could be the 
        original one or the new one. Here the self-correction is applied.

        Args:
            query (str): Natural language query for the database
            n_tries (int): Number of times to try excuting the query. Defaults to 3
            
        Returns:
            result (pandas.DataFrame): Dataframe with the resulting table
            error (str or None): Error message of the query (None if there wasn't one)
            total_usage (dict): API usage after the whole process
            prompts (dict): Dictonary with the prompts used in every step of the 
            pipeline
            table (str): Generated query
        """
        # Check if the new pipeline is being used
        if self.rag_pipe:
            table, total_usage, prompts, tables, label = pipeline(query, 
                                                                  self.model, 
                                                                  self.max_tokens, self.size, 
                                                overlap, quantity, format, 
                                                direct)
            # If self-correction is enabled, use the respective prompts to correct
            if self_corr:
                result, error = run_sql_alerce(table, format, min, n_tries)
                # Check if there was an error. If there was, correct it
                if error is not None:
                    print(f"Raised exception: {error}", flush=True)
                    print("Start retry with self-correction", flush=True)
                    
                    # Correct it in the appropiate format
                    if format == "sql":
                        corr_prompt = prompt_self_correction_v2(
                            gen_task=general_context_selfcorr_v1, 
                            tab_schema=tables, 
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
                    
                    elif format == "python" and label == "simple":
                        format = "sql"
                        corr_prompt = prompt_self_correction_v2(
                            gen_task=general_context_selfcorr_v1, 
                            tab_schema=tables, 
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
                            tab_schema=tables, 
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
            pipe = queryPipeline(
                query,
                self.model, 
                self.lang_type, 
                self.max_tokens, 
                self.prompts
            )
            
            # Schema Linking
            pipe.schema_linking(query)

            # Classification
            pipe.classify(query)

            # Decomposition
            pipe.decomposition(query)
            
            # Generating the queries
            pipe.query_generation()
            
            # If self-correction is enabled, use the respective prompts to correct
            if self.self_corr:
                # Check if there was an error. If there was, correct it
                result, error = run_sql_alerce(pipe.final_prompt, self.lang_type, self.t_conn, n_tries)
                
                # Correct it in the appropiate format      
                if self.lang_type == "sql":
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1, 
                        tab_schema=tables, 
                        req=query, 
                        sql_pred=table, 
                        error=str(error))
                    new, new_usage = api_call(self.model, self.max_tokens, corr_prompt)
                    new = format_response(self.lang_type, new)
                    
                    total_usage["Self-correction"] = new_usage
                    total_usage = pricing(total_usage, self.model)
                    prompts["Self-correction"] = corr_prompt
                    
                    # Run the corrected query
                    result, error = run_sql_alerce(table, format, min, n_tries)
                    
                elif format == "python" and label == "simple":
                    format = "sql"
                    corr_prompt = prompt_self_correction_v2(
                        gen_task=general_context_selfcorr_v1, 
                        tab_schema=tables, 
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
                    tab_schema=tables, 
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
                
        return result, error, total_usage, prompts, table, label