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
    def __init__(self, model, lang_type, max_tokens, prompts, df):
        """
        Class generator
        """
        self.model = model
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.prompts = prompts
        self.df = df
        
    def create_conn(self, min: int=2) -> sa.engine.base.Engine:
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

    def run_sql_alerce(self, sql: str, min: int = 2, 
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
        engine = self.create_conn(min=min)
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
    
    def run_pipeline(self, query: str, model: str, max_tokens: int, size: int, 
                 overlap: int, quantity: int, format: str, 
                 direct: bool = False, rag_pipe: bool = True, 
                 self_corr: bool = True, min: int = 2, 
                 n_tries: int = 3) -> tuple[pd.DataFrame, str, dict, dict, str]:
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
            format (str): The type of formatting to use. It can be 'sql' for a 
            singular query string or 'python' for the decomposition in Python 
            variables
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
            table (str): Generated query
        """
        # Check if the new pipeline is being used
        if rag_pipe:
            table, total_usage, prompts, tables, label = pipeline(query, model, max_tokens, size, 
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
            pipe = queryPipeline(self.model, self.lang_type, self.max_tokens, self.prompts)
            
            # Schema Linking
            schema_usage = pipe.schema_linking(query)

            # Classification
            class_usage = pipe.classify(query)

            # Decomposition
            decomp_prompt, decomp_usage = pipe.decomposition(query)
            
            # Generating the queries
            decomp_gen_query, decomp_gen_usage = pipe.query_generation(decomp_prompt)
            
            # If self-correction is enabled, use the respective prompts to correct
            if self_corr:
                # Check if there was an error. If there was, correct it
                result, error = run_sql_alerce(decomp_gen_query, self.lang_type, min, n_tries)
                
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
                
        return result, error, total_usage, prompts, table, label