def evaluate(df, preds, engine):
    """Function to evaluate the pipeline

    Args:
        df (pandas.DataFrame): Dataframe with all the data for the queries
        preds (dict): Dictionary with the predicted tables from the pipeline
        engine (SQL engine): SQL database engine  TODO: Check datatype
    """
    # TODO: Fill out like Jorge's function and whta he explained