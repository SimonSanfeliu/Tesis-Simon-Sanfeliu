### Self-Correction Task Prompts
# This file contains the prompts for the self-correction task.
# Self-correction tasks are designed to help the model learn to correct SQL queries based on the error returned when executing the query in the ALeRCE database.
# The main prompt include the task description, context, final instructions, and the SQL query that needs to be corrected with the expected error.
# The structure of the prompt can be modified to include more information or to change the order of the sections using the prompt functions.
###

# Setting up astronomical context
with open("final_prompts/astrocontext.txt", "r") as f:
    astro_context = f.read()

# General Self-Correction Prompt
## Self-correcting task prompt
general_task_selfcorr_v1='''
As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
## General Context of the database schema
general_context_selfcorr_v1=f'''
As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.

## ALeRCE Pipeline Details
- Stamp Classifier (denoted as 'stamp_classifier'): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as 'lc_classifier'): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [periodic, stochastic, transient], denoted as 'lc_classifier_top.'
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as 'lc_classifier_periodic,' 'lc_classifier_transient,' and 'lc_classifier_stochastic,' respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('Ceph'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Spatial Object Types by Classifier
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('transient', 'stochastic', 'periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the "SELECT *" SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' or 'object' if it is necessary (priority in this order).

{astro_context}

# Generate a query for each step, resolving and analysing it, with the following format:
```step_number [STEP QUERY HERE] ```
# Finally, join all the steps in a final query, with the following format: 
```sql [FINAL QUERY HERE] ```
DON'T include anything else inside and after your FINAL answer.
'''

general_context_selfcorr_v1_python=f'''
As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.

## ALeRCE Pipeline Details
- Stamp Classifier (denoted as 'stamp_classifier'): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as 'lc_classifier'): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [periodic, stochastic, transient], denoted as 'lc_classifier_top.'
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as 'lc_classifier_periodic,' 'lc_classifier_transient,' and 'lc_classifier_stochastic,' respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('Ceph'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Spatial Object Types by Classifier
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('transient', 'stochastic', 'periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the "SELECT *" SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' or 'object' if it is necessary (priority in this order).

{astro_context}

# Generate a query for each step, resolving and analysing it, with the following format:
```python sub_queries = [VARIABLE SUB-QUERY HERE] ```
# Finally, join all the steps in a final query like so: 
```python full_query = [FINAL QUERY HERE] ```
DON'T include anything else inside and after your FINAL answer.
'''

# Final Instructions, emphasizing the importance to correct the query and the format to answer
## version 1
final_instr_selfcorr_v1 = '''# Using valid PostgreSQL, the CORRECT the query given the error, using the correct database schema or nested queries to optimize.
# Answer ONLY with the SQL query
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# SQL:
'''
# General Self-Correction Prompt Structure
self_corr_prompt='''{}
# Correct a SQL query given the next user request:
{}

# This are the tables Schema. Assume that only the next tables are required for the query:
{}

# SQL Query
{}

# Error returned when executing the query in the ALeRCE database
{}

# Using valid PostgreSQL, the CORRECT names of the tables only with their respective columns, and the information given in "Context", Correct the query given the error, using the correct database schema or nested queries to optimize.
# Answer ONLY with the SQL query
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# SQL:
'''

# Separate prompts for Timeout and Schema Error
## Timeout Self-Correction Prompt
### version 1
self_corr_timemout_prompt='''{}

# Correct a SQL query given the next user request:
{}

# This are the tables Schema. Assume that only the next tables are required for the query:
{}

# The next query is not working due to a timeout Error, correct the query using the correct database schema or nested queries to optimize.
# SQL Query
{}
# Error returned when executing the query in the ALeRCE database
{}

# Follow the next advices to correct the query:
- Check if the query is using the correct tables and columns, and if the conditions are correct, given the user request.
- Check if the SQL code includes all the requested conditions.

# If there are no problems with the previous steps, follow the next advices to correct the query:
- Check if the SQL code includes the necessary conditions to optimize the query, and if the query is using the correct database schema or nested queries to optimize.
    - It is possible that the query is too complex and it is necessary to use nested queries to optimize the query.
    - If there is a JOIN or a sub-query between object and probability, check if the condition 'ranking=1' is set in the probability table, unless the request said otherwise.
- Check if are at least 3 conditions over the probability table, because if not, the query is too general. Add more conditions if necessary.
- If there are conditions involving dates or times, check if the dates are not too far away, or are in a reasonable range.

# Using valid PostgreSQL, CORRECT the query
# Answer ONLY with the SQL query
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# SQL:
'''

## Schema Error Self-Correction Prompt
### version 1
self_corr_schema_prompt='''{}

# Correct a SQL query given the next user request:
{}

# This are the tables Schema. Assume that only the next tables are required for the query:
{}

# The next query is not working due to a syntax error, correct the query using the correct database schema.
# SQL Query
{}
# Error returned when executing the query in the ALeRCE database
{}

# Follow the next advices to correct the query:
- Check if the query is using the correct database schema. This includes the correct names of the tables and the correct names of the columns. If not, correct the query.
- Check if the query have the correct syntax. If not, correct the query.
- If there is a "missing FROM-clause entry", check where the table or sub-query is used in the query and add the correct name of the table or sub-query.

# Using valid PostgreSQL, the CORRECT the query given the error, using the correct database schema or nested queries to optimize.
# Answer ONLY with the SQL query
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# SQL:
'''




# {simple_query_task_vf}
# General task description for the self-correction task
general_task_selfcorr_vf='''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) database. This database serves as a repository for information about astronomical variable objects. The information for every variable object originates from a sequence of one or more astronomical alerts, data packets streamed when an astronomical object shows a significant variation with respect to a reference image. The database information includes flux variations as a function of time (known as light curve), basic object properties such as the coordinates, and advanced features or statistics computed for each object.
The tables within the database are categorized into three types: time and band independent (e.g., object and probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection, forced-photometry). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
# Separate prompts for Timeout and Schema Error
## Timeout Self-Correction Prompt
### version f
self_correction_timeout_prompt_vf='''
{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# This are the tables Schema. Assume that only the next tables are required for the query:
{tab_schema}

# The next query is not working due to a timeout Error, correct the query using the correct database schema or nested queries to optimize.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow the next advices to correct the query:
- Check if the query is using the correct tables and columns, and if the conditions are correct, given the user request.
- Check if the SQL code includes all the requested conditions.

# If there are no problems with the previous steps, follow the next advices to correct the query:
- Check if the SQL code includes the necessary conditions to optimize the query, and if the query is using the correct database schema or nested queries to optimize.
    - It is possible that the query is too complex and it is necessary to use nested queries to optimize the query.
    - If there is a JOIN or a sub-query between object and probability, check if the condition 'ranking=1' is set in the probability table, unless the request said otherwise.
- Check if are at least 3 conditions over the probability table, because if not, the query is too general. Add more conditions if necessary.
- If there are conditions involving dates or times, check if the dates are not too far away, or are in a reasonable range.

# Using valid PostgreSQL, CORRECT the query
# Answer ONLY with the SQL query
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# SQL:
'''
## Schema Error Self-Correction Prompt
### version f
self_correction_schema_prompt_vf='''
{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# This are the tables Schema. Assume that only the next tables are required for the query:
{tab_schema}

# The next query is not working due to a syntax error, correct the query using the correct database schema.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow the next advices to correct the query:
- Check if the query is using the correct database schema. This includes the correct names of the tables and the correct names of the columns. If not, correct the query.
- Check if the query have the correct syntax. If not, correct the query.
- If there is a ""missing FROM-clause entry"", check where the table or sub-query is used in the query and add the correct name of the table or sub-query.

# Using valid PostgreSQL, the CORRECT the query given the error, using the correct database schema or nested queries to optimize.
# Answer ONLY with the SQL query
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# SQL:
'''


# Separate prompts for Timeout, No Exist and Schema Error
## version 2
## Timeout Self-Correction Prompt
self_correction_timeout_prompt_v2='''{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# This are the tables Schema. Assume that only the next tables are required for the query:
{tab_schema}

# The next query is not working due to a timeout Error, correct the query using the correct database schema or nested queries to optimize.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow the next advices to correct the query:
- Check if the SQL or Python code includes the necessary conditions to optimize the query, and if the query is using the correct database schema or nested queries to optimize.
    - It is possible that the query is too complex and it is necessary to use nested queries to optimize the query.
    - If there is a JOIN or a sub-query between some table and probability, check if the condition 'ranking=1' is set in the probability table, unless the request said otherwise.
- If there are conditions involving dates or times, check if the dates are not too far away, or are in a reasonable range.
# If the probability table is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- Check if are at least 3 conditions over the probability table, because if not, the query is too general. Add more conditions if necessary.

# Check the query and correct the query modifying the SQL or Python code where the error is found.
# Add COMMENTS so that the user can understand.
# Answer ONLY with the SQL query or the Python sub-queries
'''

## No Exist Self-Correction Prompt
self_correction_no_exist_prompt_v2='''{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# This are the tables Schema. Assume that only the next tables are required for the query:
{tab_schema}

# The next query is not working due to a timeout Error, correct the query using the correct database schema or nested queries to optimize.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# The query is not working because the table or column does not exist in the database schema. Check the table or column name and correct the query.
# Follow the next advices to correct the query:
- Check if the query is using the correct tables and columns provided, and if the conditions are correct, given the user request.
- Check if the SQL or Python code includes all the requested conditions.
- If the error is due to a table or column that does not exist, check the table or column name and correct the query using the correct database schema provided.
- If the error is due to a function that does not exist, try to modify the query using only the information given in the database schema.
- If the error is due to a relation that does not exist, check the relation name and correct the query using the correct database schema.
# All the information needed is in the database schema, use only the information provided in the database schema to correct the query. If it is not explicitly provided, go for the most common sense approach.

# Check the query and correct the query modifying the SQL or Python code where the error is found.
# Add COMMENTS so that the user can understand.
# Answer ONLY with the SQL query or the Python sub-queries
'''

## Schema Error Self-Correction Prompt
self_correction_schema_prompt_v2='''{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# This are the tables Schema. Assume that only the next tables are required for the query:
{tab_schema}

# The next query is not working due to a syntax error, correct the query using the correct database schema.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow the next advices to correct the query:
- Check if the query is using the correct database schema. This includes the correct names of the tables and the correct names of the columns. If not, correct the query.
- Check if the query have the correct syntax. If not, correct the query.
- If there is a "missing FROM-clause entry", check where the table or sub-query is used in the query and add the correct name of the table or sub-query.
- Use only the information provided in the database schema to correct the query. If it is not explicitly provided, go for the most common sense approach.

# Check the query and correct the query modifying the SQL or Python code where the error is found.
# Add COMMENTS so that the user can understand.
# Answer ONLY with the SQL query or the Python sub-queries
'''


## General Self-Correction Prompt Structure
# The functions are used to fill the variables of the prompt with the specific information of the request.
# It is possible to modify the functions to include more information or to change the order of the sections of the prompt.
##

# base version
def general_prompt_self_correction(sql_pred, sql_error, prompt):
  return prompt.format(sql_pred, sql_error)
# version 1 general error
def general_prompt_self_correctionV1(sql_pred, sql_error, prompt):
  return f'''{general_task}
  # DataBase Schema
  # DataBase Schema without description
  ## TABLE object ( oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change);
  ## TABLE probability ( oid, class_name, classifier_name, classifier_version, probability, ranking);
  ## TABLE feature ( oid, name, value, fid, version);
  ## TABLE magstat ( oid, fid, stellar, corrected, ndet, ndubious, dmdt_first, dm_first, sigmadm_first, dt_first, magmean, magmedian, magmax, magmin, magsigma, maglast, magfirst, magmean_corr, magmedian_corr, magmax_corr, magmin_corr, magsigma_corr, maglast_corr, magfirst_corr, firstmjd, lastmjd, step_id_corr, saturation_rate);
  ## TABLE non_detection ( oid, fid, mjd, diffmaglim);
  ## TABLE detection ( candid, oid, mjd, fid, pid, diffmaglim, isdiffpos, nid, ra, dec, magpsf, sigmapsf, magap, sigmagap, distnr, rb, rbversion, drb, drbversion, magapbig, sigmagapbig, rfid, magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext, corrected, dubious, parent_candid, has_stamp, step_id_corr);
  ## TABLE step ( step_id, name, version, comments, date);
  ## TABLE taxonomy (classifier_name,classifier_version,classes);
  ## TABLE feature_version ( version, step_id_feature, step_id_preprocess);
  ## TABLE xmatch ( oid, catid, oid_catalog, dist, class_catalog, period,);
  ## TABLE allwise ( oid_catalog, ra, dec, w1mpro, w2mpro, w3mpro, w4mpro, w1sigmpro, w2sigmpro, w3sigmpro, w4sigmpro, j_m_2mass, h_m_2mass, k_m_2mass, j_msig_2mass, h_msig_2mass, k_msig_2mass);
  ## TABLE dataquality ( candid, oid, fid, xpos, ypos, chipsf, sky, fwhm, classtar, mindtoedge, seeratio, aimage, bimage, aimagerat, bimagerat, nneg, nbad, sumrat, scorr, dsnrms, ssnrms, magzpsci, magzpsciunc, magzpscirms, nmatches, clrcoeff, clrcounc, zpclrcov, zpmed, clrmed, clrrms, exptime);
  ## TABLE gaia_ztf ( oid, candid, neargaia, neargaiabright, maggaia, maggaiabright, unique1);
  ## TABLE ss_ztf ( oid, candid, ssdistnr, ssmagnr, ssnamenr);
  ## TABLE ps1_ztf ( oid, candid, objectidps1, sgmag1, srmag1, simag1, szmag1, sgscore1, distpsnr1, objectidps2, sgmag2, srmag2, simag2, szmag2, sgscore2, distpsnr2, objectidps3, sgmag3, srmag3, simag3, szmag3, sgscore3, distpsnr3, nmtchps, unique1, unique2, unique3);
  ## TABLE reference (oid,rfid,candid,fid,rcid,field,magnr,sigmagnr,chinr,sharpnr,ranr,decnr,mjdstartref,mjdendref,nframesref);
  ## TABLE pipeline (pipeline_id, step_id_corr, step_id_feat, step_id_clf, step_id_out, step_id_stamp, date);

  # SQL Query
  {sql_pred}

  # Error returned when excecuted the query in the ALeRCE database
  {sql_error}

  # Using valid PostgreSQL, the CORRECT names of the tables only with their respective columns, and the information given in "Context", Correct the query given the error, using the correct database schema or nested queries to optimize.
  # Answer ONLY with the SQL query
  # Add COMMENTS IN PostgreSQL format so that the user can understand.
  # The query was generated from the next request:
  '''

# version 1 schema vs timeout
def prompt_self_correction(gen_task: str, tab_schema: str, gen_cntx: str, final_instructions: str, req: str, sql_pred: str, error: str) -> str:
  '''
  Fill the variables of the prompt with the specific information of the request, to generate the self-correction prompt.
  Two types of errors are considered: timeout and schema error, given the error returned when executing the query in the database.

  Parameters:
  gen_task (str): General task description for the self-correction task.
  tab_schema (str): Tables schema required for the query.
  gen_cntx (str): General context of the database schema.
  final_instructions (str): Instructions to correct the query.
  req (str): User request.
  sql_pred (str): SQL query that needs to be corrected.
  error (str): Error returned when executing the query in the ALeRCE database, in string format.

  Returns:
  str: Prompt with the specific information of the request.
  '''

  # Timeout error
  if 'timeout' in str(error):
    return f'''{gen_task}
    # Context:
    ## General information of the schema and the database
    {gen_cntx}

    # Correct a SQL query given the next user request:
    {req}
    # This are the tables Schema. Assume that only the next tables are required for the query:
    {tab_schema}

    # The next query is not working due to a timeout Error, correct the query using the correct database schema or nested queries to optimize.
    # SQL Query
    {sql_pred}
    # Error returned when executing the query in the ALeRCE database
    {error}

    # Follow the next advices to correct the query:
    - Check if the query is using the correct tables and columns, and if the conditions are correct, given the user request.
    - Check if the SQL code includes all the requested conditions.

    # If there are no problems with the previous steps, follow the next advices to correct the query:
    - Check if the SQL code includes the necessary conditions to optimize the query, and if the query is using the correct database schema or nested queries to optimize.
        - It is possible that the query is too complex and it is necessary to use nested queries to optimize the query.
        - If there is a JOIN or a sub-query between object and probability, check if the condition 'ranking=1' is set in the probability table, unless the request said otherwise.
    - Check if are at least 3 conditions over the probability table, because if not, the query is too general. Add more conditions if necessary.
    - If there are conditions involving dates or times, check if the dates are not too far away, or are in a reasonable range.

    {final_instructions}
    '''
  # Other error
  else:
    return f'''{gen_task}
    # Context:
    ## General information of the schema and the database
    {gen_cntx}
    # Correct a SQL query given the next user request:
    {req}

    # This are the tables Schema. Assume that only the next tables are required for the query:
    {tab_schema}

    # The next query is not working due to a syntax error, correct the query using the correct database schema.
    # SQL Query
    {sql_pred}
    # Error returned when executing the query in the ALeRCE database
    {error}

    # Follow the next advices to correct the query:
    - Check if the query is using the correct database schema. This includes the correct names of the tables and the correct names of the columns. If not, correct the query.
    - Check if the query have the correct syntax. If not, correct the query.
    - If there is a "missing FROM-clause entry", check where the table or sub-query is used in the query and add the correct name of the table or sub-query.

    {final_instructions}
    '''

# version f schema vs timeout
def prompt_self_correction_vf(gen_task, tab_schema, gen_cntx, final_instructions, req, sql_pred, error):
  # Timeout error
  if 'timeout' in str(error):
    return self_correction_timeout_prompt_vf.format(Self_correction_task= gen_task, request= req, tab_schema=tab_schema, sql_query=sql_pred, sql_error=error)
  # Other error
  else:
    return self_correction_schema_prompt_vf.format(Self_correction_task= gen_task, request= req, tab_schema=tab_schema, sql_query=sql_pred, sql_error=error)
  
# version 2 schema vs timeout
def prompt_self_correction_v2(gen_task: str, tab_schema: str, req: str, sql_pred: str, error):
  '''
  Fill the variables of the prompt with the specific information of the request, to generate the self-correction prompt.
  Three types of errors are considered: timeout, schema error, and no exist error, given the error returned when executing the query in the database.

  Parameters:
  gen_task (str): General task description for the self-correction task.
  tab_schema (str): Tables schema required for the query.
  req (str): User request.
  sql_pred (str): SQL query that needs to be corrected.
  error (str): Error returned when executing the query in the ALeRCE database, in string format.
  
  Returns:
  str: Prompt with the specific information of the request.
  '''

  # if the error is a psycopg2 error, extract the error message
  if "psycopg2.errors" in str(error):
        error_ = str(error).split("\n[SQL:")[0]
  else: 
    error_ = error
            
  # Timeout error
  if 'timeout' in str(error_):
    return self_correction_timeout_prompt_v2.format(Self_correction_task= gen_task, request= req, tab_schema=tab_schema, sql_query=sql_pred, sql_error=error_)
  # [object] does not exist error
  elif 'not exist' in str(error_):
    return self_correction_no_exist_prompt_v2.format(Self_correction_task= gen_task, request= req, tab_schema=tab_schema, sql_query=sql_pred, sql_error=error_)
  # Other error
  else:
    return self_correction_schema_prompt_v2.format(Self_correction_task= gen_task, request= req, tab_schema=tab_schema, sql_query=sql_pred, sql_error=error_)
  

