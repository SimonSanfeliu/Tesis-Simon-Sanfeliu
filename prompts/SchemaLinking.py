### Schema Linking Task Prompts
# This file contains the prompts for the Schema Linking task.
# With Schema linking, the model is expected to identify the tables and columns needed to answer a question.
# The main prompt for the Schema Linking task is composed by a short general description of the task, followed by a detailed description of the tables and columns in the database.
# The structure of the prompt can be modified to include more information or to change the order of the sections using the prompt functions.
###

# Final instructions for the Schema Linking task, to emphasize the importance of providing only the tables needed to answer a question in a specific format.
## Version 1
sl_final_instructions_v1 = '''
# Give ONLY the TABLES that are needed to generate the SQL query, nothing more
# Give the answer in the following format: ['table1', 'table2', ...]
# For example, if the TABLES needed for the user request are TABLE object and TABLE taxonomy, then you should type: ['object', 'taxonomy']
# Remember to use the exact name of the TABLES, as they are written in the DATABASE SCHEMA
# Just give the tables and ignore any other task given in the request given as "request".
'''
## Version 2
sl_final_instructions_v2 = '''
# Information about the database
## Table object is the main table, so any request that ask for information about the object, will need this table.
## Table probability is the only table that has information of the classifications, so any request that ask for probabilities, any classifier, or somespecific object type (for example Supernovaes (SNe), 'SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other'), will need this table.
## Table feature is the only table that has information of the features used for the classification, so any request that ask for features, will need this table.

# Give ONLY the TABLES that are needed to generate the SQL query.
# Give the answer in the following format: ['table1', 'table2', 'table3', ...]. For example, if the TABLES needed for the user request are TABLE object and TABLE taxonomy, then you should type: ['object', 'taxonomy']
# Just give the tables and ignore any other task given in the request given as "request".
# Remember to use the exact name of the TABLES, as they are written in the DATABASE SCHEMA. Do NOT create table names.
'''
## Version 3
sl_final_instructions_v3 = '''
# Give ONLY the TABLES that are needed to generate the SQL query.
# Give the answer in the following format: ['table1', 'table2', 'table3', ...]. For example, if the TABLES needed for the user request are TABLE object and TABLE taxonomy, then you should type: ['object', 'taxonomy']
# Just give the tables and ignore any other task given in the request given as "request".
# Remember to use the exact name of the TABLES, as they are written in the DATABASE SCHEMA. Do NOT create table names.
# If you think that no table mentioned above is needed, then type: ""
'''

# general prompt for the Schema Linking task, with the variables to be filled with the specific information of the request
## Version 1
tables_linking_prompt = '''
# Given the user request, select the tables needed to generate a SQL query
# The Database has the following tables:
{table_schema}

{final_instructions}
'''
## Version 2
### w/ sl_final_instructions_v2

## Version 3
### w/ sl_final_instructions_v3

## Version 4
## Multiple prompts to separate the decision if the object and probability tables are needed due to their importance in the database.
## Then, there is a final prompt to select the other tables needed to generate the SQL query.
### object table prompt
tables_linking_prompt_v4_1 = '''
# Given the user request, you have to decide if the table object is needed to generate a SQL query
# DATABASE SCHEMA
## Information about the table object:
TABLE "object": contains basic filter and time–aggregated statistics such as location, number of observations, and the times of first and last detection.
columns= [
    oid VARCHAR PRIMARY KEY,  /* object identifier */
    ndethist INTEGER,  /* number of posible detections above 3 sigma */
    ncovhist INTEGER,  /* number of visits */
    mjdstarthist DOUBLE PRECISION,  /* time of first observation even if not detected */
    mjdendhist DOUBLE PRECISION, /* time of last observation even if not detected */
    corrected BOOLEAN, /* whether the object was corrected */
    stellar BOOLEAN, /* whether the object is likely psf shaped */
    ndet INTEGER, /* number of detections */
    g_r_max DOUBLE PRECISION, /* g-r difference color at maximum */
    g_r_max_corr DOUBLE PRECISION, /* g-r color at maximum*/
    g_r_mean DOUBLE PRECISION, /* mean g-r difference color */
    g_r_mean_corr DOUBLE PRECISION, /* mean g-r color */
    meanra DOUBLE PRECISION,  /* mean right ascencion */
    meandec DOUBLE PRECISION,  /* mean declination */
    sigmara DOUBLE PRECISION, /* right ascension dispersion */
    sigmadec DOUBLE PRECISION, /* declination dispersion */
    deltajd DOUBLE PRECISION, /* time difference between last and first detection */
    firstmjd DOUBLE PRECISION, /* time of first detection */
    lastmjd DOUBLE PRECISION, /* time of last detection */
    step_id_corr VARCHAR,
    diffpos BOOLEAN, /* whether the first detection was positive or negative */
    reference_change BOOLEAN /* whether the reference image changes */]
# Information about the database
## It is necessary only if it asks for information about the object, for example, if it asks for a specific number of detections, the number of observations, the times of first and last detection, etc.
## If the user does not ask for information in the Table (check the columns), then the table is not needed.

# Is the table object needed to generate the SQL query?
# Answer ONLY "yes" or "no":
'''
### probability table prompt
tables_linking_prompt_v4_2 = '''
# Given the user request, you have to decide if the table probability is needed to generate a SQL query
# DATABASE SCHEMA
## Information about the table probability:
TABLE "probability": classification probabilities associated to a given object, classifier, and class. Contain the object classification probabilities, including those from the stamp and light curve classifiers, and from different versions of these classifiers.
columns=[
    oid VARCHAR PRIMARY KEY, /* object identifier */
    class_name, /* name of the class */
    classifier_name, /* name of the classifier */
    classifier_version, /* version of the classiifer */
    probability, /* probability of the class given a classifier and version */
    ranking /* class probability ranking (1 is the most likely class) */
    ]

# Information about the database
## Table probability is the only table that has information of the classifications, so any request that ask for probabilities, any classifier, or somespecific object type (for example Supernovaes (SNe), 'SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other'), will need this table.

# Is the table probability needed to generate the SQL query?
# Answer ONLY "yes" or "no":
'''

### other tables prompt
tables_linking_prompt_v4_3 = '''
# Given the user request, select the tables needed to generate a SQL query
# DATABASE SCHEMA
## The Database has the following tables:
{table_schema}

{final_instructions}
'''

### Final prompt for the Schema Linking task, version 4
tables_linking_prompt_v4 = [{"prompt": tables_linking_prompt_v4_1, "table": "object"},
                              {"prompt": tables_linking_prompt_v4_2, "table": "probability"}]


## Schema Linking Prompt Functions
# The functions are used to fill the variables of the prompt with the specific information of the request.
# It is possible to modify the functions to include more information or to change the order of the sections of the prompt.
##
# version 1
def prompt_schema_linking_v1(table_schema: str, final_instructions: str) -> str:
    '''
    Generate the prompt for the Schema Linking task, version 1.

    Parameters:
    table_schema (str): The information about the tables in the database.
    final_instructions (str): The final instructions for the task.

    Returns: 
    str: the final prompt for the Schema Linking task.
    '''
    prompt = tables_linking_prompt.format(table_schema=table_schema, final_instructions=final_instructions)
    return prompt
# version 2
def prompt_schema_linking_v2(table_schema: str, final_instructions: str) -> str:
    '''
    Generate the prompt for the Schema Linking task, version 2
    
    Parameters:
    table_schema (str): The information about the tables in the database.
    final_instructions (str): The final instructions for the task.

    Returns: 
    str: the final prompt for the Schema Linking task.
    '''
    prompt = tables_linking_prompt_v4_3.format(table_schema=table_schema, final_instructions=final_instructions)
    return prompt













######################################


#### Basic Schema Linking Prompt, to retrieve the tables and columns needed to answer a question
schema_linking_prompt = '''
# Given the user request, select the tables needed to generate a SQL query
# The Database has the following tables:
TABLE object, columns=[oid, ndethist, ncovhist, mjdstarthist, mjdendhist, corrected, stellar, ndet, g_r_max, g_r_max_corr, g_r_mean, g_r_mean_corr, meanra, meandec, sigmara, sigmadec, deltajd, firstmjd, lastmjd, step_id_corr, diffpos, reference_change]
TABLE probability, columns=[oid, class_name, classifier_name, classifier_version, probability, ranking]
TABLE feature, columns=[oid, name, value, fid, version]
TABLE magstat, columns=[oid, fid, stellar, corrected, ndet, ndubious, dmdt_first, dm_first, sigmadm_first, dt_first, magmean, magmedian, magmax, magmin, magsigma, maglast, magfirst, magmean_corr, magmedian_corr, magmax_corr, magmin_corr, magsigma_corr, maglast_corr, magfirst_corr, firstmjd, lastmjd, step_id_corr, saturation_rate]
TABLE non_detection, columns=[oid, fid, mjd, diffmaglim]
TABLE detection, Columns=[candid, oid, mjd, fid, pid, diffmaglim, isdiffpos, nid, ra, dec, magpsf, sigmapsf, magap, sigmagap, distnr, rb, rbversion, drb, drbversion, magapbig, sigmagapbig, rfid, magpsf_corr, sigmapsf_corr, sigmapsf_corr_ext, corrected, dubious, parent_candid, has_stamp, step_id_corr]
TABLE step, columns=[step_id, name, version, comments, date]
TABLE taxonomy, columns=[ classifier_name, classifier_version, classes]
TABLE feature_version, columns=[version, step_id_feature, step_id_preprocess]
TABLE xmatch, columns=[oid, catid, oid_catalog, dist, class_catalog, period]
TABLE allwise, columns=[oid_catalog, ra, dec, w1mpro, w2mpro, w3mpro, w4mpro, w1sigmpro, w2sigmpro, w3sigmpro, w4sigmpro, j_m_2mass, h_m_2mass, k_m_2mass, j_msig_2mass, h_msig_2mass, k_msig_2mass]
TABLE dataquality, columns=[candid, oid, fid, xpos, ypos, chipsf, sky, fwhm, classtar, mindtoedge, seeratio, aimage, bimage, aimagerat, bimagerat, nneg, nbad, sumrat, scorr, dsnrms, ssnrms, magzpsci, magzpsciunc, magzpscirms, nmatches, clrcoeff, clrcounc, zpclrcov, zpmed, clrmed, clrrms, exptime]
TABLE gaia_ztf, columns=[oid, candid, neargaia, neargaiabright, maggaia, maggaiabright, unique1]
TABLE ss_ztf, columns=[oid, candid, ssdistnr, ssmagnr, ssnamenr]
TABLE ps1_ztf, columns=[oid, candid, objectidps1, sgmag1, srmag1, simag1, szmag1, sgscore1, distpsnr1, objectidps2, sgmag2, srmag2, simag2, szmag2, sgscore2, distpsnr2, objectidps3, sgmag3, srmag3, simag3, szmag3, sgscore3, distpsnr3, nmtchps, unique1, unique2, unique3]
TABLE reference, columns=[oid, rfid, candid, fid, rcid, field, magnr, sigmagnr, chinr, sharpnr, ranr, decnr, mjdstartref, mjdendref, nframesref]
TABLE pipeline, columns=[pipeline_id, step_id_corr, step_id_feat, step_id_clf, step_id_out, step_id_stamp, date]
TABLE information_schema.tables, columns=[table_catalog, table_schema, table_name, table_type, self_referencing_column_name, reference_generation, user_defined_type_catalog, user_defined_type_schema, user_defined_type_name, is_insertable_into, is_typed, commit_action]

# Give the answer in the following format: [table1, table2, ...]
# For example, if the answer is table object and table taxonomy, then you should type: [object, taxonomy]
# Request: {}
# Answer: 
'''

### Tables Linking Prompt, focusing on the tables needed to answer a question
#### Version 1
tables_linking_prompt_V1 = '''
# Given the user request, select the tables needed to generate a SQL query
# The Database has the following tables:
TABLE object: contains basic filter and time–aggregated statistics such as location, number of observations, and the times of first and last detection.
TABLE probability: contain the object classification probabilities, including those from the stamp and light curve classifiers, and from different versions of these classifiers.
TABLE feature: contains the object light curve statistics and other features used for ML classification and which are stored as json files in our database.
TABLE magstat: contains time–aggregated statistics separated by filter, such as the average magnitude, or the initial magnitude change rate.
TABLE non_detection: contains the limiting magnitudes of previous non–detections separated by filter.
TABLE detection: contains the object light curves including their difference and corrected magnitudes and associated errors separated by filter (see Section 4.4).
TABLE step: 
TABLE taxonomy: contains details about the different taxonomies used in our stamp and light curve classifiers, which can evolve with time.
TABLE feature_version: 
TABLE xmatch: contains the object cross–matches and associated cross–match catalogs.
TABLE allwise: 
TABLE dataquality: detailed object information regarding the quality of the data
TABLE gaia_ztf: GAIA objects near detected ZTF objects
TABLE ss_ztf: known solar system objects near detected objects
TABLE ps1_ztf: PanSTARRS objects near detected ZTF objects
TABLE reference: properties of the reference images used to build templates
TABLE pipeline: 
TABLE information_schema: information about the database tables and columns

# Give ONLY the TABLES that are needed to generate the SQL query, nothing more
# Give the answer in the following format: ['table1', 'table2', ...]
# For example, if the TABLES needed for the user request are TABLE object and TABLE taxonomy, then you should type: ['object', 'taxonomy']
# Remember to use the exact name of the TABLES, as they are written in the DATABASE SCHEMA
# Just give the tables and ignore any other task given in the request given as "request".
'''
#### Version 2, w/ References
tables_linking_prompt_V2 = '''
# Given the user request, select the tables needed to generate a SQL query
# The Database has the following tables:
TABLE object: contains basic filter and time–aggregated statistics such as location, number of observations, and the times of first and last detection.
TABLE probability: contain the object classification probabilities, including those from the stamp and light curve classifiers, and from different versions of these classifiers.
TABLE feature: contains the object light curve statistics and other features used for ML classification and which are stored as json files in our database.
TABLE magstat: contains time–aggregated statistics separated by filter, such as the average magnitude, or the initial magnitude change rate.
TABLE non_detection: contains the limiting magnitudes of previous non–detections separated by filter.
TABLE detection: contains the object light curves including their difference and corrected magnitudes and associated errors separated by filter (see Section 4.4).
TABLE step: 
TABLE taxonomy: contains details about the different taxonomies used in our stamp and light curve classifiers, which can evolve with time.
TABLE feature_version: 
TABLE xmatch: contains the object cross–matches and associated cross–match catalogs.
TABLE allwise: 
TABLE dataquality: detailed object information regarding the quality of the data
TABLE gaia_ztf: GAIA objects near detected ZTF objects
TABLE ss_ztf: known solar system objects near detected objects
TABLE ps1_ztf: PanSTARRS objects near detected ZTF objects
TABLE reference: properties of the reference images used to build templates
TABLE pipeline: 
TABLE information_schema: information about the database tables and columns

# References Keys
probability(oid) VARCHAR REFERENCES object(oid),
feature(oid) VARCHAR REFERENCES object(oid),
feature(version) VARCHAR REFERENCES feature_version(version) NOT NULL,
magstat(oid) VARCHAR REFERENCES object(oid),
non_detection(oid) VARCHAR REFERENCES object(oid),
detection(oid) VARCHAR REFERENCES object(oid),
feature_version(step_id_feature) VARCHAR REFERENCES step(step_id),
feature_version(step_id_preprocess) VARCHAR REFERENCES step(step_id)
xmatch(oid) VARCHAR REFERENCES object(oid),
gaia_ztf(oid) VARCHAR REFERENCES object(oid),
ss_ztf(oid) VARCHAR REFERENCES object(oid),
ps1_ztf(oid) VARCHAR REFERENCES object(oid),
reference(oid) VARCHAR REFERENCES object(oid),

# Give ONLY the TABLES that are needed to generate the SQL query, nothing more
# Give the answer in the following format: ['table1', 'table2', ...]
# For example, if the TABLES needed for the user request are TABLE object and TABLE taxonomy, then you should type: ['object', 'taxonomy']
# Remember to use the exact name of the TABLES, as they are written in the DATABASE SCHEMA
# Just give the tables and ignore any other task given in the request given as "request".
'''


### Columns Linking Prompt
#### Version 1
cols_linking_prompt = '''
# Given the user request, select the columns needed to generate a SQL query
# Answer
'''
