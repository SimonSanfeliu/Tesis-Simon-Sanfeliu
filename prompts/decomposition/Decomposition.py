### Decomposition Prompts
# This file contains the prompts for the decomposition task.
# The prompts are used to guide the model to generate a decomposition plan for a given user request, then to generate the SQL query based on the decomposition plan.
# For simple queries, the decomposition task is not needed, so the prompts are simplified.
# The decomposition task is composed of two main prompts: one for the decomposition task and another for the generation task.
## The decomposition task prompt is used to guide the model to generate a decomposition plan for a given user request.
## The generation task prompt is used to guide the model to generate the SQL query based on the decomposition plan.
# The structure of the prompt can be modified to include more information or to change the order of the sections with the prompt functions.
###

#### SIMPLE QUERY ####
# Simple queries don't require decomposition, so the base prompt for text2sql in ALeRCE can be simplified
# simple query general task of text-to-sql
# general prompt for the simple query generation task
simple_query_task='''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
## General Context of the database schema
simple_query_cntx='''
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as ‘lc_classifier’): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [periodic, stochastic, transient], denoted as ‘lc_classifier_top.’
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as ‘lc_classifier_periodic,’ ‘lc_classifier_transient,’ and ‘lc_classifier_stochastic,’ respectively.
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
'''
# Final instructions for the text2sql task, to emphasize the most important details
simple_query_instructions = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Avoid changing the names of columns or tables unless it is necessary for the SQL query.

# If you need to use 2 tables, try using a INNER JOIN statement, or a sub-query over 'probability' or 'object' if it is necessary (priority in this order).
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# Answer ONLY with a SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
DON'T include anything else in your answer.
'''

## More complex queries are decomposed into steps to guide the model in generating the SQL query
## First, a decomposition prompt is generated to guide the model in generating the decomposition plan
## Then, a generation prompt is generated to guide the model in generating the SQL query based on the decomposition plan

#### MEDIUM QUERY ####
### Decomposition Prompts for the medium query decomposition task ###
# medium query decomposition task
medium_decomp_task = '''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, etc.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
List the steps in the order in which they should be planned. Add to each numbered step a label in square brackets, like [initial planning], [join table], [replace], [condition], [join], [sub-query], etc.
With the labels mark explicitly in which step you should use a sub-query, and other statements.
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''
# medium query context, including details about the database schema
medium_query_cntx='''
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as ‘lc_classifier’): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [periodic, stochastic, transient], denoted as ‘lc_classifier_top.’
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as ‘lc_classifier_periodic,’ ‘lc_classifier_transient,’ and ‘lc_classifier_stochastic,’ respectively.
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
'''
# Final instructions for the medium decomposition task, to emphasize the most important details for the decomposition
medium_query_instructions_1 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' or 'object' if it is necessary (priority in this order).
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''
# medium decomposition prompt format
medium_decomp_prompt = '''
# {medium_decomp_task}
# General context about the database:
# {medium_query_cntx}
# {user_request_with_tables}
# # Important details about the database required for the query:
# {medium_query_instructions_1}
# '''
# return: decomposition steps

### Text2SQL prompts for the medium query generation task ###
# medium query general task of text-to-sql
# general prompt for the medium query generation task
medium_query_task='''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
# Final instructions for the medium query generation task, to emphasize the most important details for the query
medium_query_instructions_2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities, if the user request all ranking probabilities, don't use it.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Avoid changing the names of columns or tables unless it is necessary for the SQL query.

# If you need to use 2 tables, try using a sub-query over 'probability' TABLE, 'object' TABLE, over an INNER JOIN between 'probabbility' and 'object', if it is necessary (priority in this order).
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# Answer ONLY with the final SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
DON'T include anything else in your answer. If you want to add comments, use the SQL comment format inside the query.
'''
# medium generation prompt format
medium_decomp_gen =  '''
{medium_query_task}
{user_request_with_tables}
{medium_query_instructions_2}
# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''
# return: final query


#### ADVANCED QUERY ####
### Decomposition Prompts for the advanced query decomposition task ###
# Advanced query decomposition task
adv_decomp_task = '''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, etc.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
List the steps in the order in which they should be planned. Add to each numbered step a label in square brackets, like [initial planning], [join table], [replace], [condition], [join], [sub-query], etc.
The request is a very difficult and advanced query, so you will need to use JOINs, INTERSECTs and UNIONs statements, together with Nested queries. It is very important that you give every possible detail in each step, describing the statements and the nested-queries that are required.
'''
# Advanced query context, including details about the database schema
adv_query_cntx='''
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as ‘lc_classifier’): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [periodic, stochastic, transient], denoted as ‘lc_classifier_top.’
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as ‘lc_classifier_periodic,’ ‘lc_classifier_transient,’ and ‘lc_classifier_stochastic,’ respectively.
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
'''
# Final instructions for the advanced decomposition task, to emphasize the most important details for the decomposition
adv_query_instructions_1 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier'
### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, including an INNER JOIN with the 'probability' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the UNION query selecting the 'oids' in the INTERSECT query
VI. Add to the final result from step V the remaining conditions
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' TABLE, 'object' TABLE, over an INNER JOIN between 'probabbility' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (priority in this order).
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''
# Advanced decomposition prompt format
adv_decomp_prompt = '''
# {adv_decomp_task}
# General context about the database:
# {adv_query_cntx}
# {user_request_with_tables}
# # Important details about the database required for the query:
# {adv_query_instructions_1}
# '''
# return: decomposition steps

### Text2SQL prompts for the advanced query generation task ###
# Advanced query general task of text-to-sql
# General prompt for the advanced query generation task
adv_query_task= '''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
# Final instructions for the advanced query generation task, to emphasize the most important details for the query
adv_query_instructions_2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities, if the user request all ranking probabilities, don't use it.
- 'probability.classifier_name='lc_classifier'
### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, including an INNER JOIN with the 'probability' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the UNION query selecting the 'oids' in the INTERSECT query
VI. Add to the final result from step V the remaining conditions
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Avoid changing the names of columns or tables unless it is necessary for the SQL query.

# If you need to use 2 or 3 tables, try using a sub-query over 'probability' TABLE, 'object' TABLE, over an INNER JOIN between 'probabbility' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (priority in this order).
# Generate a query for each step, resolving and analysing it, with the following format:
```step_number [STEP QUERY HERE] ```
# Finally, join all the steps in a final query, with the following format: 
```sql [FINAL QUERY HERE] ```
DON'T include anything else inside and after your FINAL answer.
'''
# Advanced generation prompt format
adv_decomp_gen =  '''
{adv_query_task}
{user_request_with_tables}
{adv_query_instructions_2}
# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''
# return: final query






### decomposition prompt based on prompt generated by gpt4-turbo decomposition
# version 2
medium_decomp_task_v2 = '''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, etc.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''

gpt4turbo1106_decomposed_prompt_2 = '''Creating a decomposition plan to generate a PostgreSQL query for retrieving information from the ALeRCE astronomy broker database involves several steps. ALeRCE (Automatic Learning for the Rapid Classification of Events) is a system designed to classify large amounts of astronomical data, typically from surveys like the Zwicky Transient Facility (ZTF). To create a detailed and understandable plan, follow these steps:

1. **Understand the Database Schema:**
   - Obtain the database schema, which includes tables, columns, data types, relationships, and constraints.
   - Identify the relevant tables and columns that contain the information you need.

2. **Define the Information Needed:**
   - Clearly specify what information you want to retrieve. For example, you might be interested in transient events, their classifications, light curves, or cross-matches with other catalogs.
   - Determine the level of detail required (e.g., specific time ranges, magnitude limits, or particular sky regions).

3. **Formulate the Query Requirements:**
   - Decide on the selection criteria (e.g., date, magnitude, classification confidence).
   - Determine if you need to join multiple tables and how they are related.
   - Consider if you need to aggregate data (e.g., average magnitudes, count of events).

4. **Design the Query:**
   - Start with the main table that contains the bulk of the information you need.
   - Use `JOIN` clauses to combine related tables based on common keys.
   - Apply `WHERE` clauses to filter the data according to your criteria.
   - Use `GROUP BY` and aggregate functions if necessary.
   - Decide on the sorting order of the results using `ORDER BY`.

5. **Document the Query:**
   - Write comments within the SQL code to explain the purpose of different parts of the query.
   - Create external documentation that describes the query's purpose, the information it retrieves, and any assumptions or limitations.

Here's an example of a simple PostgreSQL query structure based on the steps above:

```sql
-- Retrieve transient events with their classifications and light curves
-- for a specific time range and magnitude limit

SELECT
    e.event_id,
    e.ra,
    e.dec,
    c.classification,
    c.confidence,
    lc.mag,
    lc.time
FROM
    events e
JOIN
    classifications c ON e.event_id = c.event_id
JOIN
    light_curves lc ON e.event_id = lc.event_id
WHERE
    e.time_observed BETWEEN '2023-01-01' AND '2023-01-31'
    AND lc.mag < 20
ORDER BY
    e.time_observed DESC, lc.time ASC;
```

Remember that the actual query will depend on the specific schema and requirements of the ALeRCE database. Always test your queries to ensure they perform as expected and return accurate results. 
'''

# Decomposition version 4
medium_decomp_prompt_v4 = f'''
{medium_decomp_task_v2}
{gpt4turbo1106_decomposed_prompt_2}
# General context about the database:
{medium_query_cntx}
# Important details about the database required for the query:
{medium_query_instructions_1}
{{req_prompt}}
'''
medium_decomp_prompt_v5 = f'''
{medium_decomp_task_v2}
# General context about the database:
{medium_query_cntx}
# Important details about the database required for the query:
{medium_query_instructions_1}
{gpt4turbo1106_decomposed_prompt_2}
{{req_prompt}}
'''

# version 2
adv_decomp_task_v2 = '''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, etc.
Take in consideration the advices, conditions and names from "General Context" and details of the database, or the query will not be optimal.
The request is a very difficult and advanced query, so you will need to use JOINs, INTERSECTs and UNIONs statements, together with Nested queries. It is very important that you give every possible detail in each step, describing the statements and the nested-queries that are required.
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''
adv_query_instructions_1_v2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier'
### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, taking only the oids in the 'probability' sub-query with an INNER JOIN inside each 'feature' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the 'UNION' query from step III selecting only the 'oids' that are in the 'INTERSECT' query from step IV
VI. Add the remaining conditions to the final result of step V, using the 'probability' sub-query from step I.
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' TABLE, 'object' TABLE, over an INNER JOIN between 'probabbility' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (priority in this order).
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''
adv_query_instructions_2_v2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities, if the user request all ranking probabilities, don't use it.
- 'probability.classifier_name='lc_classifier'
### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, taking only the oids in the 'probability' sub-query with an INNER JOIN inside each 'feature' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the 'UNION' query from step III selecting only the 'oids' that are in the 'INTERSECT' query from step IV
VI. Add the remaining conditions to the final result of step V, using the 'probability' sub-query from step I.
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Avoid changing the names of columns or tables unless it is necessary for the SQL query.

# If you need to use 2 or 3 tables, try using a sub-query over 'probability' TABLE, 'object' TABLE, over an INNER JOIN between 'probabbility' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (priority in this order).
# Finally, join all the steps in a final query, with the following format: 
```sql [FINAL QUERY HERE] ```
DON'T include anything else inside and after your FINAL answer.
'''

# version 4s
adv_decomp_prompt_v4 = f'''
{adv_decomp_task_v2}
{gpt4turbo1106_decomposed_prompt_2}
# General context about the database:
{adv_query_cntx}
# Important details about the database required for the query:
{adv_query_instructions_1_v2}
{{req_prompt}}
'''

# version 5
adv_decomp_prompt_v5 = f'''
{adv_decomp_task_v2}
# General context about the database:
{adv_query_cntx}
# Important details about the database required for the query:
{adv_query_instructions_1_v2}
{gpt4turbo1106_decomposed_prompt_2}
{{req_prompt}}
'''

# version f
## Simple
simple_query_task_vf= '''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) database. This database serves as a repository for information about astronomical variable objects. The information for every variable object originates from a sequence of one or more astronomical alerts, data packets streamed when an astronomical object shows a significant variation with respect to a reference image. The database information includes flux variations as a function of time (known as light curve), basic object properties such as the coordinates, and advanced features or statistics computed for each object.
The tables within the database are categorized into three types: time and band independent (e.g., object and probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection, forced-photometry). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''

simple_query_cntx_vf= '''
## General Information about the Schema and Database
- An object is uniquely identified by its object identifier or 'oid' index, used in most tables
- A detection from an object is identified by the candidate identifier or 'candid' index, used only in the detection table
- A given band is identified by the filter identifier or 'fidâ€™ index, used in the magstats, feature, and detection tables
- In most cases you will need to use information from the object table
- When particular astronomical classes are requested, you will need to use the probability table 
- Prioritize obtaining oids in a subquery to optimize the main query.
- Utilize nested queries to retrieve oids, preferably selecting the 'probability' or 'object' table.
- Avoid JOIN clauses; instead, favor nested queries.
- Beware of variables that are not indexed when doing the queries. Favour using nested queries where the inner queries use indexed variables.
- Note that the typical timeout time is 2 minutes
- Special attention needs to be paid to the feature table. In this table the name of a given feature is stored in the column 'name' and its value for a given object in the column 'value'. These columns are not indexed, so you should query this table in the outer levels of a nested query, after most of the filtering has already happened using indexed variables.

## ALeRCE Pipeline Details
- Stamp Classifier (denoted as ""stamp_classifierâ€™): A convolutional neural network that uses as input the image stamps from a given object and that uses a 5 class taxonomy. This classifier is triggered only by the first alert of every object.
- Light Curve Classifier (denoted as ""lc_classifierâ€™): A balanced hierarchical random forest classifier that uses as input object features and that consists of four models with a taxonomy of 15 classes in total. This classifier is triggered with every new alert of an object with at least six detections in a given band.
- The first hierarchical classifier of the Light Curve Classifier has three classes: [periodic, stochastic, transient], denoted as ""lc_classifier_top.â€™
- Three additional classifiers of the Light Curve Classifier specialize in different types of object: Periodic, Transient, and Stochastic, denoted as ""lc_classifier_periodic,â€™ ""lc_classifier_transient,â€™ and ""lc_classifier_stochastic,â€™ respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('Ceph'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Probability Variable Names
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('transient', 'stochastic', 'periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
'''
simple_query_instructions_vf='''
## Default Parameters to Consider
- Class probabilities for a given classifier and object are sorted from most to least likely, where the relative position is indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1.
- The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical classification. If no classifier is specified, use 'classifier_name=â€™lc_classifierâ€™ when selecting probabilities.
- If the user doesn't specify explicit columns, use the â€œSELECT *â€ SQL statement to choose all possible columns.
- Avoid changing the names of columns or tables unless necessary for the SQL query.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.

# If you need to use 2 tables, try using a INNER JOIN statement, or a sub-query over 'probability' or 'object' if it is necessary (in this order).
# Add COMMENTS IN PostgreSQL format so that the user can understand.
'''

## Medium
medium_query_task_vf = simple_query_task_vf
medium_query_cntx_vf = simple_query_cntx_vf
medium_query_instructions_1_vf = simple_query_instructions_vf
medium_query_instructions_2_vf = simple_query_instructions_vf

medium_decomp_task_vf = '''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, etc.
Take in consideration the advices, conditions and names from ""General Context"" and details of the database, or the query will not be optimal.
List the steps in the order in which they should be planned. Add to each numbered step a label in square brackets, like [initial planning], [join table], [replace], [condition], [join], [sub-query], etc.
With the labels mark explicitly in which step you should use a sub-query, and other statements.
'''


medium_decomp_prompt_vf = '''
# {medium_decomp_task}
# General context about the database:
# {medium_query_cntx}
# {user_request_with_tables}
# # Important details about the database required for the query:
# {medium_query_instructions_1}
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''



medium_decomp_gen_vf = '''
{medium_query_task}
{user_request_with_tables}
{medium_query_instructions_2}
# Answer ONLY with the final SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
DON'T include anything else in your answer. If you want to add comments, use the SQL comment format inside the query.

# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''

medium_decomp_gen_vf_python = '''
{medium_query_task}
{user_request_with_tables}
{medium_query_instructions_2}
# Answer ONLY with the final SQL query divided in different sub-queries given by a Python script, with the following format: 
  ```python PYTHON_SCRIPT_HERE ```
DON'T include anything else in your answer. If you want to add comments, use the Python comment format inside the query.
The variable that concatenates all the sub-queries MUST be named 'full_query'.

# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''


## Advanced
adv_query_task_vf=simple_query_task_vf
adv_query_cntx_vf=simple_query_cntx_vf
adv_query_instructions_1_vf = '''
## Default Parameters to Consider
- Class probabilities for a given classifier and object are sorted from most to least likely, where the relative position is indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1.
- The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical classification. If no classifier is specified, use 'classifier_name=â€™lc_classifierâ€™ when selecting probabilities.
- If the user doesn't specify explicit columns, use the â€œSELECT *â€ SQL statement to choose all possible columns.
- Avoid changing the names of columns or tables unless necessary for the SQL query.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.

### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, including an INNER JOIN with the 'probability' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the UNION query selecting the 'oids' in the INTERSECT query
VI. Add to the final result from step V the remaining conditions

### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the â€œSELECT *â€ SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' TABLE, 'object' TABLE, over an INNER JOIN between 'probability' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (in this order).
# Add COMMENTS IN PostgreSQL format so that the user can understand.
'''
adv_query_instructions_2_vf = '''
## Default Parameters to Consider
- Class probabilities for a given classifier and object are sorted from most to least likely, where the relative position is indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1.
- The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical classification. If no classifier is specified, use 'classifier_name=â€™lc_classifierâ€™ when selecting probabilities.
- If the user doesn't specify explicit columns, use the â€œSELECT *â€ SQL statement to choose all possible columns.
- Avoid changing the names of columns or tables unless necessary for the SQL query.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.

### IF THE 'feature' TABLE is used with 2 or more features, you need to take the following steps, because it is a transposed table (each feature is in a different row).
I. Create a sub-query using the 'probability' TABLE filtering the desired objects.
II. For each feature, you have to make a sub-query retrieving the specific feature adding the condition of its value, including an INNER JOIN with the 'probability' sub-query to retrieve only the features associated with the desired spatial objects.
III. Make an UNION between the sub-queries of each feature from step II
IV. Make an INTERSECT between the sub-queries of each feature from step II
V. Filter the UNION query selecting the 'oids' in the INTERSECT query
VI. Add to the final result from step V the remaining conditions

### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the â€œSELECT *â€ SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' TABLE, 'object' TABLE, over an INNER JOIN between 'probability' and 'object', or over an INNER JOIN between 'probability', 'object' and 'magstat', if it is necessary (in this order).
# Add COMMENTS IN PostgreSQL format so that the user can understand.
'''

adv_decomp_task_vf='''
# Your task is to DECOMPOSE the user request into a series of steps required to generate a PostgreSQL query that will be used for retrieving requested information from the ALeRCE database.
For this, outline a detailed decomposition plan for its systematic resolution, describing and breaking down the problem into subtasks and/or subqueries. 
Be careful to put all the information and details needed in the description, like conditions, the table and column names, etc.
Take in consideration the advices, conditions and names from ""General Context"" and details of the database, or the query will not be optimal.
List the steps in the order in which they should be planned. Add to each numbered step a label in square brackets, like [initial planning], [join table], [replace], [condition], [join], [sub-query], etc.
The request is a very difficult and advanced query, so you will need to use JOINs, INTERSECTs and UNIONs statements, together with Nested queries. It is very important that you give every possible detail in each step, describing the statements and the nested-queries that are required.
'''

adv_decomp_prompt_vf= '''
# {adv_decomp_task}
# General context about the database:
# {adv_query_cntx}
# {user_request_with_tables}
# # Important details about the database required for the query:
# {adv_query_instructions_1}
# DON'T RETURN ANY SQL CODE, just the description of each step required to generate it.
'''
adv_decomp_gen_vf='''
{adv_query_task}
{user_request_with_tables}
{adv_query_instructions_2}
# Generate a query for each step, resolving and analysing it, with the following format:
```step_number [STEP QUERY HERE] ```
# Finally, join all the steps in a final query, with the following format: 
```sql [FINAL QUERY HERE] ```
DON'T include anything else inside and after your FINAL answer.

# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''

adv_decomp_gen_vf_python='''
{adv_query_task}
{user_request_with_tables}
{adv_query_instructions_2}
# Generate a query for each step, resolving and analysing it, with the following format:
```python [VARIABLE SUB-QUERY HERE] ```
# Finally, join all the steps in a final query like so: 
```python full_query = [FINAL QUERY HERE] ```
DON'T include anything else inside and after your FINAL answer.

# Use the next decomposed planification to write the query:
{decomp_plan}
# If there is SQL code, use it only as reference, changing the conditions you consider necessary.
# You can join some of the steps if you consider it better for the query. For example, if 2 or more use the same table and are not requested to be different sub-queries, then you can join them.
'''






final_prompt_simple_vf='''
{simple_query_task}

# Context:
## General information of the schema and the database
# {simple_query_cntx}
# # Important details about the database required for the query:
# {simple_query_instructions}
# Answer ONLY with a SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
DON'T include anything else in your answer.

{request}
'''



### V2
# More simplified simple and medium query task
simple_query_task_v2='''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
# simple query context
simple_query_cntx_v2='''
## ALeRCE Pipeline Details:
- Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as ‘lc_classifier’): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [periodic, stochastic, transient], denoted as ‘lc_classifier_top.’
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as ‘lc_classifier_periodic,’ ‘lc_classifier_transient,’ and ‘lc_classifier_stochastic,’ respectively.
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
'''
# final instructions with the most important details
simple_query_instructions_v2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Avoid changing the names of columns or tables unless it is necessary for the SQL query.

# If you need to use 2 tables, try using a INNER JOIN statement, or a sub-query over 'probability' or 'object' if it is necessary (priority in this order).
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# Answer ONLY with a SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
DON'T include anything else in your answer.
'''
