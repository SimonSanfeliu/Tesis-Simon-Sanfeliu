### Classification Task Prompts
# This file contains the prompts for the classification task.
# The prompts are used to guide the model to classify the difficulty of the query based on the request and the tables required for the query.
# The main prompt for the classification task is composed by a general description of the task, the description of the difficulty levels, and the tables required for the query.
# The structure of the prompt can be modified to include more information or to change the order of the sections with the prompt functions.
###

# General task description prompt
## version 1
general_task_classification_v1='''
You are a SQL expert with a willingness to assist users, your final task is to create a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). 
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
# Final instructions for the difficulty classification task, to emphasize the importance of providing only the predicted difficulty and other relevant information.
## version 1
final_instructions_diff_v1 = '''
# Give ONLY the predicted difficulty, nothing more
# Give the answer in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the only need a simple join between object and probability, then you should type: "label: simple"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given in the request given as "request".
'''
## version 2
final_instructions_diff_v2 = '''
# Give ONLY the predicted difficulty, nothing more.
# Give the answer in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the only need a simple join or nested query between object and probability, then you should type: "label: simple"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given in the request given as "request".
'''
## version 3
final_instructions_diff_v3 = '''
# Give the answer in the following format: "class: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the only need a simple join or nested query between object and probability, then you should type: "class: simple"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given inside the user request. Do NOT give any other information, like the SQL query.
'''

# general prompt for the classification task, with the variables to be filled with the specific information of the request
## w/ final_instructions_diff_v1
diff_class_prompt_v1 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.
### "simple":
If (Only 1 table is used, OR 2 common tables (probability, object or magstat) are used)
OR (No nested-query or JOIN clause is neccesary, OR need only a simple nested-query, OR a simple JOIN between probability and object tables)
### "medium":
If (2 tables are used, OR 3 common tables (probability, object, magstat, or features with only one feature) are used)
OR (Need 1 complex nested-query (clause other than 'WHERE' on probability))
### "advanced":
If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required)

# Tables required for the query:
{table_schema}

{final_instructions_diff}
'''

## w/ final_instructions_diff_v2
diff_class_prompt_v2 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.
## "simple":
If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only a simple nested-query between probability, object or magstat tables is required, OR a simple JOIN between probability, object or magstat tables)
## "medium":
If (2 not common tables are used (NOT probability, object, magstat TABLES), OR 3 most common tables (probability, object and magstat TABLES) are used, OR features with only one feature) are used)
OR (Need 1 complex nested-query, OR a complex JOIN)
OR (Need 2 simple nested-query, OR 2 simple JOIN, OR 1 simple nested-query and 1 simple JOIN)
## "advanced":
If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required)

# Tables required for the query:
{table_schema}

{final_instructions_diff}
'''

## w/ final_instructions_diff_v2
diff_class_prompt_v3 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.

If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only a simple nested-query between probability, object or magstat tables is required, OR a simple JOIN between probability, object or magstat tables):
THEN "label: simple"

If (2 not common tables are used (NOT probability, object, magstat TABLES))
OR (3 most common tables (probability, object and magstat TABLES) are used)
OR (features with only one feature are used)
OR (Need 1 complex nested-query, OR a complex JOIN)
OR (Need 2 simple nested-query, OR 2 simple JOIN, OR 1 simple nested-query and 1 simple JOIN):
THEN "label: medium"

If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required):
THEN "label: advanced"

# Assume this are the only tables required for the query:
{table_schema}

{final_instructions_diff}
'''

## w/ final_instructions_diff_v2
diff_class_prompt_v4 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.

If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only a simple nested-query between probability, object or magstat tables is required, OR a simple JOIN between probability, object or magstat tables):
THEN "label: simple"

If (2 not common tables are used (NOT probability, object, magstat TABLES))
OR (3 most common tables (probability, object and magstat TABLES) are used)
OR (2 most common tables (probability, object and magstat TABLES) with only one feature from feature TABLE are used)
OR (Need 1 complex nested-query, OR a complex JOIN)
OR (Need 2 simple nested-query, OR 2 simple JOIN, OR 1 simple nested-query and 1 simple JOIN):
THEN "label: medium"

If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required):
THEN "label: advanced"

# Assume this are the only tables required for the query:
{table_schema}

{final_instructions_diff}
'''

## w/ general_task_classification_v1
## w/ final_instructions_diff_v2
diff_class_prompt_v5 = '''
{general_task_classification}
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.

If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only a simple nested-query between probability, object or magstat tables is required, OR a simple JOIN between probability, object or magstat tables):
THEN "label: simple"

If (2 not common tables are used (NOT probability, object, magstat TABLES))
OR (3 most common tables (probability, object and magstat TABLES) are used)
OR (features with only one feature are used)
OR (Need 1 complex nested-query, OR a complex JOIN)
OR (Need 2 simple nested-query, OR 2 simple JOIN, OR 1 simple nested-query and 1 simple JOIN):
THEN "label: medium"

If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required):
THEN "label: advanced"

# Assume this are the only tables required for the query:
{table_schema}

{final_instructions_diff}
'''

## w/ final_instructions_diff_v3
diff_class_prompt_v6 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.

If (2 not common tables are used (NOT probability, object, magstat TABLES))
OR (3 most common tables (probability, object and magstat TABLES) are used)
OR (features with only one feature are used)
OR (Need 1 complex nested-query, OR a complex JOIN)
OR (Need 2 simple nested-query, OR 2 simple JOIN, OR 1 simple nested-query and 1 simple JOIN):
THEN it is a "medium" query

If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required):
THEN it is a "advanced" query

If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only a simple nested-query between probability, object or magstat tables is required, OR a simple JOIN between probability, object or magstat tables):
THEN it is a "simple"

# Assume this are the only tables required for the query:
{table_schema}

{final_instructions_diff}
User Request: ""{request}""
First let's understand the request and the tables required for the query, 
'''
## simple vs other prompt
## w/ final_instructions_diff_v2
diff_class_prompt_v7 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.

If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only one nested-query between 'probability', 'object' or 'magstat' TABLES is required, OR one JOIN between 'probability', 'object' or 'magstat' TABLES):
THEN "label: simple"

If (2 not common tables are used (NOT probability, object, magstat TABLES))
OR (3 most common tables (probability, object and magstat TABLES) are used)
OR (features with only one feature are used)
OR (Need 1 very complex nested-query, OR a very complex JOIN)
OR (Need 2 nested-query, OR 2 JOIN, OR 1 nested-query with 1 JOIN):
THEN "label: medium"

If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required):
THEN "label: advanced"

# Assume this are the only tables required for the query:
{table_schema}

{final_instructions_diff}
'''

# simple vs other prompt. Final instructions are different for each difficulty level
# simple vs other final instructions 
## version 1
final_instructions_diff_simple_v1 = '''
# Give ONLY the predicted difficulty, nothing more.
# Give the answer in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the query ONLY need a simple join or nested query between 'object' and 'probability' TABLES, then you should type: "label: simple"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given inside the user request. Do NOT give any other information, like the SQL query.
'''
# medium vs advanced final instructions
final_instructions_diff_other_v1 = '''
# Give ONLY the predicted difficulty, nothing more.
# Give the answer in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the query requires 4 tables, then you should type: "label: advanced"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given in the request given as "request".
'''
## version 2
final_instructions_diff_simple_v2 = '''
**Provide ONLY the predicted difficulty in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.**

**For example:**
- The query requires only a direct join between 'object' and 'probability' tables, then: "label: simple"
- The query requires only the table 'magstat', then: "label: simple"
- The query requires a complex join between 'object' and 'probability' tables, then: "label: other"

**Remember:**
- Do not consider the number of where clauses or the complexity of the where clauses.
- Use the exact labels provided above.
- Ignore any other task in the user request.

**Before providing the answer, explain the reasoning behind your choice.**
'''
final_instructions_diff_other_v2 = '''
**Provide ONLY the predicted difficulty in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.**

**For example:**
- The query requires 4 tables, type: "label: advanced"
- The query requires 3 tables, but the tables are common, type: "label: medium"
- The query requires to join tables and specify a complex nested query, type: "label: advanced"

**Remember:**
- Use the exact labels provided above.
- Ignore any other tasks given in the user request.

**Before providing the answer, explain the reasoning behind your choice.**
'''

final_instructions_diff_simple_v3 = '''
**Provide ONLY the predicted difficulty in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.**

**For example:**
- The query requires only a direct join between 'object' and 'probability' tables, then: "label: simple"
- The query requires only the table 'magstat', then: "label: simple"
- The query requires a complex join between 'object' and 'probability' tables, then: "label: other"

**Remember:**
- Do not consider the number of where clauses or the complexity of the where clauses.
- Use the exact labels provided above.
- Ignore any other task in the user request.
'''

final_instructions_diff_other_v3 = '''
**Provide ONLY the predicted difficulty in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.**

**For example:**
- The query requires 4 tables, type: "label: advanced"
- The query requires 3 tables, but the tables are common, type: "label: medium"
- The query requires to join tables and specify a complex nested query, type: "label: advanced"

**Remember:**
- Use the exact labels provided above.
- Ignore any other tasks given in the user request.
'''

# simple vs other prompt.
## first step (_1) simple vs other
## second step (_2) medium vs advanced
# final_instructions_diff_simple_v1
diff_class_prompt_v8_1 = '''
# For the given request, classify it by difficulty as "simple" or "other" based on the next description.

If AT LEAST ONE of the next conditions are met:
- Only 1 table is required
- Only one nested-query and/or one JOIN between 'probability', 'object' or 'magstat' TABLES is required at most
THEN "label: simple"

ELSE: "label: other"

{final_instructions_diff}

# table(s) required for the query:
{table_schema}
User Request: ""{request}""
'''

# final_instructions_diff_other_v1
diff_class_prompt_v8_2 = '''
# For the given request, classify it by difficulty as "medium" or "advanced" based on the next description.

If ONE of the next conditions are met:
- 2 not common tables are used (NOT 'probability', 'object', 'magstat' TABLES)
- 3 most common tables ('probability', 'object' and 'magstat' TABLES) are used)
- Only 1 feature from the features table is required
- Need 1 complex nested-query, OR a complex JOIN)
- Need 2 nested-query, OR 2 JOIN, OR 1 nested-query with 1 simple JOIN
THEN "label: medium"

If AT LEAST ONE of the next conditions are met:
- 2 or more nested query are needed)
- If 3 tables or more are used)
- If 2 features from the 'feature' TABLE are required
THEN "label: advanced"

{final_instructions_diff}

# Assume this are the ONLY tables required for the query:
{table_schema}
User Request: ""{request}""
'''



diff_class_prompt_v9_1 = '''
**Classify the following request by difficulty as "simple" or "other" based on the criteria below.**

**Label as "simple" if ANY of the following conditions are met:**
- Requires only ONE nested query and/or one JOIN between 'probability', 'object', or 'magstat' tables
- Requires only ONE table

**Otherwise, label as "other".**

**Table(s) required for the query:**
{table_schema}

{final_instructions_diff}
'''

diff_class_prompt_v9_2 = '''
**Classify the following request by difficulty as "medium" or "advanced" based on the criteria below.**

**Label as "medium" if ANY of the following conditions are met:**
- Uses TWO uncommon tables (not 'probability', 'object', or 'magstat')
- Uses the THREE common tables ('probability', 'object', and 'magstat')
- Requires only ONE feature from the 'feature' table
- Requires only ONE complex nested query or ONE complex JOIN
- Requires TWO direct nested queries, TWO direct JOINS, or ONE direct nested query with ONE direct JOIN

**Label as "advanced" if ANY of the following conditions are met:**
- Requires TWO or more complex nested queries
- Requires TWO complex JOINS
- Uses THREE or more non common tables tables
- Uses FOUR or more tables
- Requires TWO features from the 'feature' table

**Assume these are the ONLY tables required for the query:**
{table_schema}

{final_instructions_diff}
'''

diff_class_prompt_v10_1 = '''
**Classify the following request by difficulty as "simple" or "other" based on the criteria below.**
**'probability', 'object', and 'magstat' tables are defined as common tables.**

**Label as "simple" if ONE of the following conditions are met:**
- Requires only ONE non common table
- Requires at most two common tables ('probability', 'object', or 'magstat')

**Otherwise, label as "other".**

{final_instructions_diff}

**Consider only the tables defined in the schema below.**
**Schema of the table(s) required for the query:**
{table_schema}
'''

diff_class_prompt_v10_2 = '''
**Classify the following request by difficulty as "medium" or "advanced" based on the criteria below.**
**'probability', 'object', and 'magstat' tables are defined as common tables.**

**Label as "medium" if ANY of the following conditions are met:**
- Uses TWO uncommon tables (not 'probability', 'object', or 'magstat')
- Uses the THREE common tables ('probability', 'object', and 'magstat')
- Requires only ONE feature from the 'feature' table
- Requires only ONE complex nested query or ONE complex JOIN
- Requires TWO direct nested queries, TWO direct JOINS, or ONE direct nested query with ONE direct JOIN

**Label as "advanced" if ANY of the following conditions are met:**
- Requires TWO or more complex nested queries
- Requires TWO complex JOINS
- Uses THREE or more non common tables
- Uses FOUR or more tables, including common and non common tables
- Requires TWO or more features from the 'feature' table

{final_instructions_diff}

**Consider only the tables defined in the schema below.**
**Schema of the table(s) required for the query:**
{table_schema}
'''




## Difficulty classification prompt functions. 
# The functions are used to fill the variables of the prompt with the specific information of the request.
# It is possible to modify the functions to include more information or to change the order of the sections of the prompt.
## 
## version 1
def prompt_diff_class_v1(diff_class_prompt: str, table_schema: str, final_instructions_diff: str, **kwargs) -> str:
    '''
    Fill the variables of the prompt with the specific information of the request.
    
    Parameters: 
    diff_class_prompt (str): The prompt for the difficulty classification task.
    table_schema (str): The tables required for the query.
    final_instructions_diff (str): The final instructions for the difficulty classification task.

    Returns:
    str: The prompt for the difficulty classification task with the specific information of the request.
    
    '''
    return diff_class_prompt.format(table_schema=table_schema, final_instructions_diff=final_instructions_diff)
## version 2
def prompt_diff_class_v2(diff_class_prompt: str, table_schema: str, final_instructions_diff: str, general_task_classification: str, **kwargs) -> str:
    '''
    Fill the variables of the prompt with the specific information of the request.

    Parameters:
    diff_class_prompt (str): The prompt for the difficulty classification task.
    table_schema (str): The tables required for the query.
    final_instructions_diff (str): The final instructions for the difficulty classification task.
    general_task_classification (str): The general description of the task.

    Returns:
    str: The prompt for the difficulty classification task with the specific information of the request.
    '''
    return diff_class_prompt.format(general_task_classification=general_task_classification, 
                                    table_schema=table_schema, final_instructions_diff=final_instructions_diff)
## version 3
def prompt_diff_class_v3(diff_class_prompt: str, table_schema: str, final_instructions_diff: str, request: str, **kwargs) -> str:
    '''
    Fill the variables of the prompt with the specific information of the request.

    Parameters:
    diff_class_prompt (str): The prompt for the difficulty classification task.
    table_schema (str): The tables required for the query.
    final_instructions_diff (str): The final instructions for the difficulty classification task.
    request (str): The user request.

    Returns:
    str: The prompt for the difficulty classification task with the specific information of the request.
    '''
    return diff_class_prompt.format(table_schema=table_schema, final_instructions_diff=final_instructions_diff, request=request)






###########################

# basic classification prompts
## difficulty classification
diff_class_prompt = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.
### "simple":
I Only 1 table is used, or 2 common tables (probability, object or magstat) are used
II No nested-query or JOIN clause is neccesary or need a simple nested-query or JOIN between probability and object tables
### "medium":
I If 2 or 3 tables are used
II Need 1 complex nested-query (clause other than 'WHERE' on probability)
### "advanced":
If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required)
# ONLY return the value as such: 'label: "<difficulty>"', nothing more.
'''
# spatial classification
type_class_prompt = '''
### - Type of query
###   - Object query
###     - Request of specific objects, given by their probabilities, features, characteristics, etc
###   - Spatial query
###     - Use q3 or coordinates
###   - Other:
###     - requests of periods or ligth curves query
###     - Requests for specific features or info about the light curves more than the object
'''
# nested query classification
nested_class_prompt = '''
### - Nested query
###   - Tree nested query
###     - Use multiple nested queries to join/uninon/intersect tables and/or to join tables
###   - Multi-nested query
###     - at least one nested query inside other nested query
###   - Simple
###     - one or two simple nested query
###   - None
###     - No nested query are needed
'''

# Classification description
### - The requests are modified to request only the information that can be obtained from the database.

### - Difficulty Classification
### Ways to be classified: Number of tables; types of tables;
###   - Advanced:
###     - If 3 tables or more are used
###     - If two features from features are used
###     - If 2 or more nested query are used
###   - Medium:
###     - If 2 tables are used +
###     - If it only use 1 complex nested-query (clause other than 'WHERE' on probability)
###   - Simple:
###     - 1 table or 2 common tables (prob-obj)
###     - If it only use one table or do a simple nested-query/Join

### - Type of query
###   - Object query
###     - Request of specific objects, given by their probabilities, features, characteristics, etc
###   - Spatial query
###     - Use q3 or coordinates
###   - Other:
###     - requests of periods or ligth curves query
###     - Requests for specific features or info about the light curves more than the object

### - Nested query
###   - Tree nested query
###     - Use multiple nested queries to join/uninon/intersect tables and/or to join tables
###   - Multi-nested query
###     - at least one nested query inside other nested query
###   - Simple
###     - one or two simple nested query
###   - None
###     - No nested query are needed

# Context Rules V2
### - general_context: Tips or information about the most important information of the database, like how to use the probability table or the values each column can have. Also set default parameters.

### - External Knowledge: Minimum information required to create the query with information that couldn't be retrieved from information of the database, like parameters or function calling answers, e.g. mjd time to put in the query,

### - Domain Knowledge:
### "Domain Knowledge: this
### category consists of domain-specific knowledge that is utilized to generate SQL operations [10, 59].
### For instance, a business analyst in the banking business may require knowledge of financial indicators
### such as return on investment and net income in order to generate effective SQL queries" ([BIRD-SQL](https://bird-bench.github.io/), Jinyang Li et.al.)

### - context: Minimum information required for a human to create a query, such as the description of the necessary columns tables, the names of the features table, how it is distributed the objects using the ids, etc. It also contains parameters that are necessary for the query, such as a possible mjd date that could be requested or oids lists (for now it is only to check for the generated SQL query).

###   - (Note: This is very subjective, so it should be checked)

###   - (Note2: )

### - astro_context: Astronomical information that is not explicited in the request. Features description are included here.

### - used_tables: The minimum tables needed for the gold_query.

### (Note: The purpose of having this information is to check if with the minimum requirements the model is able to generate a correct query.)


