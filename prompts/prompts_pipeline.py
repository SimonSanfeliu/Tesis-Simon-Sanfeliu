### Prompts pipeline
# This file contains the prompts pipeline, with the different prompts to be used in the LLM prompt generation, grouping them by type of prompt and task
# The prompts are defined in the prompts folder, with the different types of prompts, like base prompts, self correction prompts, schema linking prompts, classification prompts, etc.
# The prompts are grouped in dictionaries, with the different parameters and functions to be used in the prompt generation, like the general task description, general context, final instructions, etc.
###


# Functions and prompts
from prompts.base.prompts import base_prompt, prompt_request
# import all prompts and functions from the different types of prompts
from prompts.base.prompts import *
from prompts.decomposition.Decomposition import *
from prompts.correction.SelfCorrection import *
from prompts.schema_linking.SchemaLinking import *
from prompts.classification.Classification import *
from prompts.schema_linking.DBSchema import *

## Base Prompt: 
# Base prompt for the basic pipeline with a direct approach to complete the task and generate the SQL query
##
# Prompt structure
## prompt_func: function to generate the final prompt, considering the other parameters
## general_task: general task description for the LLM to understand the task
## general_context: general context description, with DB information, ALeRCE pipeline information, etc.
## final_instructions: final instructions to focus on specific details like the format of the answer
## schema: Type of schema to be used for the prompt, schema_all_cntxV1 is the most detailed format

base_v4 = {"prompt_func" : base_prompt, "general_task" : general_taskv5, "general_context" : general_contextv5,
           "final_instructions" : final_instructionsv1, "schema": schema_all}
base_v5 = {"prompt_func" : base_prompt, "general_task" : general_taskv5, "general_context" : general_contextv5,
           "final_instructions" : final_instructionsv1, "schema": schema_all_cntxV1}
base_v6 = {"prompt_func" : base_prompt, "general_task" : general_taskv6, "general_context" : general_contextv6,
           "final_instructions" : final_instructionsv1, "schema": schema_all_cntxV1}
base_v7 = {"prompt_func" : base_prompt, "general_task" : general_taskv5, "general_context" : general_contextv5,
           "final_instructions" : final_instructionsv1, "schema": schema_all_indx}
base_v7 = {"prompt_func" : base_prompt, "general_task" : general_taskv5, "general_context" : general_contextv5,
           "final_instructions" : final_instructionsv1, "schema": schema_all_cntxV2}
base_v8 = {"prompt_func" : base_prompt, "general_task" : general_taskv5, "general_context" : general_contextv5,
           "final_instructions" : final_instructionsv1, "schema": schema_all_cntxV1_indx}
base_v15 = {"prompt_func" : base_prompt, "general_task" : general_taskv15, "general_context" : general_contextv15,
           "final_instructions" : final_instructionsv1, "schema": schema_all_cntxV1}
base_vf = {"prompt_func" : base_prompt, "general_task" : general_task_vf, "general_context" : general_context_vf,
            "final_instructions" : final_instructions_vf, "schema": schema_all_cntxV1}
base_v16 = {"prompt_func" : base_prompt, "general_task" : general_taskv16, "general_context" : general_context_vf,
           "final_instructions" : final_instructions_v16, "schema": schema_all_cntxV1_indx}
base_v17 = {"prompt_func" : base_prompt, "general_task" : general_taskv17, "general_context" : general_context_v17,
           "final_instructions" : final_instructions_v17, "schema": schema_all_cntxV1_indx}

## Decomposition Prompt: 
# Decomposition prompt for the advanced pipeline with a decomposed approach to complete the task and generate the SQL query
##
# Prompt structure
## Simple query:
### prompt_func: function to generate the final prompt for the simple queries, considering the other parameters
### general_task: general task description for simple queries
### general_context: general context description for simple queries
### final_instructions: final instructions for simple queries
### schema: Type of schema to be used for the prompt, schema_all_cntxV1 is the most detailed format
## [q] query: [q] = {medium, advanced (adv)}
### [q]_decomp_prompt: prompt for the [q] decomposition task
### [q]_decomp_task: general task description for [q] decomposition task
### [q]_query_cntx: general context description for [q] decomposition task
### [q]_query_instructions_1: final instructions for [q] decomposition task
### [q]_decomp_gen: prompt for the [q] generation task
### [q]_query_task: general task description for [q] generation task
### [q]_query_instructions_2: final instructions for [q] generation task

decomp_v1 = {"prompt_func" : base_prompt, "general_task" : simple_query_task, "general_context" : simple_query_cntx,
             "final_instructions" : simple_query_instructions, "schema":schema_all_cntxV1, "schema_decomp": schema_columns,
             "medium_decomp_prompt": medium_decomp_prompt, "medium_decomp_task": medium_decomp_task, "medium_query_cntx": medium_query_cntx, "medium_query_instructions_1": medium_query_instructions_1,
             "medium_decomp_gen": medium_decomp_gen, "medium_query_task": medium_query_task, "medium_query_instructions_2": medium_query_instructions_2,
             "adv_decomp_prompt": adv_decomp_prompt, "adv_decomp_task": adv_decomp_task, "adv_query_cntx": adv_query_cntx, "adv_query_instructions_1": adv_query_instructions_1, 
             "adv_decomp_gen": adv_decomp_gen, "adv_query_task": adv_query_task, "adv_query_instructions_2": adv_query_instructions_2}

decomp_v2 = {"prompt_func" : base_prompt, "general_task" : simple_query_task, "general_context" : simple_query_cntx,
             "final_instructions" : simple_query_instructions, "schema":schema_all_cntxV1, "schema_decomp": schema_columns,
             "medium_decomp_prompt": medium_decomp_prompt, "medium_decomp_task": medium_decomp_task, "medium_query_cntx": medium_query_cntx, "medium_query_instructions_1": medium_query_instructions_1,
             "medium_decomp_gen": medium_decomp_gen, "medium_query_task": medium_query_task, "medium_query_instructions_2": medium_query_instructions_2,
             "adv_decomp_prompt": adv_decomp_prompt, "adv_decomp_task": adv_decomp_task, "adv_query_cntx": adv_query_cntx, "adv_query_instructions_1": adv_query_instructions_1, 
             "adv_decomp_gen": adv_decomp_gen, "adv_query_task": adv_query_task, "adv_query_instructions_2": adv_query_instructions_2}

decomp_v4 = {"prompt_func" : base_prompt, "general_task" : simple_query_task, "general_context" : simple_query_cntx,
             "final_instructions" : simple_query_instructions, "schema":schema_all_cntxV1, "schema_decomp": schema_all,
             "medium_decomp_prompt": medium_decomp_prompt, "medium_decomp_task": medium_decomp_task_v2 + gpt4turbo1106_decomposed_prompt_2, "medium_query_cntx": medium_query_cntx, "medium_query_instructions_1": medium_query_instructions_1,
             "medium_decomp_gen": medium_decomp_gen, "medium_query_task": medium_query_task, "medium_query_instructions_2": medium_query_instructions_2,
             "adv_decomp_prompt": adv_decomp_prompt, "adv_decomp_task": adv_decomp_task_v2 + gpt4turbo1106_decomposed_prompt_2, "adv_query_cntx": adv_query_cntx, "adv_query_instructions_1": adv_query_instructions_1_v2, 
             "adv_decomp_gen": adv_decomp_gen, "adv_query_task": adv_query_task, "adv_query_instructions_2": adv_query_instructions_2_v2}

decomp_vf = {"prompt_func" : base_prompt, "general_task" : simple_query_task_vf, "general_context" : simple_query_cntx_vf,
             "final_instructions" : simple_query_instructions_vf, "schema":schema_all_cntxV1, "schema_decomp": schema_columns,
             "medium_decomp_prompt": medium_decomp_prompt_vf, "medium_decomp_task": medium_decomp_task_vf, "medium_query_cntx": medium_query_cntx_vf, "medium_query_instructions_1": medium_query_instructions_1_vf,
             "medium_decomp_gen": medium_decomp_gen_vf, "medium_query_task": medium_query_task_vf, "medium_query_instructions_2": medium_query_instructions_2_vf,
             "adv_decomp_prompt": adv_decomp_prompt_vf, "adv_decomp_task": adv_decomp_task_vf, "adv_query_cntx": adv_query_cntx_vf, "adv_query_instructions_1": adv_query_instructions_1_vf, 
             "adv_decomp_gen": adv_decomp_gen_vf, "adv_query_task": adv_query_task_vf, "adv_query_instructions_2": adv_query_instructions_2_vf}


## Self Correction Prompt:
# Self correction prompt for the advanced pipeline with a self correction approach to complete the task and generate the SQL query
##
# Prompt structure
## prompt_func_selfcorr: function to generate the final prompt for the self correction task, considering the other parameters
## general_task_selfcorr: general task description for the self correction task
## general_context_selfcorr: general context description for the self correction task
## final_instructions_selfcorr: final instructions for the self correction task
## schema: Type of schema to be used for the prompt, schema_all_cntxV1 is the most detailed format

self_corr_v1 = {"prompt_func_selfcorr" : prompt_self_correction, "general_task_selfcorr" : general_task_selfcorr_v1, "general_context_selfcorr" : general_context_selfcorr_v1,
                                  "final_instructions_selfcorr" : final_instr_selfcorr_v1, "schema": schema_all}

self_corr_v2 = {"prompt_func_selfcorr" : prompt_self_correction, "general_task_selfcorr" : general_task_selfcorr_v1, "general_context_selfcorr" : general_context_selfcorr_v1,
                                  "final_instructions_selfcorr" : final_instr_selfcorr_v1, "schema": schema_all_cntxV1}

self_corr_v3 = {"prompt_func_selfcorr" : prompt_self_correction_v2, "general_task_selfcorr" : general_task_selfcorr_v1, "general_context_selfcorr" : general_context_selfcorr_v1,
                                  "final_instructions_selfcorr" : final_instr_selfcorr_v1, "schema": schema_all}

self_corr_vf = {"prompt_func_selfcorr" : prompt_self_correction_vf, "general_task_selfcorr" : general_task_selfcorr_vf, "general_context_selfcorr" : None,
                                  "final_instructions_selfcorr" : None, "schema": schema_all}


## Schema Linking Prompt:
# Schema linking prompt for the advanced pipeline with a schema linking approach to define the schema of the tables and columns to be used in the SQL query
##
# Prompt structure
## prompt_func_schlink: function to generate the final prompt for the schema linking task, considering the other parameters
## table_schema_link: table schema linking description
## final_instructions_sclink: final instructions for the schema linking task

schema_linking_v1 = {"prompt_func_schlink" : prompt_schema_linking_v1, "table_schema_link": alerce_tables_desc, "final_instructions_sclink": sl_final_instructions_v1}
schema_linking_v2 = {"prompt_func_schlink" : prompt_schema_linking_v1, "table_schema_link": alerce_tables_desc+"\n"+ref_keys, "final_instructions_sclink": sl_final_instructions_v1}
schema_linking_v3 = {"prompt_func_schlink" : prompt_schema_linking_v1, "table_schema_link": alerce_tables_desc, "final_instructions_sclink": sl_final_instructions_v2}
schema_linking_v4 = {"prompt_func_schlink" : prompt_schema_linking_v2, "h_sch_link": tables_linking_prompt_v4, "table_schema_link": alerce_tables_desc_v2, "final_instructions_sclink": sl_final_instructions_v3}

# Classification prompts
## Difficult Classification Prompt:
# Difficult classification prompt for the advanced pipeline with a classification approach to guide the LLM with a more specific task description and context given the difficulty of the task
##
# Prompt structure
## Simple Classification:
### prompt_func_diff: function to generate the final prompt for the classification task, considering the other parameters
### diff_class_prompt: classification prompt for the difficult classification task
### table_schema_diff: table schema for the difficult classification task
### final_instructions_diff: final instructions for the difficult classification task
## hierarchical prompts:
### h_diff: True if the prompt is hierarchical
### prompt_func_simple: function to generate the final prompt for the simple queries, considering the other parameters
### prompt_func_other: function to generate the final prompt for the other queries, considering the other parameters
### diff_class_prompt_simple: classification prompt for the simple classification task
### table_schema_simple: table schema for the simple classification task
### final_instructions_diff_simple: final instructions for the simple classification task
### diff_class_prompt_other: classification prompt for the other classification task
### table_schema_other: table schema for the other classification task
### final_instructions_diff_other: final instructions for the other classification task
### request: if True, the request is included directly in the 'system' prompt

diff_class_v1 = {"prompt_func_diff" : prompt_diff_class_v1, "diff_class_prompt": diff_class_prompt_v1,
                 "table_schema_diff": schema_all_cntxV1, "final_instructions_diff" : final_instructions_diff_v1}
diff_class_v2 = {"prompt_func_diff" : prompt_diff_class_v1, "diff_class_prompt": diff_class_prompt_v2,
                 "table_schema_diff": schema_all_cntxV1, "final_instructions_diff" : final_instructions_diff_v2}
diff_class_v3 = {"prompt_func_diff" : prompt_diff_class_v1, "diff_class_prompt": diff_class_prompt_v3,
                 "table_schema_diff": schema_all_cntxV1, "final_instructions_diff" : final_instructions_diff_v2}
diff_class_v4 = {"prompt_func_diff" : prompt_diff_class_v1, "diff_class_prompt": diff_class_prompt_v4,
                 "table_schema_diff": schema_all_cntxV1, "final_instructions_diff" : final_instructions_diff_v2}
diff_class_v5 = {"prompt_func_diff" : prompt_diff_class_v2, "diff_class_prompt": diff_class_prompt_v3, "general_task_classification": general_task_classification_v1,
                 "table_schema_diff": schema_all_cntxV1, "final_instructions_diff" : final_instructions_diff_v2}
diff_class_v6 = {"prompt_func_diff" : prompt_diff_class_v3, "diff_class_prompt": diff_class_prompt_v6,
                 "table_schema_diff": schema_all_cntxV1, "final_instructions_diff" : final_instructions_diff_v3,
                 "request": True}
diff_class_v7 = {"prompt_func_diff" : prompt_diff_class_v1, "diff_class_prompt": diff_class_prompt_v7,
                 "table_schema_diff": schema_all_cntxV1, "final_instructions_diff" : final_instructions_diff_v2}
## hierarchical prompts
diff_class_v8 = {"h_diff": True, "prompt_func_simple" : prompt_diff_class_v3, "prompt_func_other" : prompt_diff_class_v3,
                 "diff_class_prompt_simple": diff_class_prompt_v8_1, "table_schema_simple": schema_all_cntxV1, "final_instructions_diff_simple" : final_instructions_diff_simple_v1,
                 "diff_class_prompt_other": diff_class_prompt_v8_2, "table_schema_other": schema_all_cntxV1, "final_instructions_diff_other" : final_instructions_diff_other_v1,
                 "request": True}
diff_class_v9 = {"h_diff": True, "prompt_func_simple" : prompt_diff_class_v1, "prompt_func_other" : prompt_diff_class_v1,
                 "diff_class_prompt_simple": diff_class_prompt_v9_1, "table_schema_simple": schema_columns, "final_instructions_diff_simple" : final_instructions_diff_simple_v2,
                 "diff_class_prompt_other": diff_class_prompt_v9_2, "table_schema_other": schema_columns, "final_instructions_diff_other" : final_instructions_diff_other_v2,}
diff_class_v10 = {"h_diff": True, "prompt_func_simple" : prompt_diff_class_v1, "prompt_func_other" : prompt_diff_class_v1,
                 "diff_class_prompt_simple": diff_class_prompt_v10_1, "table_schema_simple": schema_columns, "final_instructions_diff_simple" : final_instructions_diff_simple_v2,
                 "diff_class_prompt_other": diff_class_prompt_v10_2, "table_schema_other": schema_columns, "final_instructions_diff_other" : final_instructions_diff_other_v2,
                 }

## Advanced Pipeline Prompts:
# Advanced pipeline prompts with a combination of the different steps in the pipeline, like base prompts, self correction prompts, schema linking prompts, classification prompts, etc.
##
# Prompt structure
## generation_type (base or decomposition): Approach to generate the queries
## Schema linking prompts: Version of the schema linking prompt to be used
## Classification prompts: Version of the classification prompt to be used

adv_base_v1 = {**base_v15,  **schema_linking_v1, **diff_class_v1}
adv_base_v2 = {**base_v15,  **schema_linking_v3, **diff_class_v7}
adv_base_v3 = {**base_v5,  **schema_linking_v3, **diff_class_v7}
adv_base_v4 = {**base_vf,  **schema_linking_v3, **diff_class_v9}
adv_base_v5 = {**base_v15,  **schema_linking_v3, **diff_class_v9}
adv_base_v6 = {**base_v16,  **schema_linking_v3, **diff_class_v7}
adv_decomp_v1 = {**decomp_v4,  **schema_linking_v3, **diff_class_v7}
adv_decomp_v2 = {**decomp_v1,  **schema_linking_v3, **diff_class_v7}
adv_decomp_v3 = {**decomp_v1,  **schema_linking_v3, **diff_class_v9}




## Prompt templates

# schema linking prompts
schema_linking_templates = {"schema_linking_v1": schema_linking_v1,
                            "schema_linking_v2": schema_linking_v2,
                            "schema_linking_v3": schema_linking_v3,
                            "schema_linking_v4": schema_linking_v4}

# classification prompts
classification_templates = {"diff_class_v1": diff_class_v1,
                            "diff_class_v2": diff_class_v2,
                            "diff_class_v3": diff_class_v3,
                            "diff_class_v4": diff_class_v4,
                            "diff_class_v5": diff_class_v5,
                            "diff_class_v6": diff_class_v6,
                            "diff_class_v7": diff_class_v7,
                            "diff_class_v8": diff_class_v8,
                            "diff_class_v9": diff_class_v9,
                            "diff_class_v10": diff_class_v10
                            }

# base prompts
base_prompt_templates = {"base_v6": base_v6,
                            "base_v5": base_v5,
                            "base_v7": base_v7,
                            "base_v8": base_v8,
                            "base_v15": base_v15,
                            "base_v16": base_v16,
                            "base_v17": base_v17,
                            "base_vf": base_vf}
# decomp prompts
decomp_prompt_templates = {"decomp_v1": decomp_v1, 
                            "decomp_v2": decomp_v2, 
                            # "decomp_v3": decomp_v3, 
                            "decomp_v4": decomp_v4, 
                            "decomp_vf": decomp_vf}

# self correction prompts
self_corr_prompt_templates = {"self_corr_v1": self_corr_v1,
                              "self_corr_v2": self_corr_v2,
                              "self_corr_v3": self_corr_v3,
                              "self_corr_vf": self_corr_vf,
                              }

# advanced pipeline prompts
adv_prompts_templates = {"adv_base_v1": adv_base_v1,
                            "adv_base_v2": adv_base_v2,
                            "adv_base_v3": adv_base_v3,
                            "adv_base_v4": adv_base_v4,
                            "adv_base_v5": adv_base_v5,
                            "adv_base_v6": adv_base_v6,
                            "adv_decomp_v1": adv_decomp_v1,
                            "adv_decomp_v2": adv_decomp_v2,
                            "adv_decomp_v3": adv_decomp_v3,
                            }

# Join all the prompt templates in a single dictionary
prompts_templates = {**base_prompt_templates, **decomp_prompt_templates, **self_corr_prompt_templates, **schema_linking_templates, **classification_templates, **adv_prompts_templates}
