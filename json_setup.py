import json

from pipeline.process import *
from prompts.base.prompts import *

# Prompt dictionary guideline and used by Jorge
prompts = {
    "Schema Linking": {
        "base_prompt": tables_linking_prompt_V2 + q3c_info,
        "context1": schema_all_cntxV1,
        "context2": schema_all_cntxV2_indx,
        "context3": schema_all_cntxV2,
    },
    "Classify": {
        "base_prompt": diff_class_prompt_v7,
        "final_instructions": final_instructions_diff_v2
    },
    "Decomposition": {
        "simple": {
            "query_task": simple_query_task_v2,
            "query_context": simple_query_cntx + q3c_info,
            "external_knowledge": "placeholder",
            "domain_knowledge": "placeholder",
            "query_instructions": simple_query_instructions_v2
        },
        "medium": {
            "decomp_plan": {
                "base_prompt": medium_decomp_prompt,
                "decomp_task": medium_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2,
                "decomp_task_python": medium_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2_python,
                "query_context": medium_query_cntx + q3c_info,
                "query_instructions": medium_query_instructions_1_v2
            },
            "decomp_gen": {
                "sql": {
                    "base_prompt": medium_decomp_gen,
                    "query_task": medium_query_task_v2,
                    "query_instructions": medium_query_instructions_2_v2,
                },
                "python": {
                    "base_prompt": medium_decomp_gen_vf_python,
                    "query_task": medium_query_task_v2,
                    "query_instructions": medium_query_instructions_2_v2_python,
                }
            }
        },
        "advanced": {
            "decomp_plan": {
                "base_prompt": adv_decomp_prompt,
                "decomp_task": adv_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2,
                "decomp_task_python": adv_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2_python,
                "query_context": adv_query_cntx + q3c_info,
                "query_instructions": adv_query_instructions_1_v3
            },
            "decomp_gen": {
                "sql": {
                    "base_prompt": adv_decomp_gen,
                    "query_task": adv_query_task_v2,
                    "query_instructions": adv_query_instructions_2_v3,
                },
                "python": {
                    "base_prompt": adv_decomp_gen_vf_python,
                    "query_task": adv_query_task_v2,
                    "query_instructions": adv_query_instructions_2_v3_python,
                }
            }
        }
    },
    "Direct": {
        "base_prompt": {
            "general_task": general_taskv18,
            "general_context": general_contextv15 + q3c_info,
            "final_instructions": final_instructions_v19
        },
        "request_prompt": {
            "external_knowledge": "placeholder",
            "domain_knowledge": "placeholder"
        }
    }
}

# Write the dictionary to a JSON file
with open("final_prompts/prompts_v4.json", "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)