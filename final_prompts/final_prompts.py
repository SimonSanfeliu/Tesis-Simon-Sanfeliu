### Schema linking prompt

with open("final_prompts\schema_linking\sch_linking.txt", "r") as f:
  sch_linking = f.read()


### Decomposition prompt

## Medium

with open("final_prompts\decomposition\decomp_medium.txt", "r") as f:
  decomp_medium = f.read()

## Advanced

with open("final_prompts\decomposition\decomp_advanced.txt", "r") as f:
  decomp_advanced = f.read()


### Query generation prompt

## SQL

# Simple

with open("final_prompts\query_generation\sql\query_sql_simple.txt", "r") as f:
  query_sql_simple = f.read()

# Medium

with open("final_prompts\query_generation\sql\query_sql_medium.txt", "r") as f:
  query_sql_medium = f.read()

# Advanced

with open("final_prompts\query_generation\sql\query_sql_advanced.txt", "r") as f:
  query_sql_advanced = f.read()

## Python

# Medium

with open("final_prompts\query_generation\python\query_python_medium.txt", "r") as f:
  query_python_medium = f.read()

# Advanced

with open("final_prompts\query_generation\python\query_python_advanced.txt", "r") as f:
  query_python_advanced = f.read()
