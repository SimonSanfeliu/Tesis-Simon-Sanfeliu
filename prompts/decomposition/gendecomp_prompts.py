### Description: Generated prompts for SQL decomposition planning using LLMs


# How would you make a decomposition plan to generate a PostgreSQL query?, trying to have the most details possible to replicate and to be understandable
# temperature = 1e-5, top_p=0.1, seed = 817, max_token=1500
gpt4_1106turbo_decomp_promptv1 = '''Creating a decomposition plan to generate a PostgreSQL query involves breaking down the query creation process into smaller, manageable steps. This ensures that each aspect of the query is carefully considered and that the final query is optimized for performance and accuracy. Here's a detailed plan to guide you through the process:

1. **Define the Objective:**
   - Clearly state what you want to achieve with the query (e.g., retrieve data, insert new records, update existing data, delete records).
   - Identify the tables and columns that are relevant to the objective.

2. **Gather Requirements:**
   - Determine the specific data points needed in the result set.
   - Identify any filtering criteria (WHERE conditions).
   - Establish sorting preferences (ORDER BY clause).
   - Decide on grouping requirements (GROUP BY clause) and aggregate functions (e.g., COUNT, SUM, AVG).
   - Consider any joins needed to combine data from multiple tables.
   - Determine if subqueries or common table expressions (CTEs) are necessary.
   - Define any limits on the number of rows returned (LIMIT clause).

3. **Design the Query Structure:**
   - Choose the type of query (SELECT, INSERT, UPDATE, DELETE).
   - Sketch out the basic structure of the query, including the clauses you will need based on the requirements gathered.

4. **Specify Selection Criteria:**
   - Write down the SELECT clause, listing the columns to be retrieved.
   - If using aggregate functions, include them in the SELECT clause with appropriate aliases.

5. **Determine Data Sources:**
   - Write down the FROM clause, specifying the tables from which data will be retrieved.
   - If multiple tables are involved, plan out the type of join (INNER, LEFT, RIGHT, FULL, CROSS) and the joining conditions.

6. **Apply Filters:**
   - Write down the WHERE clause, including all filtering conditions.
   - Use logical operators (AND, OR, NOT) to combine multiple conditions if necessary.

7. **Group Data:**
   - If grouping is required, write down the GROUP BY clause, specifying the columns by which the data should be grouped.
   - Include any HAVING clause to filter groups based on aggregate functions.

8. **Sort Results:**
   - Write down the ORDER BY clause, specifying the columns for sorting and their sort order (ASC for ascending, DESC for descending).

9. **Limit Results:**
   - If a limit on the number of rows is needed, write down the LIMIT clause with the appropriate number.

10. **Optimize Query:**
    - Review the query for any opportunities to optimize performance, such as using indexes or rewriting subqueries.
    - Consider the execution plan to identify potential bottlenecks.

11. **Write the Query:**
    - Combine all the elements from the previous steps into a complete SQL statement.
    - Format the query for readability, using indentation and line breaks.

12. **Test the Query:**
    - Execute the query in a development or staging environment to ensure it works as expected.
    - Validate the results by checking if they meet the initial objective and requirements.

13. **Review and Refine:**
    - Review the query for any syntax errors or logical mistakes.
    - Refine the query if necessary, based on the test results and performance considerations.

14. **Document the Query:**
    - Add comments to the query explaining its purpose, the logic behind certain conditions, and any non-obvious aspects.
    - Document any assumptions or dependencies related to the query.

15. **Deploy the Query:**
    - Once the query is tested and reviewed, deploy it to the production environment if applicable.
    - Monitor the query's performance and impact on the production system.

By following this decomposition plan, you can systematically approach the creation of a PostgreSQL query, ensuring that it is well-structured, efficient, and meets the desired objectives. 
'''


# How would you make a decomposition plan to generate a PostgreSQL query?, trying to have the most details possible to replicate and to be understandable
# temperature = 1e-5, top_p=0.1, seed = 817, max_token=1500
gpt4_1106turbo_decomp_promptv2 = ''' Creating a decomposition plan for generating a PostgreSQL query involves breaking down the process into smaller, manageable steps. This ensures that each aspect of the query is carefully considered and that the final query is optimized for performance and readability. Here's a detailed plan to guide you through the process:

### 1. Define the Objective
- Clearly state what you want to achieve with the query (e.g., retrieve data, insert new records, update existing data, delete records).
- Identify the tables and columns involved.

### 2. Gather Requirements
- Determine the specific data needed (e.g., all columns or specific ones).
- Identify any conditions or filters to apply (e.g., WHERE clauses).
- Decide on the sorting order, if necessary (e.g., ORDER BY clauses).
- Determine if aggregation or grouping is needed (e.g., GROUP BY, HAVING clauses).
- Establish if there are any joins required between tables.

### 3. Design the Query Structure
- Choose the type of query (SELECT, INSERT, UPDATE, DELETE).
- Sketch out the basic structure of the query with placeholders for tables, columns, and conditions.

### 4. Specify Selection Criteria
- Write down the conditions for data retrieval or manipulation (e.g., WHERE `column_name` = 'value').
- Consider using parameterized queries or prepared statements for variable conditions to enhance security.

### 5. Determine Joins and Relationships
- Identify the relationships between tables (e.g., one-to-one, one-to-many, many-to-many).
- Choose the appropriate type of join (INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL OUTER JOIN) based on the relationship and data requirements.

### 6. Incorporate Aggregations and Groupings
- If needed, add aggregate functions (e.g., COUNT, SUM, AVG) to the query.
- Include GROUP BY clauses to group data based on one or more columns.
- Add HAVING clauses to filter groups based on aggregate conditions.

### 7. Add Sorting and Limiting
- Use ORDER BY to sort the results by one or more columns.
- Apply LIMIT and OFFSET to control the number of rows returned or to implement pagination.

### 8. Optimize the Query
- Review the query for any potential performance issues (e.g., missing indexes, expensive joins).
- Use EXPLAIN to analyze the query execution plan and make adjustments as needed.

### 9. Test the Query
- Run the query in a development or staging environment with a dataset similar to production.
- Verify that the results meet the initial objectives and requirements.
- Check for any errors or unexpected behavior and adjust the query accordingly.

### 10. Document the Query
- Add comments within the query to explain complex logic or decisions.
- Document any assumptions, limitations, or important considerations for future reference.

### 11. Implement and Monitor
- Deploy the query to the production environment.
- Monitor its performance and impact on the system.
- Be prepared to make further optimizations if necessary.

### Example Query Creation

Let's say you want to create a query that retrieves the top 10 most recent orders for a specific customer from an e-commerce database. Here's how you might apply the decomposition plan:

1. **Define the Objective**: Retrieve the latest 10 orders for customer ID 123.
2. **Gather Requirements**: Need order ID, order date, and total amount.
3. **Design the Query Structure**: It will be a SELECT query.
4. **Specify Selection Criteria**: WHERE `customer_id` = 123.
5. **Determine Joins and Relationships**: No joins needed if all data is in the orders table.
6. **Incorporate Aggregations and Groupings**: No aggregations or groupings needed.
7. **Add Sorting and Limiting**: ORDER BY `order_date` DESC, LIMIT 10.
8. **Optimize the Query**: Ensure there's an index on `customer_id` and `order_date`.
9. **Test the Query**: Run the query in a non-production environment.
10. **Document the Query**: Add comments explaining the purpose and any nuances.
11. **Implement and Monitor**: Deploy to production and monitor performance.

The resulting query might look like this:

```sql
-- Retrieves the 10 most recent orders for customer 123
SELECT order_id, order_date, total_amount
FROM orders
WHERE customer_id = 123
ORDER BY order_date DESC
LIMIT 10;
```

By following this decomposition plan, you can systematically approach the creation of PostgreSQL queries, ensuring they are well-structured, optimized, and meet the required objectives. 
'''

# How would you make a decomposition plan to generate a PostgreSQL query to retrieve information from a astronomy broker?, trying to have the most details possible to replicate and to be understandable
# temperature = 1e-5, top_p=0.1, seed = 817, max_token=1500
gpt4_1106turbo_decomp_promptv3 = '''Creating a decomposition plan to generate a PostgreSQL query for retrieving information from an astronomy broker involves breaking down the task into smaller, manageable steps. Here's a detailed plan to guide you through the process:

### 1. Understand the Requirements
- Identify the specific information you need to retrieve (e.g., star names, coordinates, magnitudes, spectral types).
- Determine any filters or conditions (e.g., stars brighter than a certain magnitude, within a specific constellation).
- Decide on the sorting and grouping of the results (e.g., order by magnitude, group by spectral type).

### 2. Familiarize with the Database Schema
- Obtain the database schema, which includes table names, column details, and relationships between tables.
- Identify the relevant tables and columns that contain the required information.
- Understand any foreign key relationships if you need to join tables.

### 3. Establish the Query Structure
- Choose the type of query (e.g., SELECT, JOIN) based on the information needed.
- Determine if subqueries or common table expressions (CTEs) are necessary for complex data retrieval.

### 4. Write the Query
#### 4.1. Select the Columns
- Write the SELECT clause with the specific columns you need.
- Use aliases for columns if needed for clarity.

#### 4.2. Specify the Tables
- Write the FROM clause with the relevant table(s).
- If joining tables, specify the type of JOIN (INNER, LEFT, RIGHT, FULL) and the condition for the join.

#### 4.3. Filter the Results
- Write the WHERE clause to filter the data according to the requirements.
- Use logical operators (AND, OR, NOT) and comparison operators (=, <, >, BETWEEN, IN, etc.) as needed.

#### 4.4. Group the Results
- If required, use the GROUP BY clause to aggregate data.
- Use aggregate functions (COUNT, SUM, AVG, MAX, MIN) if necessary.

#### 4.5. Sort the Results
- Write the ORDER BY clause to sort the results by one or more columns.
- Specify the sort direction (ASC for ascending, DESC for descending).

#### 4.6. Limit the Results
- If you need to limit the number of results returned, use the LIMIT clause.

### 5. Test the Query
- Run the query in a PostgreSQL client or through a script.
- Verify that the results match the requirements.
- Check for any errors and optimize the query for performance if necessary.

### 6. Document the Query
- Comment the query to explain the purpose, the choice of tables and columns, and any complex logic.
- Include the version of the PostgreSQL and any relevant information about the database environment.

### 7. Review and Refine
- Review the query for any potential improvements or simplifications.
- Test the query with different datasets to ensure it works under various conditions.

### Example Query
Here's an example of what the final query might look like, assuming we want to retrieve the names and coordinates of stars brighter than magnitude 5 in the constellation Orion:

```sql
-- Retrieve star names and coordinates for stars brighter than magnitude 5 in the constellation Orion
SELECT
    s.name AS star_name,
    s.ra AS right_ascension,
    s.dec AS declination
FROM
    stars AS s
JOIN
    constellations AS c ON s.constellation_id = c.id
WHERE
    s.magnitude < 5
    AND c.name = 'Orion'
ORDER BY
    s.magnitude ASC;
```

Remember, the actual query will depend on the specific database schema and the information you need to retrieve. Always test your queries to ensure they return the expected results and perform well. 
'''

# How would you make a decomposition plan to generate a PostgreSQL query to retrieve information from the ALeRCE astronomy broker database?, trying to have the most details possible to replicate and to be understandable
# temperature = 1e-5, top_p=0.1, seed = 817, max_token=1500
gpt4_1106turbo_decomp_promptv4 = '''Creating a decomposition plan to generate a PostgreSQL query for retrieving information from the ALeRCE astronomy broker database involves several steps. ALeRCE (Automatic Learning for the Rapid Classification of Events) is a system designed to classify large amounts of astronomical data, typically from surveys like the Zwicky Transient Facility (ZTF). To create a detailed and understandable plan, follow these steps:

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

5. **Optimize the Query:**
   - Consider indexing columns that are frequently used in `JOIN`, `WHERE`, or `ORDER BY` clauses to improve performance.
   - Review the query to ensure it is not returning unnecessary data (e.g., using `SELECT *` when only specific columns are needed).

6. **Write the Query:**
   - Translate the design into SQL syntax.
   - Use aliases for tables and columns to make the query more readable.
   - Format the query to enhance readability, with proper indentation and line breaks.

7. **Test the Query:**
   - Run the query on a subset of the data to ensure it executes correctly and returns the expected results.
   - Validate the results with known data points to ensure accuracy.

8. **Iterate and Refine:**
   - If the initial results are not satisfactory, refine the selection criteria, joins, or aggregation methods.
   - Optimize the query further if performance is an issue.

9. **Document the Query:**
   - Write comments within the SQL code to explain the purpose of different parts of the query.
   - Create external documentation that describes the query's purpose, the information it retrieves, and any assumptions or limitations.

10. **Prepare for Production:**
    - If the query will be used frequently, consider creating a view or stored procedure.
    - Ensure that the query adheres to any security and privacy standards required by the database or organization.

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



# How would you make a decomposition plan to generate a PostgreSQL query to retrieve information from the ALeRCE astronomy broker database?, trying to have the most details possible to replicate and to be understandable
# temperature = 1e-5, top_p=0.1, seed = 817, max_token=1500
gpt4_0125turbo_decomp_promptv1 = '''Creating a decomposition plan to generate a PostgreSQL query for retrieving information from the ALeRCE astronomy broker database involves breaking down the task into smaller, manageable steps. This approach ensures that each aspect of the query creation process is addressed, leading to a comprehensive and efficient retrieval of data. Here's a detailed plan:

### 1. Understand the Database Schema
- **1.1. Identify Available Tables:** Understand the tables available in the ALeRCE database, focusing on those relevant to your query (e.g., objects, detections, classifications).
- **1.2. Understand Table Relationships:** Determine how tables are related to each other (e.g., foreign keys, one-to-many relationships).
- **1.3. Identify Relevant Columns:** For each table involved in your query, identify the columns that contain the data you need.

### 2. Define the Query Requirements
- **2.1. Specify Information Needs:** Clearly define the information you want to retrieve (e.g., object IDs, magnitudes, classifications).
- **2.2. Determine Conditions:** Specify any conditions that the data must meet (e.g., date ranges, object types).
- **2.3. Decide on Sorting/Grouping:** Determine if the results need to be sorted or grouped in a specific way.

### 3. Design the Query
- **3.1. Select Statement:** Start with the `SELECT` statement, specifying the columns you need based on your requirements.
- **3.2. From Clause:** Use the `FROM` clause to specify the primary table from which to retrieve the data.
- **3.3. Join Conditions:** If you need data from multiple tables, use `JOIN` clauses to specify how the tables are related.
- **3.4. Where Clause:** Use the `WHERE` clause to filter the data based on your specified conditions.
- **3.5. Group By and Having Clauses:** If needed, use `GROUP BY` to aggregate data and `HAVING` to filter aggregated data.
- **3.6. Order By Clause:** Use the `ORDER BY` clause to sort the results.

### 4. Optimize the Query
- **4.1. Index Usage:** Ensure that indexes are used where appropriate to speed up query execution.
- **4.2. Query Performance:** Test the query and analyze its performance; consider using `EXPLAIN` to understand the query plan.
- **4.3. Refinement:** Refine the query as needed to improve performance or accuracy.

### 5. Execute the Query
- **5.1. Use a PostgreSQL Client:** Use a PostgreSQL client (e.g., pgAdmin, psql) to connect to the ALeRCE database.
- **5.2. Run the Query:** Execute your query and review the results.
- **5.3. Adjust as Needed:** If the results are not as expected, adjust your query and re-execute.

### 6. Document the Query
- **6.1. Comment Your SQL:** Add comments to your SQL query explaining the purpose of each part of the query.
- **6.2. Documentation:** Document the query purpose, execution steps, and any assumptions or decisions made during the query design process.

### 7. Example
Here's a simple example to illustrate part of the process:

```sql
-- Retrieve object IDs and their latest magnitude for objects classified as supernovae
SELECT o.object_id, d.magnitude
FROM objects o
JOIN detections d ON o.object_id = d.object_id
WHERE o.classification = 'supernova'
AND d.detection_date = (SELECT MAX(detection_date) FROM detections WHERE object_id = o.object_id)
ORDER BY d.magnitude DESC;
```

This example assumes the existence of `objects` and `detections` tables with the specified columns and relationships. It's a starting point and would need to be adjusted based on the actual database schema and specific requirements.

By following this decomposition plan, you can systematically approach the task of generating a PostgreSQL query to retrieve information from the ALeRCE astronomy broker database, ensuring that all aspects of the query creation process are covered. 
'''




# How would you make a decomposition plan to generate a PostgreSQL query to retrieve information from the ALeRCE astronomy broker database?, trying to have the most details possible to replicate and to be understandable
# temperature = 1e-5, top_p=0.1, seed = 817, max_token=2000
gpt4_0613_decomp_promptv1 = '''1. Understand the Database Structure: The first step in creating a PostgreSQL query is to understand the structure of the ALeRCE astronomy broker database. This includes understanding the tables, their relationships, and the fields within each table. 

2. Define the Information Needed: Clearly define the information you need to retrieve from the database. This could be specific fields from one or more tables. 

3. Design the Query: Based on the information needed, design the query. This could be a simple SELECT statement if you're retrieving data from a single table, or it could involve JOINs if you're retrieving data from multiple tables. 

4. Implement the Query: Write the SQL query in PostgreSQL syntax. Test the query to ensure it retrieves the correct data. 

5. Optimize the Query: If the query is slow or inefficient, optimize it. This could involve adding indexes to the database, rewriting the query, or changing the database structure. 

6. Document the Query: Write clear, concise documentation for the query. This should include a description of what the query does, the tables and fields it accesses, and any assumptions or limitations. 

7. Share the Query: Share the query and its documentation with others who need to use it. This could involve adding the query to a shared repository, emailing it to colleagues, or including it in a report. 

8. Maintain the Query: As the database changes over time, the query may need to be updated. Regularly review the query to ensure it continues to function correctly. 

Here is a more detailed breakdown of the steps:

1. Understand the Database Structure:
   1.1. Identify the tables in the database.
   1.2. Understand the relationships between the tables.
   1.3. Identify the fields within each table.

2. Define the Information Needed:
   2.1. Identify the specific fields you need to retrieve.
   2.2. Determine whether you need to retrieve data from one table or multiple tables.

3. Design the Query:
   3.1. If you're retrieving data from a single table, design a simple SELECT statement.
   3.2. If you're retrieving data from multiple tables, design a query that uses JOINs.

4. Implement the Query:
   4.1. Write the SQL query in PostgreSQL syntax.
   4.2. Test the query to ensure it retrieves the correct data.

5. Optimize the Query:
   5.1. If the query is slow, consider adding indexes to the database.
   5.2. If the query is inefficient, consider rewriting it or changing the database structure.

6. Document the Query:
   6.1. Write a description of what the query does.
   6.2. Document the tables and fields the query accesses.
   6.3. Document any assumptions or limitations of the query.

7. Share the Query:
   7.1. Add the query to a shared repository.
   7.2. Email the query to colleagues.
   7.3. Include the query in a report.

8. Maintain the Query:
   8.1. Regularly review the query to ensure it continues to function correctly.
   8.2. Update the query as the database changes. '''



# How would you make a decomposition plan to generate a PostgreSQL query to retrieve information from the ALeRCE astronomy broker database?, trying to have the most details possible to replicate and to be understandable
# temperature = 1e-5, top_p=default, seed = 817, max_token=1500
gpt4_0613_decomp_promptv2 = '''1. Understand the Database Structure: The first step in creating a PostgreSQL query is to understand the structure of the ALeRCE astronomy broker database. This includes understanding the tables, their relationships, and the fields within each table. 

2. Identify the Information Needed: Once you understand the database structure, you need to identify the specific information you want to retrieve. This could be anything from a list of all objects observed on a specific date, to the details of a specific astronomical event.

3. Design the Query: With the information needed in mind, you can start designing your query. This will involve selecting the appropriate tables and fields, and specifying any conditions or filters. 

    a. Select the Tables: Identify which tables contain the information you need. If the information is spread across multiple tables, you'll need to join them together.

    b. Select the Fields: Identify which fields within those tables contain the information you need. These will be the fields you select in your query.

    c. Specify Conditions: If you only want to retrieve certain records, you'll need to specify conditions using the WHERE clause. For example, you might only want records where the observation date is after a certain date.

    d. Specify Filters: If you want to exclude certain records, you can specify filters using the NOT or EXCEPT operators.

4. Write the Query: Once you've designed your query, you can write it out in SQL. Be sure to use the correct syntax for PostgreSQL.

5. Test the Query: After writing your query, you should test it to make sure it returns the expected results. If it doesn't, you'll need to debug and revise your query.

6. Document the Query: Finally, document your query so that others can understand what it does and how to use it. This should include a description of the query, the tables and fields it uses, and any conditions or filters it applies.

7. Optimize the Query: Depending on the complexity of your query and the size of your database, you might need to optimize your query to improve its performance. This could involve indexing certain fields, rewriting your query to avoid full table scans, or using more efficient join algorithms. 

Remember, the goal is to retrieve the necessary information as efficiently as possible, while also making your query easy to understand and replicate. '''