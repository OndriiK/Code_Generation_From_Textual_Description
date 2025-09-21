TASK_INSTRUECTION2="""
Given the following GitHub problem description, your objective is to localize the specific files, classes or functions, and lines of code that need modification or contain key information to resolve the issue.

Follow these steps to localize the issue:
## Step 1: Categorize and Extract Key Problem Information
 - Classify the problem statement into the following categories:
    Problem description, error trace, code to reproduce the bug, and additional context.
 - Use extracted keywords and line numbers to search for relevant code references for additional context.

## Step 2: Locate Referenced Modules
- Accurately determine specific modules
    - Explore the repo to familiarize yourself with its structure.
    - Analyze the described execution flow to identify specific modules or components being referenced.
- Pay special attention to distinguishing between modules with similar names using context and described execution flow.

## Step 3: Analyze and Reproducing the Problem
- Clarify the Purpose of the Issue
    - If expanding capabilities: Identify where and how to incorporate new behavior, fields, or modules.
    - If addressing unexpected behavior: Focus on localizing modules containing potential bugs.
- Reconstruct the execution flow
    - Identify main entry points triggering the issue.
    - Trace function calls, class interactions, and sequences of events.
    - Identify potential breakpoints causing the issue.
    Important: Keep the reconstructed flow focused on the problem, avoiding irrelevant details.

## Step 4: Locate Areas for Modification
- Locate specific files, functions, or lines of code requiring changes or containing critical information for resolving the issue.
- Consider upstream and downstream dependencies that may affect or be affected by the issue.
- Try to include function or class names in your findings.
- Think Thoroughly: List multiple potential solutions and consider edge cases that could impact the resolution.

## Output Format for Final Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, or line numbers, ordered by importance.
Your answer would better include about 5 files.

Enclose your Final Results message inside a "<finish>...</finish>" block.

USE THIS FORMAT IF YOU NEED TO EXECUTE A CODE SEARCH TO GET CONTEXT FROM THE CODEBASE: <execute_ipython>…</execute_ipython>

### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: my_function1

full_path2/file2.py
line: 76
function: MyClass2.my_function2

full_path3/file3.py
line: 24
line: 156
function: my_function3
```

IMPORTANT!!
- Do not reference any files, functions or modules that are not in the given codebase. Work only with the existing files!
- If the file you are proposing changes for is not in the provided codebase, DON'T OUTPUT IT!
- Do not reference any files, functions or modules that are not in the given codebase. Work only with the existing files!

Return just the location(s)

DO NOT try to execute any of the observed code using the <execute_ipython> tags. Use them ONLY for searching code snippets.

DO NOT ADD ANY ADDITIONAL TEXT TO YOUR FINAL ANSWER!

Begin!
"""

TASK_INSTRUECTION="""
Given the following developer's query, your objective is to localize the specific files, classes or functions, and optionally lines of code that need modification or contain key information to resolve the issue.
The information you provide will be subsequently used by a planning module that will generate a structured actionable plan of completing the objective.

Follow these steps to localize the issue:
## Step 1: Categorize and Extract Key Problem Information
 - Decompose the problem into multiple subproblems if necessary
 - Think about possible dependencies
 - Reason over if new code will be needed

## Step 2: Locate Referenced Files or Modules
- Accurately determine specific modules or files
    - Explore the repo to familiarize yourself with its structure.
- Pay special attention to distinguishing between modules with similar names using context and described execution flow.
- If files, modules, functions or classes are mentioned in the query, they are most likely important

## Step 3: Identify and Analyze the Problem
- Clarify the Purpose of the Issue
    - If expanding capabilities: Identify where and how to incorporate new behavior, fields, or modules.
    - If addressing unexpected behavior: Focus on localizing modules containing potential bugs.
- Reconstruct the execution flow
    - Identify main entry points triggering the issue.
    - Trace function calls, class interactions, and sequences of events.
    - Identify potential breakpoints causing the issue.
    Important: Keep the reconstructed flow focused on the problem, avoiding irrelevant details.

## Step 4: Locate Areas for Modification
- Locate specific files, functions, or lines of code requiring changes or containing critical information for resolving the issue.
- Consider upstream and downstream dependencies that may affect or be affected by the issue.
- Think Thoroughly: List multiple potential solutions and consider edge cases that could impact the resolution.

## Output Format for Final Results:
Your final output should list the locations requiring modification, wrapped with triple backticks ```
Each location should include the file path, class name (if applicable), function name, (optionally) line numbers, ordered by importance.
Your answer would better include about 5 locations but that depends on the complexity of the query.

!!!! If you have already seen the referenced segments, PROCEED TO OUTPUTTING THE FINAL RESULTS. DO NOT REPEATEDLY SEARCH FOR THE SAME CODE (even general analysis suffices) !!!!
!!!! If you have already seen the referenced segments, PROCEED TO OUTPUTTING THE FINAL RESULTS. DO NOT REPEATEDLY SEARCH FOR THE SAME CODE (even general analysis suffices) !!!!

Enclose your Final Results message inside a "<finish>...</finish>" block.

Use this format if you need to execute a code search for more context from the codebase: <execute_ipython>…</execute_ipython>

### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
function: my_function1

full_path2/file2.py
line: 76
function: MyClass2.my_function2

full_path3/file3.py
line: 24
line: 156
function: my_function3
```

IMPORTANT!!
- Do not reference any files, functions or modules that are not in the given codebase. Work only with the existing files!
- If the file you are proposing changes for is not in the provided codebase, DON'T OUTPUT IT!
- Do not reference any files, functions or modules that are not in the given codebase. Work only with the existing files!

Return just the location(s)

Begin!
"""

FAKE_USER_MSG_FOR_LOC = (
    'Verify if the found locations contain all the necessary information to address the issue, and check for any relevant references in other parts of the codebase that may not have appeared in the search results. '
    'If not, continue searching for additional locations related to the issue.\n'
    'Verify that you have carefully analyzed the impact of the found locations on the repository, especially their dependencies. '
    'If you think you have solved the task, please send your final answer (including the former answer and reranking) to user through message and then call `finish` to finish.\n'
    'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
)