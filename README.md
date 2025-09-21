# CogniCollab: A Multi-Agent AI Assistant for Software Development

CogniCollab is a modular multi-agent Artificial Intelligence assistant system designed to streamline the software development process. By leveraging Large Language Models (LLMs) and an agentic architecture, it aims to automate routine coding tasks such as writing unit tests, generating documentation, fixing bugs, and adding simple features, freeing up developer time for higher-level tasks. The system enhances context awareness within existing codebases through integration with the LocAgent tool for semantic code search and analysis.

## Prerequisites

Before setting up and running CogniCollab, ensure you have the following installed:

*   **Python:** Version 3.12 (specifically recommended for LocAgent compatibility).
*   **Miniconda or Anaconda:** Recommended for managing the system environment and dependencies.
*   **API Key(s):** An API key or keys for an LLM provider compatible with `litellm` (e.g., OpenAI, Anyscale, OpenRouter). You will need to set these as environment variables. CogniCollab supports two modes for API key configuration:
    *   **Single API Key Mode:** Use one API key for all agents. Set the environment variable `API_KEY` to your key.
    *   **Separate API Key Mode:** Use different API keys for specific agents. Set the following environment variables with the corresponding keys:
        *   `ANALYST_AGENT_API_KEY`
        *   `IMPLEMENTATION_AGENT_API_KEY`
        *   `DEBUG_AGENT_API_KEY`
        *   `TEST_AGENT_API_KEY`
        *   `DOCUMENT_AGENT_API_KEY`
    You can select which mode to use via a toggle in the Streamlit UI.

## System Initialization and Usage

Follow these steps to set up and run CogniCollab:

1.  **Set up the LocAgent Conda environment:**
    CogniCollab utilizes LocAgent, which requires a specific Python environment.
    ```bash
    conda create -n cognicollab python=3.12
    conda activate cognicollab
    ```

2.  **Install dependencies:**
    While in the `cognicollab` environment, install the required Python packages stored in `requirements.txt`
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set your LLM API key(s):**
    Set the necessary environment variable(s) according to the API key configuration mode you plan to use. For example, for the single API key mode:
    ```bash
    export API_KEY="YOUR_API_KEY"
    ```
    Or for the separate API key mode, set the relevant individual agent keys. These are all variables that need to be set in the separate API key mode: ANALYST_AGENT_API_KEY, LOCAGENT_API_KEY, IMPLEMENTATION_AGENT_API_KEY, DEBUG_AGENT_API_KEY, TEST_AGENT_API_KEY, DOCUMENT_AGENT_API_KEY.

4.  **Run the Streamlit application:**
    The main application interface is built with Streamlit.
    ```bash
    streamlit run app.py
    ```
    This will open the CogniCollab UI locally in your web browser.

5.  **Interact with the system:**
    In the UI, you can input your natural language query/task, select a local codebase folder to analyze (the system handles the LocAgent indexing and search automatically when needed). Before processing, use the **"Use Separate API Keys" toggle** in the UI to select whether to use the single `API_KEY` or the individual agent API keys. Then, initiate the process.

## Agent Overview

CogniCollab employs a multi-agent architecture where different components and agents collaborate to fulfill the user's request.

*   **Analyst Agent (AA):** Acts as the central coordinator and team leader. It receives the user's natural language query, determines the operational mode (Clarification or Completion), utilizes the LocAgent tool for codebase analysis, decomposes complex tasks into smaller sub-tasks, plans the execution flow, and aggregates the final results from the specialized agents.
    *   **Clarification Mode:** Handles user questions about existing code functionality, providing explanations and context by leveraging LocAgent's search capabilities.
    *   **Completion Mode:** Handles development tasks requiring code modifications, decomposition, and delegation to specialized agents.

        *(Note: The Analyst Agent determines the operational mode autonomously)*

*   **Intent Classifier:** A fine-tuned BERT-based model that analyzes the user query to identify the intended task type (e.g., `write_tests`, `add_feature`, `debug_code`, `document_code`). This informs the Analyst Agent which specialized agents will likely be needed.
*   **Specialized Agents:** A team of agents each dedicated to a specific type of development task. They receive sub-tasks from the Analyst Agent and work iteratively using their embedded LLM and potentially LocAgent's search functions to complete them.
    *   **Implementation Agent:** Responsible for adding new features or modifying existing code.
    *   **Debugging Agent:** Focuses on localizing and resolving coding errors.
    *   **Testing Agent:** Generates unit tests and test cases.
    *   **Documentation Agent:** Creates or updates code documentation.
*   **Communication Leader (CL):** Manages the flow of messages and tasks between the Analyst Agent and the Specialized Agents using a shared memory structure. It routes tasks to agent-specific queues and handles task dependencies. It is not an LLM-powered agent itself.
*   **Shared Memory:** A central data structure used by the Communication Leader to manage messages, task assignments, and results between agents. Includes a mechanism for tracking sub-task completion status and dependencies.
*   **LocAgent (External Tool):** While not an agent within the CogniCollab agent team, this external tool is integrated to provide deep codebase understanding. It creates a graph representation of the codebase and performs semantic searches, providing crucial context to the Analyst Agent and specialized agents.

The interaction typically starts with the developer providing a natural language query via the Streamlit UI. The Intent Classifier and Analyst Agent process the query, potentially leveraging LocAgent for context. In Completion mode, the Analyst Agent decomposes the task and delegates sub-tasks to the relevant Specialized Agents via the Communication Leader and Shared Memory. Specialized Agents complete their tasks iteratively, reporting results back via Shared Memory, which are then aggregated by the Analyst Agent into the final response presented to the user.

---
