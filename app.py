import streamlit as st
import os
import subprocess
import multiprocessing
import queue
import json
import re
import time
from datetime import datetime
import atexit
import signal
import shutil

from workflows.shared_mem_com import (
    SharedMemory,
    CommunicationLeader,
    Agent,
)
from analyst_agent.analyst_agent import AnalystAgent
from prompts.prompts import (
    IMPLEMENTATION_TASK_INSTRUCTION,
    DEBUG_TASK_INSTRUCTION,
    TEST_TASK_INSTRUCTION,
    DOCUMENT_TASK_INSTRUCTION,
)

# LocAgent's function to set/reset global variables for a specific issue/user query
from LocAgent.plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue
)


# function to extract JSON structured plan from the Analyst Agent's output
def extract_json_plan(raw_output):
    """
    Extracts the JSON plan from the analyst agent's raw output.
    Returns the plan as a list, or None if extraction fails or no plan is found.
    """
    if not raw_output or not isinstance(raw_output, str):
        print("Error: Analyst Agent did not return a valid plan string.")
        return None
    json_pattern = r'\[\s*\{.*?\}\s*\]'
    match = re.search(json_pattern, raw_output, re.DOTALL | re.MULTILINE)

    if match:
        json_str = match.group(0)
        try:
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                return parsed_json
            else:
                print(f"DEBUG: Parsed JSON is not a list:\n{parsed_json}")
                return None
        except json.JSONDecodeError as e:
            print(f"DEBUG: Error parsing JSON plan: {e}")
            return None
    else:
        print(f"DEBUG: No valid JSON array found in the Analyst output")
        return None


# --- Agent Configuration (Global in main process, passed to child process) ---
# API key environment variable names
SINGLE_API_KEY_ENV_VAR = 'API_KEY'
SEPARATE_API_KEY_ENV_VARS = {
    "analyst_agent": "ANALYST_AGENT_API_KEY",
    "implementation_agent": "IMPLEMENTATION_AGENT_API_KEY",
    "debug_agent": "DEBUG_AGENT_API_KEY",
    "test_agent": "TEST_AGENT_API_KEY",
    "document_agent": "DOCUMENT_AGENT_API_KEY",
}

# Configuration using a single API key for all agents
single_api_key_config = {
    "analyst_agent": os.getenv(SINGLE_API_KEY_ENV_VAR),
    "implementation_agent": os.getenv(SINGLE_API_KEY_ENV_VAR),
    "debug_agent": os.getenv(SINGLE_API_KEY_ENV_VAR),
    "test_agent": os.getenv(SINGLE_API_KEY_ENV_VAR),
    "document_agent": os.getenv(SINGLE_API_KEY_ENV_VAR),
}

# Configuration using separate API keys for each agent
separate_api_keys_config = {
    name: os.getenv(env_var_name)
    for name, env_var_name in SEPARATE_API_KEY_ENV_VARS.items()
}


# Model names for each agent type
model_names = {
    "analyst_agent": "openrouter/qwen/qwen-2.5-72b-instruct:free",
    "implementation_agent": "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
    "debug_agent": "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
    "test_agent": "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
    "document_agent": "openrouter/qwen/qwen-2.5-coder-32b-instruct:free",
}

# User prompts for each agent type
agent_user_prompts = {
    "implementation_agent": "user_prompt_implementation_agent",
    "debug_agent": "user_prompt_debug_agent",
    "test_agent": "user_prompt_test_agent",
    "document_agent": "user_prompt_document_agent",
}

# Specialized agent prompts for each task type
specialized_agent_prompts = {
    "implementation_agent": IMPLEMENTATION_TASK_INSTRUCTION,
    "debug_agent": DEBUG_TASK_INSTRUCTION,
    "test_agent": TEST_TASK_INSTRUCTION,
    "document_agent": DOCUMENT_TASK_INSTRUCTION,
}


# Main Agent Workflow Function. This function is run in a separate subprocess.
def run_agent_system(
    query: str,
    repo_path: str,
    result_mp_queue: multiprocessing.Queue, # multiprocessing queue to obtain the final result from the communication leader
    api_keys_config: dict,
    model_names_config: dict,
    agent_user_prompts_config: dict,
    specialized_agent_prompts_config: dict,
    separate_api_key_mode: bool,
):
    """Initializes and runs the full agent system in a separate process."""
    instance_id = f"agent_run_{int(time.time())}"
    leader_thread = None
    agent_threads = [] # To store agent thread instances
    process_shutdown_event = multiprocessing.Event()


    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"[{instance_id}] Process received signal {signum}. Initiating shutdown...")
        process_shutdown_event.set()
        # Also trigger internal shutdown mechanisms of threads
        if leader_thread and hasattr(leader_thread, 'shutdown'):
            leader_thread.shutdown()
        for ag_thread in agent_threads:
            if hasattr(ag_thread, 'shutdown'):
                ag_thread.shutdown()

    # Register signal handler for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[{instance_id}] Process started.")

    try:
        if process_shutdown_event.is_set(): return

        instance_data_dict = {
            "instance_id": instance_id,
            "problem_statement": query,
            "repo": repo_path,
            "base_commit": "latest",
            "patch": ""
        }

        # this sets the global variables in LocAgent for the selected codebase
        set_current_issue(instance_data=instance_data_dict)

        # Initialize shared memory structure and the analyst agent
        shared_memory = SharedMemory()
        analyst_agent = AnalystAgent(
            repo_path=repo_path,
            shared_memory=shared_memory,
            api_key=api_keys_config["analyst_agent"],
            model_name=model_names_config["analyst_agent"],
            separate_api_key_mode=separate_api_key_mode
        )

        local_agents = []
        specialized_agent_names = [
            "implementation_agent", "debug_agent", "test_agent", "document_agent"
        ]
        # instantiate specialized agents
        for name in specialized_agent_names:
            if name not in api_keys_config or not api_keys_config[name]:
                raise ValueError(f"API Key for {name} not found in config.")

            agent = Agent(
                name=name,
                shared_memory=shared_memory,
                model_name=model_names_config[name],
                prompt_template_name=agent_user_prompts_config[name],
                specialized_agent_prompt=specialized_agent_prompts_config[name],
                instance_id=instance_id,
                api_key=api_keys_config[name],
            )
            local_agents.append(agent)
        agent_threads = local_agents # Store for joining

        # initialize the task queues for each agent
        specialized_agent_queues = {agent.name: agent.task_queue for agent in local_agents}
        specialized_agent_queues["analyst_agent"] = analyst_agent.queue

        # initialize the communication leader with references to the agents' queues and the queue for final result
        leader = CommunicationLeader( # This is a threading.Thread subclass
            shared_memory,
            specialized_agent_queues,
            analyst_agent=analyst_agent,
            final_result_queue=result_mp_queue
        )
        leader_thread = leader

        if process_shutdown_event.is_set(): return
        leader_thread.start()
        for ag_thread in agent_threads:
            if process_shutdown_event.is_set(): break
            ag_thread.start()
        
        # send the original task assignment to the analyst agent
        initial_task_for_analyst = {
            "type": "task",
            "sender": f"agent_process_{instance_id}",
            "content": query,
        }
        response = analyst_agent.process_task(initial_task_for_analyst, instance_id)
        clarification_mode = False

        if response:
            if analyst_agent.mode == "clarification":
                # if the current task is a clarification task, skip plan generation
                clarification_mode = True
            else:
                plan_steps = extract_json_plan(response)

                if plan_steps is not None: # Plan extracted
                    # initialize the subtask statuses
                    shared_memory.write_message({
                        "type": "subtasks_init",
                        "sender": f"agent_process_{instance_id}",
                        "tasks_plan": plan_steps
                    })
                    for step in plan_steps:
                        shared_memory.write_message(step)
                    # now the agents will start working on their tasks and the result will be eventually available in result_mp_queue
                else: # plan_steps is None
                    err_msg = "Error: Failed to extract a valid plan from the Analyst Agent."
                    result_mp_queue.put(err_msg)
        else:
            err_msg = "Error: Analyst Agent failed to response."
            result_mp_queue.put(err_msg)

        # If clarification mode is active, the initial response from the analyst agent is the final result.
        if clarification_mode:
            result_mp_queue.put(response)
            shared_memory.active = False
            process_shutdown_event.set()

        # Wait for the leader thread to complete.
        if leader_thread:
            while leader_thread.is_alive():
                if process_shutdown_event.is_set(): break
                leader_thread.join(timeout=0.5)
        for ag_thread in agent_threads:
            while ag_thread.is_alive():
                if process_shutdown_event.is_set(): break
                ag_thread.join(timeout=0.5)

        print(f"[{instance_id}] All threads completed. Process will exit.")


    except Exception as e:
        # Handle any exceptions that occur during the agent system process
        error_message = f"Error in agent system process: {str(e)}"
        result_mp_queue.put(error_message)
    finally:
        # Ensure all threads are signaled to stop and joined if an error occurred mid-way or if they are still alive for some reason.
        if leader_thread and leader_thread.is_alive():
            leader_thread.shutdown() # signals to shutdown
            leader_thread.join(timeout=5)
        for ag_thread in agent_threads:
            if ag_thread.is_alive():
                ag_thread.shutdown()
                ag_thread.join(timeout=5)
        try:
            # Reset the LocAgent context
            reset_current_issue()
        except Exception as e:
            print(f"[{instance_id}] Warning: Error resetting LocAgent context: {e}")
        
        print(f"[{instance_id}] Process finished.")


# --- Streamlit UI ---

# Function to clean up the background process on application exit
def perform_cleanup():
    print("ATELEXIT: Initiating cleanup...")

    # Clean up the background process
    if "background_process" in st.session_state and \
       st.session_state.background_process and \
       st.session_state.background_process.is_alive():
        print("ATELEXIT: Attempting to terminate background agent process...")
        try:
            st.session_state.background_process.terminate() # Send SIGTERM
            st.session_state.background_process.join(timeout=5)
            if st.session_state.background_process.is_alive():
                print("ATELEXIT: Background process still alive after SIGTERM, sending SIGKILL.")
                st.session_state.background_process.kill() # Send SIGKILL
                st.session_state.background_process.join(timeout=2) # Wait for kill
        except Exception as e:
            print(f"ATELEXIT: Error during background process termination: {e}")
        st.session_state.background_process = None
        print("ATELEXIT: Background agent process cleanup attempt finished.")

    # Clean up LocAgent temporary directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(script_dir, "loc_agent_outputs_resources")
    playground_dir = os.path.join(script_dir, "playground")
    dirs_to_clean = [
        os.path.join(resource_dir, "bm25_index_dir"),
        os.path.join(resource_dir, "graph_index_dir"),
        os.path.join(resource_dir, "LocAgentResults"),
        playground_dir
    ]

    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                # Delete the directory and its contents
                shutil.rmtree(dir_path)
                # Recreate the empty directory
                os.makedirs(dir_path)
            except OSError as e:
                print(f"ATELEXIT: Error cleaning directory {dir_path}: {e}")
            except Exception as e:
                print(f"ATELEXIT: An unexpected error occurred while cleaning {dir_path}: {e}")
        else:
            print(f"ATELEXIT: Directory {dir_path} does not exist, skipping cleanup.")

    print("ATELEXIT: Cleanup complete.")

# register cleanup function to be called on exit
atexit.register(perform_cleanup)

st.set_page_config(layout="wide")
st.title("CogniCollab - AI-Powered Software Assistant")

# a helpful usage message to the user
with st.expander("About this tool"):
    st.markdown("""
    This AI-powered software assistant uses a multi-agent system to automate routine coding tasks:
    
    1. **Analyst Agent**: Processes your query, analyzes the codebase, and creates a plan.
    2. **Specialized Agents**: Implementation, testing, debugging, and documentation agents work together to complete the tasks.
    3. **Communication Framework**: Coordinates the work between agents, managing dependencies and passing information.
    
    The system leverages large language models to understand code and generate solutions based on your instructions.
    
    ### How to Use
    
    1. **Select a Repository**: Click "Browse for Repository..." to choose the local codebase you want to work with.
    2. **Describe Your Task**: In the text area, clearly explain what you want to accomplish. You can:
       - Ask for new features to be implemented
       - Request bug fixes or improvements
       - Ask for tests to be written
       - Request documentation for existing code
    3. **Process Your Query**: Click the "Process Query" button and wait for the agents to work.
    4. **Review Results**: The system will provide detailed results of the completed task. Previous queries and their results will be stored in the Query History section
    
    For best results, be specific in your task descriptions and ensure your repository contains all necessary context.
    """)

# --- Initialize Session State ---
if "folder_path" not in st.session_state:
    st.session_state.folder_path = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "result_queue" not in st.session_state:
    st.session_state.result_queue = None
if "background_process" not in st.session_state:
    st.session_state.background_process = None

# for storing the history of past queries and results
if "results_history" not in st.session_state:
    st.session_state.results_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# --- UI Elements ---
query_input = st.text_area("Enter your development task or query:", height=150, key="query_input_area")

col1, col2 = st.columns([3, 1])

with col1:
    st.text(f"Selected Repository Path: {st.session_state.folder_path if st.session_state.folder_path else 'None'}")

# Add the API key configuration toggle here
use_separate_keys = st.toggle(
    "Use Separate API Keys (from env vars like IMPLEMENTATION_AGENT_API_KEY)",
    value=False, # Default to using a single API key
    help=f"Toggle to use separate API keys per agent (e.g., {', '.join(SEPARATE_API_KEY_ENV_VARS.values())}) instead of a single '{SINGLE_API_KEY_ENV_VAR}' for all.",
    key="separate_keys_toggle"
)

# button for selecting the local repository through a dedicated subprocess
with col2:
    if st.button("Browse for Repository..."):
        st.session_state.processing = False
        # st.session_state.result = None # Not used directly
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            selector_script = os.path.join(script_dir, "lib", "tk_dir_selector.py")
            if not os.path.exists(selector_script):
                 st.error(f"Error: tkDirSelector.py not found at {selector_script}")
            else:
                # Use subprocess to run the Tkinter directory selector script
                process = subprocess.Popen(
                    ["python", selector_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.path.join(script_dir, "lib")
                )
                stdout, stderr = process.communicate(timeout=60)
                if process.returncode == 0 and stdout:
                    st.session_state.folder_path = stdout.strip()
                    st.rerun()
                elif stderr:
                     st.error(f"Folder selection error: {stderr.strip()}")
                     st.session_state.folder_path = ""
                else:
                     st.warning("Folder selection cancelled.")
                     st.session_state.folder_path = ""
        except FileNotFoundError:
             st.error("Error: 'python' command not found. Make sure Python is in your system's PATH.")
        except subprocess.TimeoutExpired:
             st.error("Folder selection timed out.")
        except Exception as e:
            st.error(f"An unexpected error occurred during folder selection: {e}")
            st.session_state.folder_path = ""


# button to process the inputted query
if st.button("Process Query", disabled=st.session_state.processing):
    if query_input and st.session_state.folder_path:
        if not os.path.isdir(st.session_state.folder_path):
            st.error(f"Invalid repository path: {st.session_state.folder_path}")
        else:
            st.session_state.processing = True
            st.session_state.current_query = query_input
            # queue for the final system result
            st.session_state.result_queue = multiprocessing.Queue()

            separate_api_key_mode = False
            # --- Select API Key configuration based on toggle ---
            if st.session_state.get("separate_keys_toggle", False):
                current_api_keys_config = separate_api_keys_config
                # Optional: Basic check if any required keys are missing
                missing_keys = [name for name, key in current_api_keys_config.items() if not key]
                separate_api_key_mode = True
                if missing_keys:
                     st.error(f"Cannot proceed: Using separate keys, but API keys for the following agents are missing from environment variables: {', '.join(missing_keys)}. Please set them (e.g., {', '.join([SEPARATE_API_KEY_ENV_VARS[mk] for mk in missing_keys])}).")
                     # Stop execution here if keys are missing in separate mode
                     st.stop()
            else:
                current_api_keys_config = single_api_key_config
                if not os.getenv(SINGLE_API_KEY_ENV_VAR):
                     st.error(f"Cannot proceed: Environment variable for the collective API key (API_KEY) is missing")
                     # Stop execution here if keys are missing in separate mode
                     st.stop()            # --- End API Key selection ---

            process_args = (
                query_input,
                st.session_state.folder_path,
                st.session_state.result_queue,
                current_api_keys_config,
                model_names,
                agent_user_prompts,
                specialized_agent_prompts,
                separate_api_key_mode
            )

            # start the system initialization in a separate process
            st.session_state.background_process = multiprocessing.Process(
                target=run_agent_system,
                args=process_args,
                daemon=False
            )
            st.session_state.background_process.start()
            st.rerun()
    else:
        st.warning("Please enter a query and select a repository path.")

# While the agents are processing, show a spinner and wait for the final result to appear in the queue
if st.session_state.processing:
    with st.spinner("Agents are working in a separate process... Please wait."):
        try:
            # Check the queue non-blockingly
            result = st.session_state.result_queue.get(block=True, timeout=2)

            # Create a new entry in the results history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = {
                "query": st.session_state.current_query,
                "result": result,
                "timestamp": timestamp,
                "repo_path": st.session_state.folder_path
            }
            st.session_state.results_history.insert(0, history_entry)
            st.session_state.processing = False
            st.session_state.current_query = ""

            # cleanup the background process
            if st.session_state.background_process:
                st.write("Processing complete. Joining agent process...")
                st.session_state.background_process.join(timeout=10)
                if st.session_state.background_process.is_alive():
                    st.warning("Agent process did not terminate gracefully, attempting to terminate.")
                    st.session_state.background_process.terminate() # force if stuck
                    st.session_state.background_process.join(timeout=5) # Wait for terminate
                st.write("Agent process cleanup complete.")
                st.session_state.background_process = None
            st.rerun()

        except queue.Empty:
            time.sleep(0.1) # brief pause before application rerun
            st.rerun()
        except Exception as e: # catch other potential errors from queue or processing
            st.error(f"Error retrieving result: {e}")
            st.session_state.processing = False
            # put the error in the results history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_entry = {
                "query": st.session_state.current_query,
                "result": f"Error during processing: {e}",
                "timestamp": timestamp,
                "repo_path": st.session_state.folder_path
            }
            st.session_state.results_history.insert(0, error_entry)
            st.session_state.current_query = ""
            if st.session_state.background_process and st.session_state.background_process.is_alive():
                st.warning("Attempting to terminate hung agent process due to error.")
                st.session_state.background_process.terminate()
                st.session_state.background_process.join(timeout=5)
            st.session_state.background_process = None
            st.rerun()


# Display the latest result if available
if st.session_state.results_history:
    latest_entry = st.session_state.results_history[0]
    with st.container():
        st.markdown("**Query:**")
        st.code(latest_entry.get('query', ''))
        st.markdown("**Repository Path:**")
        st.write(latest_entry.get('repo_path', 'N/A'))
        st.divider()
        st.markdown("**Result:**")
        result_content = latest_entry.get('result', '')
        if isinstance(result_content, str) and result_content.startswith("Error:"):
            st.error(result_content)
        elif isinstance(result_content, str):
            st.markdown(result_content, unsafe_allow_html=False)
        else:
            st.write(str(result_content))
    st.markdown("---")

st.subheader("Query History")
# button to clear the history of old queries and results
if st.button("Clear History"):
    st.session_state.results_history = []
    st.rerun()

if not st.session_state.results_history:
    st.caption("No results yet.")
else:
    # display each entry of the results history
    for i, entry in enumerate(st.session_state.results_history):
        query_str = str(entry.get('query', ''))
        query_snippet = query_str[:50] + ('...' if len(query_str) > 50 else '')
        timestamp = entry.get('timestamp', f'unknown_time_{i}')
        with st.expander(f"Result {i+1}: {timestamp} - {query_snippet}"):
            st.write("**Query:**")
            st.code(query_str)
            st.write("**Repository Path:**")
            st.write(entry.get('repo_path', 'N/A'))
            st.divider()
            st.write("**Result:**")
            result_content = entry.get('result', '')
            if isinstance(result_content, str) and result_content.startswith("Error:"):
                st.error(result_content)
            elif isinstance(result_content, str):
                st.markdown(result_content, unsafe_allow_html=False)
            else:
                st.write(str(result_content))


if __name__ == '__main__':
    pass

