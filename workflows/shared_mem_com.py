import threading
import queue
from typing import Dict, List
from workflows.unified_agent_class import UnifiedAgent
from workflows.agent_logic import agent_logic_loop
from analyst_agent.analyst_agent import AnalystAgent
import time


# Class for managing dependencies between subtasks. Some need to be completed before others can start.
class SharedCompletedSubtasks:
    """
    Centralized manager for tracking subtask completion status and dependencies.
    Only the Analyst Agent can write to this structure, while other agents have read access.
    """
    def __init__(self):
        self.subtasks = {}  # Dictionary of subtask metadata indexed by step number
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def initialize_subtasks(self, tasks_plan: List[dict]):
        """Initialize the subtasks structure from the analyst's plan."""
        # Set the initial state of all subtasks to not completed
        with self.lock:
            for task in tasks_plan:
                step = task.get("step", 0)
                self.subtasks[step] = {
                    "step": step,
                    "agent": task.get("target_agent", ""),
                    "completed": False,
                    "content": "",
                    "dependencies": task.get("dependencies", [])
                }


    def update_subtask(self, step: int, content: str, completed: bool = True):
        """Update a subtask's completion status and content (Analyst Agent only)."""
        # Change the completed subtask's entry to include the content and mark it as completed
        with self.condition:
            if step in self.subtasks:
                self.subtasks[step]["completed"] = completed
                self.subtasks[step]["content"] = content
                print(f"[SUBTASKS] Updated subtask {step} - completed: {completed}")
                self.condition.notify_all()  # Notify waiting agents
            else:
                print(f"[SUBTASKS] Error: Step {step} not found in subtasks")


    # Check if a specific subtask is completed
    def is_subtask_completed(self, step: int) -> bool:
        """Check if a specific subtask is completed."""
        with self.lock:
            if step in self.subtasks:
                return self.subtasks[step]["completed"]
            return False


    # Wait for all dependencies to be completed. Agents' operation is contained here while waiting.
    def wait_for_dependencies(self, dependencies: List[int], timeout=None):
        """Wait until all dependencies are completed or timeout occurs."""
        print(f"[SUBTASKS] Waiting for dependencies {dependencies}")
        with self.condition:  # Use condition for waiting
            end_time = None
            if timeout is not None:
                end_time = time.time() + timeout
            
            # Check dependency state
            while True:
                all_completed = True
                for dep in dependencies:
                    if dep not in self.subtasks or not self.subtasks[dep]["completed"]:
                        all_completed = False
                        print(f"[SUBTASKS] Still waiting for dependency {dep}")
                        break
                
                if all_completed:
                    return True
                    
                if end_time and time.time() >= end_time:
                    print(f"[SUBTASKS] Timeout waiting for dependencies {dependencies}")
                    return False
                
                # Wait for notification that a subtask has been completed
                self.condition.wait()
                print(f"[SUBTASKS] Thread {threading.current_thread().name} woke up")


    # Get the content of a completed subtask
    def get_subtask_content(self, step: int) -> str:
        """Get the content/output of a completed subtask."""
        with self.lock:
            if step in self.subtasks and self.subtasks[step]["completed"]:
                return self.subtasks[step]["content"]
            return ""


# Shared memory for storing messages interchanged between agents
class SharedMemory:
    def __init__(self):
        self.messages = []  # Shared message list
        self.lock = threading.Lock() # protect the memory with a lock
        self.condition = threading.Condition(self.lock)
        self.active = True
        self.completed_subtasks = SharedCompletedSubtasks()


    # Add a message to the shared memory storage
    def write_message(self, message: dict):
        """Add a message to the shared memory."""
        with self.condition:
            self.messages.append(message)
            print(f"[SHARED MEMORY] notifting all those waiting for messages...")
            self.condition.notify_all()  # Notify a waiting thread that a message is available
        return


    # Read all messages and clear the list
    def read_all_messages(self) -> List[dict]:
        """Read and clear all messages."""
        with self.condition:
            messages = self.messages[:]
            self.messages.clear()
        return messages


    # Clear all messages from the storage
    def clear(self):
        """Clear all messages."""
        print("[SHARED MEMORY] Clearing all messages...")
        with self.lock:
            self.messages.clear()


# Communication leader that manages the flow of messages between agents. All messages are processed and forwarded if necessary.
class CommunicationLeader(threading.Thread):
    def __init__(self, shared_memory: SharedMemory, agent_queues: Dict[str, queue.Queue], analyst_agent: AnalystAgent = None, final_result_queue: queue.Queue = None):
        super().__init__()
        self.shared_memory = shared_memory
        self.agent_queues = agent_queues # reference to the queues of all agents for forwarding messages
        self.shutdown_event = threading.Event()
        self.total_subtasks = 0 # Total number of subtasks to be completed
        self.completed_subtasks_count = 0 # keeps track of how many subtasks have been completed
        self.original_task = None # save the high-level objective
        self.analyst_agent = analyst_agent
        self.final_result_queue = final_result_queue
        print("[LEADER] Communication Leader initialized.")


    # Main loop for the leader agent
    def run(self):
        while not self.shutdown_event.is_set():
            # Wait for a message to be available
            with self.shared_memory.condition:
                while not self.shared_memory.messages and not self.shutdown_event.is_set():
                    self.shared_memory.condition.wait()

            if self.shutdown_event.is_set():
                print("[LEADER] Shutdown signal received.")
                break

            # Fetch messages to process
            messages = self.shared_memory.read_all_messages()

            # process all messages
            for message in messages:
                print(f"[LEADER] Processing incomming message...")
                self.process_message(message)

        
    def process_message(self, message: dict):
        """Process a message and assign tasks to agents."""
        message_type = message.get("type").lower()
        # perform the corresponding action based on the message type
        if message_type == "task":
            self.handle_task(message)
        elif message_type == "message":
            self.handle_message(message)
        elif message_type == "task_result":
            self.handle_task_result(message)
        elif message_type == "subtasks_init":
            self.handle_subtasks_init(message)
        elif message_type == "shutdown":
            self.shutdown()
        else:
            print(f"[ERROR] Unknown message type: {message_type}")


    def handle_task_result(self, message: dict):
        """Handle a task result message and update completed subtasks."""
        print(f"[LEADER] Task result received from {message.get('sender')}")
        step = message.get("step")
        content = message.get("content", "")
        
        # The agent's structure may return "Error" if it fails to execute the task
        if content == "Error":
            message["content"] = "[AGENT FAILD TO EXECUTE TASK - TRY TO REACH THE OBJECTIVE WITHOUT THIS SUB-TASK]"

        elif step is not None:
            # update the dependency management structure with the new result
            self.shared_memory.completed_subtasks.update_subtask(step, content)
            self.completed_subtasks_count += 1

        # Forward the message to analyst's queue for later consolidation
        if "analyst_agent" in self.agent_queues:
            print(f"[LEADER] Forwarding task result to analyst's queue")
            self.agent_queues["analyst_agent"].put(message)

        # Check if all subtasks are now completed
        if self.total_subtasks > 0 and self.completed_subtasks_count >= self.total_subtasks:
            print(f"[LEADER] All {self.total_subtasks} subtasks completed! Directly triggering consolidation...")
            self._trigger_consolidation()


    def handle_subtasks_init(self, message: dict):
        """Initialize the subtasks structure from the analyst's plan."""
        tasks_plan = message.get("tasks_plan", [])
        if tasks_plan:
            # Initialize dependency management
            self.shared_memory.completed_subtasks.initialize_subtasks(tasks_plan)
            self.total_subtasks = len(tasks_plan)
            self.completed_subtasks_count = 0

            # Store original task from the first task
            if tasks_plan and "original_task" in tasks_plan[0]:
                self.original_task = tasks_plan[0]["original_task"]

            print(f"[LEADER] Initialized subtasks tracking with {len(tasks_plan)} tasks")


    # triggers the consolidation process of Analyst Agent and inputs the final result into the queue for Streamlit application
    def _trigger_consolidation(self):
        """Directly call the analyst agent's consolidation method."""
        if self.analyst_agent:
            print("[LEADER] All tasks completed. Directly calling consolidation method.")
            try:
                result = self.analyst_agent.consolidate_results(original_task=self.original_task)
                print(f"[LEADER] Consolidation complete. Result:\n{result}")

                if self.final_result_queue: # Check if the queue was passed
                    self.final_result_queue.put(result)
            except Exception as e:
                print(f"[LEADER] Error during consolidation: {e}")
        else:
            print("[LEADER] Error: No analyst_agent available for consolidation")


    # Handle a task message and route it to the target agent
    def handle_task(self, task: dict):
        """Handle a task message and route it to the target agent."""
        target_agent = task.get("target_agent")
        if target_agent and target_agent.lower() in self.agent_queues:
            self.agent_queues[target_agent].put(task)
            print(f"[LEADER] Task sent to {target_agent}.")

        else:
            print(f"[ERROR] Invalid target agent: {target_agent}")


    # Messages require no additional processing
    def handle_message(self, message: dict):
        """Handle a generic message."""
        print(f"[LEADER] Received message: {message}")


    # Shutdown the leader if signaled
    def shutdown(self):
        print("[LEADER] Shutting down...")
        self.shared_memory.active = False # Set shared memory to inactive
        self.shared_memory.clear()  # Clear all messages
        self.shutdown_event.set()
        with self.shared_memory.condition:
            self.shared_memory.condition.notify_all()  # Notify all waiting threads


# Class encapsulating each specialized agent to realize active coordination.
class Agent(threading.Thread):
    def __init__(self, name: str, shared_memory: SharedMemory, specialized_agent_prompt: str = None, api_key: str = None, prompt_template_name: str = None, model_name: str = None, instance_id: str = None):
        super().__init__()
        self.name = name
        self.shared_memory = shared_memory
        self.task_queue = queue.Queue() # agent's queue for task assignments
        self.shutdown_event = threading.Event()
        self.specialized_agent_prompt = specialized_agent_prompt # specialized prompt unique to each agent
        self.instance_id = instance_id

        # Construct the parameters for the underlying LLM class initialization
        llm_kwargs = {
            "prompt_template": prompt_template_name
        }

        if model_name is not None:
            llm_kwargs["model_name"] = model_name

        if api_key is not None:
            llm_kwargs["api_key"] = api_key

        self.llm = UnifiedAgent(**llm_kwargs)
        print(f"[{self.name}] Agent initialized.")


    # Send a message to the shared memory storage
    def send_message(self, content: str, target_agent: str = "", message_type: str = "message", step: int = None):
        """Send a message to shared memory."""
        # construct the message
        message = {
            "type": message_type,
            "sender": self.name,
            "content": content,
            "target_agent": target_agent
        }
        print(f"[{self.name}] Sending message: {message} || of step {step}")
        # Add step number if provided
        if step is not None:
            message["step"] = step
            
        if self.shared_memory.active:
            self.shared_memory.write_message(message)


    # Main loop for the agent's execution
    def run(self):
        # Operate until shutdown is signaled
        while not self.shutdown_event.is_set():
            if not self.shared_memory.active:
                print(f"[{self.name}] - Shared memory inactive, exiting task processing.")
                break
            try:
                task = self.task_queue.get(timeout=1)  # Block until a task is available
                print(f"[{self.name}] - Got task: {task}")
                # call the task execution function
                result = self.process_task_iteratively(task)
                
                if not result:
                    # If no result is returned, propagate the information to the leader
                    print(f"[{self.name}] - ERROR! No result from task processing. Agent failed to respond.")
                    self.send_message("Error", message_type="task_result", target_agent="analyst_agent", step=task.get("step", 0))
                else:
                    self.send_message(result, message_type="task_result", target_agent="analyst_agent", step=task.get("step", 0))
            except queue.Empty:
                continue


    # Process a task iteratively using the agent logic loop, waiting for dependencies to complete before proceeding
    def process_task_iteratively(self, task: dict) -> str:
        """
        Uses an iterative reasoning loop to solve the task.
        Waits for dependencies to complete before processing.
        """
        query = task.get("content", "")
        context = task.get("context", "")
        original_query = task.get("original_task", "")
        step = task.get("step", 0)
        dependencies = task.get("dependencies", [])

        # Wait for dependencies if any
        if dependencies:
            print(f"[{self.name}] - Waiting for dependencies: {dependencies}")
            deps_completed = self.shared_memory.completed_subtasks.wait_for_dependencies(dependencies)
            
            if not deps_completed:
                print(f"[{self.name}] - Timeout waiting for dependencies. Proceeding anyway.")
            else:
                print(f"[{self.name}] - All dependencies completed. Proceeding with task.")
                
            # Gather context from dependency results if available
            dep_context = []
            for dep in dependencies:
                content = self.shared_memory.completed_subtasks.get_subtask_content(dep)
                if content:
                    dep_context.append(f"Result from step {dep}:\n{content}")
            
            if dep_context:
                context += "\n\n" + "\n\n".join(dep_context)

        # Build the query string with context if it's available
        if context:
            query = f"----- Subtask That You Need to Complete (step no. {step} out of the whole objective) -----\n{query}\n\n------ Code Context for Task Execution (files, classes or functions that pretain to this subtask) + results from dependent subtasks of other agents -----\n{context}\n\n----- Original Hgh-Level Objective That Your Subtask Is Part Of -----\n{original_query}"

        instance_data = {
            "instance_id": self.instance_id,
            "problem_statement": query,
            "repo": "", # custom local repo passed by the developer
            "base_commit": "latest",
            "patch": ""
        }

        # Call the agent logic loop with the specialized prompt that will complete the task
        final_output = agent_logic_loop(agent_llm=self.llm, instance_data_dict=instance_data, agent_specific_prompt=self.specialized_agent_prompt)

        return final_output


    def process_task(self, task: dict):
        """Process an incoming task."""
        print(f"[{self.name}] Processing task: {task}")
        try:
            result = self.llm.generate(task)  # Use the LLM to process the task
            print(f"[{self.name}] Task result: {result}")
            self.send_message(result, message_type="task_result")
        except Exception as e:
            print(f"[{self.name}] Error processing task: {e}")


    def shutdown(self):
        print(f"[{self.name}] Shutting down...")
        self.shutdown_event.set()

