import threading
import queue
from typing import Dict, List
from pathlib import Path
from model_loader import load_model  # Assuming model_loader.py provides this function

class SharedMemory:
    def __init__(self):
        self.messages = []  # Shared message list
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.active = True
        print("[SHARED MEMORY] Shared memory initialized.")

    def write_message(self, message: dict):
        """Add a message to the shared memory."""
        with self.condition:  # Automatically acquires the lock
            self.messages.append(message)
            print(f"[SHARED MEMORY] notifting all those waiting for messages...")
            self.condition.notify_all()  # Notify a waiting thread that a message is available
            print(f"[SHARED MEMORY] unlocking...")
        return

    def read_all_messages(self) -> List[dict]:
        """Read and clear all messages."""
        print("[SHARED MEMORY] About to lock messages...")
        with self.condition:
            print(f"[SHARED MEMORY] Reading messages: {self.messages}")
            messages = self.messages[:]
            self.messages.clear()
        return messages
    
    def clear(self):
        """Clear all messages."""
        print("[SHARED MEMORY] Clearing all messages...")
        with self.lock:
            print("[SHARED MEMORY] Lock acquired.")
            self.messages.clear()

class CommunicationLeader(threading.Thread):
    def __init__(self, shared_memory: SharedMemory, agent_queues: Dict[str, queue.Queue]):
        super().__init__()
        self.shared_memory = shared_memory
        self.agent_queues = agent_queues
        self.shutdown_event = threading.Event()
        print("[LEADER] Communication Leader initialized.")

    def run(self):
        while not self.shutdown_event.is_set():
            # Wait for a message to be available
            with self.shared_memory.condition:
                while not self.shared_memory.messages and not self.shutdown_event.is_set():
                    self.shared_memory.condition.wait()
                    print("[LEADER] Woke up from waiting.")

            if self.shutdown_event.is_set():
                print("[LEADER] Shutdown signal received.")
                break

            # Fetch messages to process
            print("[LEADER] starting to read messages...")
            messages = self.shared_memory.read_all_messages()

            print(f"[LEADER] Processing messages: {messages}")
            for message in messages:
                print(f"[LEADER] Processing message: {message}")
                self.process_message(message)

    def process_message(self, message: dict):
        """Process a message and assign tasks to agents."""
        message_type = message.get("type")
        if message_type == "task":
            self.handle_task(message)
        elif message_type == "message":
            self.handle_message(message)
        elif message_type == "task_result":
            print(f"[LEADER] Task result received: {message}")
        elif message_type == "shutdown":
            self.shutdown()
        else:
            print(f"[ERROR] Unknown message type: {message_type}")

    def handle_task(self, task: dict):
        """Handle a task message and route it to the target agent."""
        target_agent = task.get("target_agent")
        print(f"[LEADER] Handling task: {task}")
        print(f"[LEADER] Target agent: {target_agent}")
        if target_agent and target_agent in self.agent_queues:
            print(f"[LEADER] Sending task to {target_agent}: {task}")
            self.agent_queues[target_agent].put(task)
            print(f"[LEADER] Task sent to {target_agent}.")
            print(f"[LEADER] - Agent queues: {self.agent_queues}")
            print(f"[LEADER] - Current agent queue contents: {list(self.agent_queues[target_agent].queue)}")
        else:
            print(f"[ERROR] Invalid target agent: {target_agent}")

    def handle_message(self, message: dict):
        """Handle a generic message."""
        print(f"[LEADER] Received message: {message}")

    def shutdown(self):
        print("[LEADER] Shutting down...")
        self.shared_memory.active = False # Set shared memory to inactive
        self.shared_memory.clear()  # Clear all messages
        self.shutdown_event.set()
        with self.shared_memory.condition:
            self.shared_memory.condition.notify_all()  # Notify all waiting threads

class Agent(threading.Thread):
    def __init__(self, name: str, shared_memory: SharedMemory, model_path: Path, device: str = "cpu"):
        super().__init__()
        self.name = name
        self.shared_memory = shared_memory
        self.task_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.llm = load_model(model_path)
        self.device = device
        print(f"[{self.name}] Agent initialized.")

    def send_message(self, content: str, target_agent: str = None, message_type: str = "message"):
        """Send a message to shared memory."""
        message = {
            "type": message_type,
            "sender": self.name,
            "content": content,
            "target_agent": target_agent
        }
        if self.shared_memory.active:
            self.shared_memory.write_message(message)

    def run(self):
        while not self.shutdown_event.is_set():
            if not self.shared_memory.active:
                print(f"[{self.name}] - Shared memory inactive, exiting task processing.")
                break
            try:
                print(f"[{self.name}] - Waiting for task...")
                if not self.task_queue.empty():
                    print(f"[{self.name}] - Current queue contents: {list(self.task_queue.queue)}")
                task = self.task_queue.get(timeout=1)  # Block until a task is available
                print(f"[{self.name}] - Got task: {task}")
                self.process_task(task)
            except queue.Empty:
                continue

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

if __name__ == "__main__":
    # Initialize shared memory and agents
    shared_memory = SharedMemory()
    agent_queues = {"Agent1": queue.Queue(), "Agent2": queue.Queue()}

    # Create agents
    agents = [Agent(name, shared_memory, model_path=Path("path/to/model")) for name in agent_queues.keys()]
    for agent in agents:
        agent.start()

    # Create and start the Communication Leader
    leader = CommunicationLeader(shared_memory, agent_queues)
    leader.start()

    # Example interaction
    agents[0].send_message("Example task for Agent2", target_agent="Agent2", message_type="task")

    # Allow time for processing
    try:
        threading.Event().wait(5)
    finally:
        # Shutdown all threads
        leader.shutdown()
        for agent in agents:
            agent.shutdown()
        leader.join()
        for agent in agents:
            agent.join()
