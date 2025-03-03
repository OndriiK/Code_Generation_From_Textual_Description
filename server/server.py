import socket
import threading
import json
import os
import queue
from model_loader import load_model
from pathlib import Path


class OrderedChatServerWithMemory:
    def __init__(self, host='127.0.0.1', port=5000, save_file="messages.json"):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(5)
        self.clients = {}  # Map of agent_name -> (address, socket)
        self.messages = []  # In-memory message storage
        self.save_file = save_file
        self.task_queue = queue.Queue()  # Shared task queue
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.load_messages()

    def handle_client(self, client_socket, addr):
        """Handles new client connections."""
        try:
            agent_name = client_socket.recv(1024).decode()  # Expect agent to send its name first
            with self.lock:
                self.clients[agent_name] = (addr, client_socket)
                print(f"[NEW AGENT] {agent_name} connected.")
        except Exception as e:
            print(f"[ERROR] Failed to register client: {e}")
            return

        while not self.shutdown_event.is_set():
            try:
                client_socket.settimeout(1)  # Timeout to check shutdown_event periodically
                message = client_socket.recv(1024).decode()
                if message:
                    self.process_message(json.loads(message), agent_name)
            except socket.timeout:
                continue  # Check shutdown_event on timeout
            except (ConnectionResetError, json.JSONDecodeError):
                break

        # Clean up the client connection
        with self.lock:
            self.clients.pop(agent_name, None)
        print(f"[DISCONNECTED] {agent_name} disconnected.")
        client_socket.close()

    def process_message(self, message_data, sender_name):
        """Processes incoming messages from agents."""
        message_type = message_data.get("type")
        if message_type == "message":
            with self.lock:
                self.messages.append(message_data)
                self.save_messages()
                print(f"[MESSAGE CAPTURED] {message_data}")
        elif message_type == "task":
            self.add_task_to_queue(message_data)
        elif message_type == "shutdown":
            print("[SERVER] Shutdown signal received.")
            self.shutdown_event.set()  # Trigger shutdown
        elif message_type == "task_result":
            print(f"[RESULT RECEIVED] {message_data}")

    def add_task_to_queue(self, task):
        """Adds a task to the shared task queue."""
        with self.lock:
            self.task_queue.put(task)
            print(f"[TASK ADDED] {task}")

    def distribute_tasks(self):
        """Assign tasks to the intended agent."""
        while True:
            task = self.task_queue.get()
            target_agent = task.get("target_agent")
            if not target_agent:
                print(f"[ERROR] Task without a target: {task}")
                continue
            with self.lock:
                if target_agent in self.clients:
                    _, client_socket = self.clients[target_agent]
                    client_socket.send(json.dumps({"type": "task", "content": task}).encode())
                    print(f"[TASK DISTRIBUTED] Sent to {target_agent}")
                else:
                    print(f"[ERROR] Target agent {target_agent} not found.")
            self.task_queue.task_done()

    def save_messages(self):
        """Saves messages to a file."""
        with open(self.save_file, 'w') as f:
            json.dump(self.messages, f)
        print("[MESSAGES SAVED]")

    def load_messages(self):
        """Loads messages from a file if it exists."""
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                self.messages = json.load(f)
            print("[MESSAGES LOADED]")

    def run(self):
        print("[SERVER STARTED]")
        distribute_thread = threading.Thread(target=self.distribute_tasks)
        distribute_thread.start()
        try:
            while not self.shutdown_event.is_set():
                self.server.settimeout(1)  # Allow periodic checks for shutdown
                try:
                    client_socket, addr = self.server.accept()
                    threading.Thread(target=self.handle_client, args=(client_socket, addr)).start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("[SERVER] Interrupted by user.")
        finally:
            self.shutdown()  # Ensure cleanup on exit
            distribute_thread.join()

    def shutdown(self):
        """Shutdown the server and notify all connected agents."""
        self.shutdown_event.set()

        with self.lock:
            # Send shutdown message to all agents
            for agent_name, (_, client_socket) in self.clients.items():
                try:
                    shutdown_message = {
                        "type": "shutdown"
                    }
                    client_socket.send(json.dumps(shutdown_message).encode())
                    print(f"[SERVER] Sent shutdown message to {agent_name}")
                except Exception as e:
                    print(f"[SERVER] Failed to send shutdown message to {agent_name}: {e}")

            # Close all client sockets
            for agent_name, (_, client_socket) in self.clients.items():
                client_socket.close()
                print(f"[SERVER] Closed connection to {agent_name}")

            self.clients.clear()

        # Close the server socket
        self.server.close()
        print("[SERVER] Clean shutdown complete.")


class OrderedChatAgent(threading.Thread):
    def __init__(self, name, model_path, server_host='127.0.0.1', server_port=5000, device='cpu'):
        super().__init__()
        self.name = name
        self.server_host = server_host
        self.server_port = server_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.task_queue = queue.Queue()
        self.llm = load_model(Path(model_path))  # Initialize the LLM
        self.shutdown_event = threading.Event()

    def connect(self):
        """Connect to the server and register the agent."""
        print(f"[{self.name}] Connecting to the server...")
        self.client.connect((self.server_host, self.server_port))
        self.client.send(self.name.encode())  # Send name to the server
        print(f"[{self.name}] Connected to the server.")

    def listen_for_messages(self):
        """Listen for messages from the server."""
        def listen():
            while not self.shutdown_event.is_set():
                try:
                    message = self.client.recv(1024).decode()
                    if message:
                        self.process_message(json.loads(message))
                except Exception as e:
                    if not self.shutdown_event.is_set():
                        print(f"[{self.name} ERROR] Unable to receive message: {e}")
                    break

        threading.Thread(target=listen, daemon=True).start()

    def process_message(self, message_data):
        """Process incoming messages."""
        if message_data.get("type") == "task":
            task = message_data.get("content")
            print(f"[{self.name}] Received task from server: {task}")
            self.task_queue.put(task)  # Server adds tasks to the queue
        elif message_data.get("type") == "message":
            print(f"[{self.name}] Received message: {message_data.get('content')}")
        elif message_data.get("type") == "shutdown":
            print(f"[{self.name}] Shutdown signal received.")
            self.shutdown()

    def process_tasks(self):
        """Process tasks from the queue."""
        while not self.shutdown_event.is_set():
            task = self.task_queue.get()  # Block until a task is available
            print(f"[{self.name}] Processing task: {task}")
            try:
                result = self.llm.generate(task)  # Pass the task dictionary
                print(f"[{self.name}] Task result: {result}")
                self.send_message(result)  # Send the result back as a message
            except Exception as e:
                print(f"[{self.name}] Error processing task: {e}")
            finally:
                self.task_queue.task_done()

    def send_message(self, content):
        """Send a message to the server."""
        message = {
            "sender": self.name,
            "type": "message",
            "content": content
        }
        self.client.send(json.dumps(message).encode())
        print(f"[{self.name}] Message sent: {content}")

    def run(self):
        self.connect()
        self.listen_for_messages()
        while not self.shutdown_event.is_set():
            self.process_tasks()  # Continuously process tasks from the queue

    def shutdown(self):
        self.shutdown_event.set()
        self.client.close()
        print(f"[{self.name}] Clean shutdown complete.")


if __name__ == "__main__":
    server = OrderedChatServerWithMemory()
    server.run()
    agent = OrderedChatAgent(
        name="Agent1",
        model_path="path/to/model",
        server_host="127.0.0.1",
        server_port=5000,
        device="cpu"
    )
    agent.start()
