import socket
import threading
import queue
import json
from model_loader import load_model  # Assuming you have a model loader

class OrderedChatAgent(threading.Thread):
    def __init__(self, name, model_path, server_host='127.0.0.1', server_port=5000, device='cpu'):
        super().__init__(daemon=True)
        self.name = name
        self.server_host = server_host
        self.server_port = server_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.has_token = False
        self.task_queue = queue.Queue()
        self.llm = load_model(model_path)(device=device)  # Initialize the LLM

    def connect(self):
        """Connect to the server and register the agent."""
        self.client.connect((self.server_host, self.server_port))
        self.client.send(self.name.encode())  # Send name to the server
        print(f"[{self.name}] Connected to the server.")

    def listen_for_messages(self):
        """Listen for messages from the server."""
        def listen():
            while True:
                try:
                    message = self.client.recv(1024).decode()
                    if message:
                        self.process_message(json.loads(message))
                except Exception as e:
                    print(f"[{self.name} ERROR] Unable to receive message: {e}")
                    break

        threading.Thread(target=listen, daemon=True).start()

    def process_message(self, message_data):
        """Process incoming messages."""
        if message_data.get("type") == "token_grant":
            self.has_token = True
            print(f"[{self.name}] Token received. Processing tasks.")
            self.process_tasks()
            self.pass_token()
        elif message_data.get("type") == "task":
            task = message_data.get("content")
            print(f"[{self.name}] Received task: {task}")
            self.task_queue.put(task)

    def send_message(self, content):
        """Send a message to the server."""
        if self.has_token:
            message = {
                "sender": self.name,
                "type": "message",
                "content": content
            }
            self.client.send(json.dumps(message).encode())
            print(f"[{self.name}] Message sent: {content}")
        else:
            print(f"[{self.name}] Cannot send message, no token!")

    def pass_token(self):
        """Pass the token to the next agent."""
        if self.has_token:
            token_message = {"type": "token_pass"}
            self.client.send(json.dumps(token_message).encode())
            self.has_token = False
            print(f"[{self.name}] Token passed.")

    def process_tasks(self):
        """Process tasks from the queue."""
        while not self.task_queue.empty():
            task = self.task_queue.get()
            print(f"[{self.name}] Processing task: {task}")
            # Process the task using the LLM
            result = self.llm.generate(task)  # Example usage
            print(f"[{self.name}] Task result: {result}")
            self.task_queue.task_done()

    def run(self):
        """Run the agent."""
        self.connect()
        self.listen_for_messages()

# Example usage
if __name__ == "__main__":
    agent = OrderedChatAgent(
        name="Agent1",
        model_path="path/to/model",
        server_host="127.0.0.1",
        server_port=5000,
        device="cpu"
    )
    agent.start()
