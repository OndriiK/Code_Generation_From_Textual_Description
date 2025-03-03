from threading import Thread
from server import OrderedChatServerWithMemory, OrderedChatAgent
import socket
import json
import time

def run_server():
    """Starts the server."""
    server = OrderedChatServerWithMemory()
    server.run()

def run_agent():
    """Starts the intent_manager agent."""
    agent = OrderedChatAgent(
        name="intent_manager",
        model_path="/mnt/d/wsl_workspace/fine_tuned_bert_v4",
        server_host="127.0.0.1",
        server_port=5000,
        device="cpu"  # Change to 'cuda' if GPU is available
    )
    agent.start()

def send_task():
    """Sends a test task to the server."""
    time.sleep(2)  # Give time for server and agent to initialize
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 5000))
        client.send(b"test_client")  # Register as a test client

        # Construct the task message
        task_message = {
            "type": "task",
            "target_agent": "intent_manager",
            "content": "Resolve unmatching variable names and add unit tests for my code"
        }
        client.send(json.dumps(task_message).encode())
        print("[TEST] Task sent to the server.")
        client.close()
    except Exception as e:
        print(f"[TEST ERROR] Failed to send task: {e}")

if __name__ == "__main__":
    # Start the server
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()

    # Start the agent
    agent_thread = Thread(target=run_agent, daemon=True)
    agent_thread.start()

    # Send a test task
    send_task()
