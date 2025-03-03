import socket
import threading
import json
import os

class OrderedChatServerWithMemory:
    def __init__(self, host='127.0.0.1', port=5000, save_file="messages.json"):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(5)
        self.clients = []  # List of connected agents (name, address, socket)
        self.current_token_index = 0  # Index of the agent holding the token
        self.messages = []  # In-memory message storage
        self.save_file = save_file
        self.lock = threading.Lock()
        self.load_messages()

    def handle_client(self, client_socket, addr):
        agent_name = client_socket.recv(1024).decode()  # Expect agent to send its name first
        with self.lock:
            self.clients.append((agent_name, addr, client_socket))
            print(f"[NEW AGENT] {agent_name} connected.")

        while True:
            try:
                message = client_socket.recv(1024).decode()
                if message:
                    self.process_message(json.loads(message), client_socket)
            except (ConnectionResetError, json.JSONDecodeError):
                break

        with self.lock:
            self.clients = [(name, a, sock) for name, a, sock in self.clients if sock != client_socket]
        print(f"[DISCONNECTED] {agent_name} disconnected.")
        client_socket.close()

    def process_message(self, message_data, client_socket):
        if message_data.get("type") == "message":
            with self.lock:
                self.messages.append(message_data)  # Save message in memory
                self.save_messages()  # Persist messages to file
                print(f"[MESSAGE CAPTURED] {message_data}")

        elif message_data.get("type") == "token_pass":
            with self.lock:
                self.current_token_index = (self.current_token_index + 1) % len(self.clients)
                next_agent = self.clients[self.current_token_index]
                next_agent[2].send(json.dumps({"type": "token_grant"}).encode())
                print(f"[TOKEN] Passed to {next_agent[0]}")

    def save_messages(self):
        with open(self.save_file, 'w') as f:
            json.dump(self.messages, f)
        print("[MESSAGES SAVED]")

    def load_messages(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                self.messages = json.load(f)
            print("[MESSAGES LOADED]")

    def run(self):
        print("[SERVER STARTED]")
        threading.Thread(target=self.token_handler, daemon=True).start()
        while True:
            client_socket, addr = self.server.accept()
            threading.Thread(target=self.handle_client, args=(client_socket, addr), daemon=True).start()

    def token_handler(self):
        while True:
            with self.lock:
                if self.clients:
                    first_agent = self.clients[self.current_token_index]
                    first_agent[2].send(json.dumps({"type": "token_grant"}).encode())
                    break

if __name__ == "__main__":
    server = OrderedChatServerWithMemory()
    server.run()
