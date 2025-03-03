import socket
import threading
import json

class Agent:
    def __init__(self, name, server_host='127.0.0.1', server_port=5000):
        self.name = name
        self.server_host = server_host
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.socket.connect((self.server_host, self.server_port))
        self.register()

    def register(self):
        registration_message = json.dumps({"sender": self.name, "receiver": None, "message_type": "register", "content": "connect"})
        self.socket.send(registration_message.encode())

    def listen_for_messages(self):
        def listen():
            while True:
                try:
                    message = self.socket.recv(1024).decode()
                    if message:
                        print(f"[{self.name} RECEIVED] {message}")
                except ConnectionResetError:
                    print("[DISCONNECTED FROM SERVER]")
                    break
        threading.Thread(target=listen, daemon=True).start()

    def send_message(self, receiver, message_type, content):
        message = {
            "sender": self.name,
            "receiver": receiver,
            "message_type": message_type,
            "content": content,
            "timestamp": None  # Add timestamp if needed
        }
        self.socket.send(json.dumps(message).encode())

if __name__ == "__main__":
    agent = Agent("Agent1")
    agent.connect()
    agent.listen_for_messages()
    agent.send_message("Agent2", "greeting", "Hello, Agent2!")
