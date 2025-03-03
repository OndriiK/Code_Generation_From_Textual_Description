import socket
import threading
import json

class OrderedChatAgent:
    def __init__(self, name, server_host='127.0.0.1', server_port=5000):
        self.name = name
        self.server_host = server_host
        self.server_port = server_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.has_token = False

    def connect(self):
        self.client.connect((self.server_host, self.server_port))
        self.client.send(self.name.encode())  # Send name to the server

    def listen_for_messages(self):
        def listen():
            while True:
                try:
                    message = self.client.recv(1024).decode()
                    if message:
                        self.process_message(json.loads(message))
                except:
                    print("[ERROR] Unable to receive message.")
                    break
        threading.Thread(target=listen, daemon=True).start()

    def process_message(self, message_data):
        if message_data.get("type") == "token_grant":
            self.has_token = True
            print(f"[{self.name}] Token received. Ready to send.")
            # Example of sending a message when token is received
            self.send_message("Hello, I have the token!")
            self.pass_token()

    def send_message(self, content):
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
        if self.has_token:
            token_message = {"type": "token_pass"}
            self.client.send(json.dumps(token_message).encode())
            self.has_token = False
            print(f"[{self.name}] Token passed.")

if __name__ == "__main__":
    agent = OrderedChatAgent("Agent1")
    agent.connect()
    agent.listen_for_messages()
