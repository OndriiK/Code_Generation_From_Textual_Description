import socket
import threading
import json

class ChatRoomServer:
    def __init__(self, host='127.0.0.1', port=5000):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen(5)
        self.clients = {}

    def handle_client(self, client_socket, addr):
        print(f"[NEW CONNECTION] {addr} connected.")
        while True:
            try:
                message = client_socket.recv(1024).decode()
                if message:
                    message_data = json.loads(message)
                    self.route_message(message_data, client_socket)
            except (ConnectionResetError, json.JSONDecodeError):
                break
        print(f"[DISCONNECTED] {addr} disconnected.")
        self.remove_client(client_socket)

    def route_message(self, message_data, client_socket):
        receiver = message_data.get("receiver")
        if receiver and receiver in self.clients:
            self.clients[receiver].send(json.dumps(message_data).encode())
        else:
            response = {"error": f"Receiver {receiver} not found."}
            client_socket.send(json.dumps(response).encode())

    def remove_client(self, client_socket):
        for name, socket in self.clients.items():
            if socket == client_socket:
                del self.clients[name]
                break

    def run(self):
        print("[SERVER STARTED]")
        while True:
            client_socket, addr = self.server.accept()
            threading.Thread(target=self.handle_client, args=(client_socket, addr)).start()

if __name__ == "__main__":
    server = ChatRoomServer()
    server.run()
