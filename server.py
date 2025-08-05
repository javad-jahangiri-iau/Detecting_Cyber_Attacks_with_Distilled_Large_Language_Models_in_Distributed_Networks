import socket
import pickle
import gzip
import io
import torch
import threading
from tqdm import tqdm
from datetime import datetime

TIMEOUT = 300  # Timeout in seconds (5 minutes)
PORT_RECEIVE = 12345
PORT_SEND = 12346
NUM_CLIENTS = 2

received_models = []
lock = threading.Lock()

def decompress_data(data):
    buffer = io.BytesIO(data)
    with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
        return pickle.loads(f.read())

def compress_data(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write(pickle.dumps(data))
    return buffer.getvalue()

def receive_chunked_data(conn):
    print(f"[SERVER] Starting to receive chunked data at {datetime.now()}")
    size_data = b""
    while b"\n" not in size_data:
        size_data += conn.recv(1)
    total_size = int(size_data.decode().split("\n")[0])
    data = b""
    with tqdm(total=total_size, desc="Receiving data", unit="B", unit_scale=True) as pbar:
        while len(data) < total_size:
            chunk = conn.recv(min(4 * 1024 * 1024, total_size - len(data)))
            if not chunk:
                break
            data += chunk
            pbar.update(len(chunk))
    conn.sendall(b"RECEIVED")
    print(f"[SERVER] Finished receiving chunked data at {datetime.now()}")
    return data

def send_chunked_data(conn, data, chunk_size=1024 * 1024):
    print(f"[SERVER] Starting to send chunked data at {datetime.now()}")
    conn.send(str(len(data)).encode() + b"\n")
    for i in range(0, len(data), chunk_size):
        conn.sendall(data[i:i + chunk_size])
    conn.shutdown(socket.SHUT_WR)
    ack = conn.recv(1024).decode()
    print(f"[SERVER] Finished sending chunked data at {datetime.now()}")
    return ack == "RECEIVED"

def handle_client_recv(conn, addr, client_id):
    conn.settimeout(TIMEOUT)
    print(f"[RECV] Connected to client {client_id}: {addr} at {datetime.now()}")
    data = receive_chunked_data(conn)
    model = decompress_data(data)
    with lock:
        received_models.append(model)
    conn.close()
    print(f"[RECV] Client {client_id} model received at {datetime.now()}")

def aggregate_models():
    print(f"[SERVER] Starting model aggregation at {datetime.now()}")
    if len(received_models) != NUM_CLIENTS:
        print(f"[ERROR] Expected {NUM_CLIENTS} models, but received {len(received_models)} at {datetime.now()}")
        return None
    base_model = received_models[0]
    for key in base_model:
        for i in range(1, len(received_models)):
            base_model[key] += received_models[i][key]
        base_model[key] /= len(received_models)
    torch.save(base_model, 'ddos_distilbert_model.pth')
    print(f"[SERVER] Aggregated model saved to ddos_distilbert_model.pth at {datetime.now()}")
    return base_model

def send_aggregated_model(aggregated_model):
    if aggregated_model is None:
        print(f"[ERROR] No aggregated model to send at {datetime.now()}")
        return
    compressed = compress_data(aggregated_model)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.settimeout(TIMEOUT)
    server_sock.bind(('localhost', PORT_SEND))
    server_sock.listen(NUM_CLIENTS)
    print(f"[SERVER] Sending model on port {PORT_SEND} at {datetime.now()}")

    clients_served = 0
    max_attempts = 5
    while clients_served < NUM_CLIENTS:
        try:
            client_sock, addr = server_sock.accept()
            client_sock.settimeout(TIMEOUT)
            print(f"[SEND] Sending to client {clients_served}: {addr} at {datetime.now()}")
            success = send_chunked_data(client_sock, compressed)
            if success:
                print(f"[SEND] Client {clients_served} confirmed receipt at {datetime.now()}")
                clients_served += 1
            else:
                print(f"[ERROR] Client {clients_served} did not confirm receipt at {datetime.now()}")
            client_sock.close()
        except Exception as e:
            print(f"[ERROR] Sending to client {clients_served}: {e} at {datetime.now()}")
            if max_attempts <= 0:
                print(f"[ERROR] Max attempts reached for sending to clients at {datetime.now()}")
                break
            max_attempts -= 1
            print(f"[SERVER] Retrying to send to remaining clients, attempts left: {max_attempts} at {datetime.now()}")
    server_sock.close()
    print(f"[SERVER] Finished sending aggregated model to {clients_served} clients at {datetime.now()}")

def main():
    print(f"[SERVER] Starting server at {datetime.now()}")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.settimeout(TIMEOUT)
    server_sock.bind(('localhost', PORT_RECEIVE))
    server_sock.listen(NUM_CLIENTS)
    print(f"[SERVER] Listening on port {PORT_RECEIVE} for model upload at {datetime.now()}")

    threads = []
    for i in range(NUM_CLIENTS):
        conn, addr = server_sock.accept()
        thread = threading.Thread(target=handle_client_recv, args=(conn, addr, i))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    aggregated_model = aggregate_models()
    send_aggregated_model(aggregated_model)
    server_sock.close()
    print(f"[SERVER] Server shutdown at {datetime.now()}")

if __name__ == "__main__":
    main()