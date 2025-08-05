import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
from tqdm import tqdm
import socket
import pickle
import os
import gzip
import io
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

TIMEOUT = 300  # Timeout in seconds (5 minutes)
data_fraction_size = 0.1


class CICIDS2017Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': self.labels[idx]
        }


class DDoSClassifier(nn.Module):
    def __init__(self, local_model_path):
        super(DDoSClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(local_model_path)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def features_to_text(row):
    text = (
        f"Destination port is {row['Destination Port']}. "
        f"Flow duration is {row['Flow Duration']} microseconds. "
        f"Total forward packets are {row['Total Fwd Packets']}. "
        f"Total backward packets are {row['Total Backward Packets']}. "
        f"Total length of forward packets is {row['Total Length of Fwd Packets']} bytes. "
        f"Total length of backward packets is {row['Total Length of Bwd Packets']} bytes. "
        f"Maximum forward packet length is {row['Fwd Packet Length Max']}. "
        f"Minimum forward packet length is {row['Fwd Packet Length Min']}. "
        f"Flow bytes per second is {row['Flow Bytes/s']}. "
        f"Flow packets per second is {row['Flow Packets/s']}."
    )
    return text


def preprocess_data(file_path, data_fraction=data_fraction_size):
    print(f"[CLIENT 1] Starting data preprocessing at {datetime.now()}")
    df = pd.read_csv(file_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df = df.sample(frac=data_fraction, random_state=42)
    texts = df.apply(features_to_text, axis=1).tolist()
    labels = df['Label'].apply(lambda x: 1 if x == 'DDoS' else 0).tolist()
    print(f"[CLIENT 1] Finished data preprocessing at {datetime.now()}")
    return texts, labels


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    print(f"[CLIENT 1] Starting model training at {datetime.now()}")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Client 1 Epoch {epoch + 1}/{num_epochs}', unit='batch')
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        avg_loss = running_loss / len(train_loader)
        print(f'Client 1 Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    print(f"[CLIENT 1] Finished model training at {datetime.now()}")


def evaluate_model(model, test_loader, criterion, device):
    print(f"[CLIENT 1] Starting model evaluation at {datetime.now()}")
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    test_bar = tqdm(test_loader, desc='Client 1 Evaluating', unit='batch')
    with torch.no_grad():
        for batch in test_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Client 1 Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}, '
          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    print(f"[CLIENT 1] Finished model evaluation at {datetime.now()}")
    return accuracy, avg_loss, precision, recall, f1, cm, all_labels, all_probs


def plot_evaluation(local_metrics, aggregated_metrics=None, output_dir='client1_plots'):
    print(f"[CLIENT 1] Starting plot generation at {datetime.now()}")
    os.makedirs(output_dir, exist_ok=True)

    def plot_confusion_matrix(cm, title, filename):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f'Client 1 {filename} saved to {output_dir} at {datetime.now()}')

    def plot_roc_curve(labels, probs, title, filename):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f'Client 1 {filename} saved to {output_dir} at {datetime.now()}')

    def plot_precision_recall_curve(labels, probs, title, filename):
        precision, recall, _ = precision_recall_curve(labels, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f'Client 1 {filename} saved to {output_dir} at {datetime.now()}')

    def plot_metrics_comparison(local_metrics, aggregated_metrics, filename):
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        local_values = [local_metrics[0], local_metrics[2], local_metrics[3], local_metrics[4]]
        if aggregated_metrics is not None:
            aggregated_values = [aggregated_metrics[0], aggregated_metrics[2], aggregated_metrics[3],
                                 aggregated_metrics[4]]
            x = np.arange(len(metrics))
            width = 0.35
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width / 2, local_values, width, label='Local Model', color='#1f77b4')
            ax.bar(x + width / 2, aggregated_values, width, label='Aggregated Model', color='#ff7f0e')
            ax.set_title('Client 1: Local vs Aggregated Model Performance')
        else:
            x = np.arange(len(metrics))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x, local_values, label='Local Model', color='#1f77b4')
            ax.set_title('Client 1: Local Model Performance')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f'Client 1 {filename} saved to {output_dir} at {datetime.now()}')

    plot_confusion_matrix(local_metrics[5], 'Client 1 Local Model Confusion Matrix', 'local_confusion_matrix.png')
    if aggregated_metrics is not None:
        plot_confusion_matrix(aggregated_metrics[5], 'Client 1 Aggregated Model Confusion Matrix',
                              'aggregated_confusion_matrix.png')
    plot_metrics_comparison(local_metrics, aggregated_metrics, 'metrics_comparison.png')
    print(f"[CLIENT 1] Finished plot generation at {datetime.now()}")


def compress_data(data):
    print(f"[CLIENT 1] Starting data compression at {datetime.now()}")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write(pickle.dumps(data))
    print(f"[CLIENT 1] Finished data compression at {datetime.now()}")
    return buffer.getvalue()


def decompress_data(compressed_data):
    print(f"[CLIENT 1] Starting data decompression at {datetime.now()}")
    buffer = io.BytesIO(compressed_data)
    with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
        data = pickle.loads(f.read())
    print(f"[CLIENT 1] Finished data decompression at {datetime.now()}")
    return data


def send_chunked_data(client_socket, data, chunk_size=1024 * 1024):
    print(f"[CLIENT 1] Starting to send chunked data at {datetime.now()}")
    total_size = len(data)
    client_socket.send(str(total_size).encode() + b'\n')
    for i in range(0, total_size, chunk_size):
        client_socket.sendall(data[i:i + chunk_size])
    confirmation = client_socket.recv(1024).decode()
    print(f"[CLIENT 1] Finished sending chunked data at {datetime.now()}")
    return confirmation == "RECEIVED"


def receive_chunked_data(client_socket):
    print(f"[CLIENT 1] Starting to receive chunked data at {datetime.now()}")
    size_data = b""
    while b'\n' not in size_data:
        size_data += client_socket.recv(1)
    total_size = int(size_data.decode().split('\n')[0])
    data = b""
    with tqdm(total=total_size, desc="Receiving data", unit="B", unit_scale=True) as pbar:
        while len(data) < total_size:
            packet = client_socket.recv(min(4194304, total_size - len(data)))
            if not packet:
                break
            data += packet
            pbar.update(len(packet))
    client_socket.send("RECEIVED".encode())
    print(f"[CLIENT 1] Finished receiving chunked data at {datetime.now()}")
    return data


def send_model(model, host='localhost', port=12345):
    print(f"[CLIENT 1] Starting to send model to server at {datetime.now()}")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(TIMEOUT)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
        client_socket.connect((host, port))
        print(f'Client 1 connected to server at {host}:{port} at {datetime.now()}')
        model_state_dict = model.state_dict()
        compressed_data = compress_data(model_state_dict)
        if send_chunked_data(client_socket, compressed_data):
            print(f'Client 1 sent model weights to server at {datetime.now()}')
        else:
            print(f'Client 1 failed to receive server confirmation at {datetime.now()}')
        client_socket.close()
        print(f"[CLIENT 1] Finished sending model to server at {datetime.now()}")
        return True
    except Exception as e:
        print(f'Client 1 failed to send model: {e} at {datetime.now()}')
        return False


def wait_for_server(host, port, timeout=TIMEOUT, interval=1):
    import time
    print(f"[CLIENT 1] Starting to wait for server at {host}:{port} at {datetime.now()}")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
            print(f"[CLIENT 1] Server ready at {host}:{port} at {datetime.now()}")
            return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(interval)
    print(f"[CLIENT 1] Server not ready at {host}:{port} after timeout at {datetime.now()}")
    return False


def receive_aggregated_model(host='localhost', port=12346, max_retries=5):
    print(f"[CLIENT 1] Starting to receive aggregated model at {datetime.now()}")
    for attempt in range(max_retries):
        if not wait_for_server(host, port, timeout=TIMEOUT):
            print(
                f'Client 1 failed to connect: server not ready at port {port}, attempt {attempt + 1}/{max_retries} at {datetime.now()}')
            continue
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(TIMEOUT)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)
            client_socket.connect((host, port))
            print(f'Client 1 receiving aggregated model from {host}:{port} at {datetime.now()}')
            compressed_data = receive_chunked_data(client_socket)
            aggregated_state_dict = decompress_data(compressed_data)
            client_socket.close()
            print(f'Client 1 received aggregated model at {datetime.now()}')
            return aggregated_state_dict
        except Exception as e:
            print(
                f'Client 1 failed to receive aggregated model, attempt {attempt + 1}/{max_retries}: {e} at {datetime.now()}')
    print(f'Client 1 failed to receive aggregated model after {max_retries} attempts at {datetime.now()}')
    return None


def save_metrics(metrics, filename='client1_metrics.csv'):
    print(f"[CLIENT 1] Starting to save metrics at {datetime.now()}")
    metrics_dict = {
        'Accuracy': metrics[0],
        'Loss': metrics[1],
        'Precision': metrics[2],
        'Recall': metrics[3],
        'F1-Score': metrics[4]
    }
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filename, index=False)
    print(f'Client 1 metrics saved to {filename} at {datetime.now()}')


def main():
    print(f"[CLIENT 1] Starting client at {datetime.now()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = 'CICIDS2017.csv'
    local_model_path = './distilbert-base-uncased'
    client_model_path = 'client1_model.pth'

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Local model path {local_model_path} does not exist!")

    texts, labels = preprocess_data(file_path, data_fraction=data_fraction_size)
    tokenizer = DistilBertTokenizer.from_pretrained(local_model_path)
    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    train_dataset = CICIDS2017Dataset(X_train, y_train, tokenizer)
    val_dataset = CICIDS2017Dataset(X_val, y_val, tokenizer)
    test_dataset = CICIDS2017Dataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = DDoSClassifier(local_model_path).to(device)
    if os.path.exists(client_model_path):
        print(f'Client 1 loading pre-trained model from {client_model_path} at {datetime.now()}')
        model.load_state_dict(torch.load(client_model_path, map_location=device))

    train_model(model, train_loader, criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(model.parameters(), lr=2e-5), num_epochs=3, device=device)

    print('Client 1 evaluating local model on validation set...')
    val_metrics = evaluate_model(model, val_loader, nn.CrossEntropyLoss(), device)
    print('Client 1 evaluating local model on test set...')
    local_metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
    save_metrics(local_metrics, 'client1_local_metrics.csv')

    torch.save(model.state_dict(), client_model_path)
    print(f'Client 1 model saved to {client_model_path} at {datetime.now()}')

    sent_success = send_model(model)
    if sent_success:
        aggregated_state_dict = receive_aggregated_model()
        if aggregated_state_dict is not None:
            model.load_state_dict(aggregated_state_dict)
            print(f'Client 1 updated with aggregated model at {datetime.now()}')
            print('Client 1 evaluating aggregated model on validation set...')
            aggregated_val_metrics = evaluate_model(model, val_loader, nn.CrossEntropyLoss(), device)
            print('Client 1 evaluating aggregated model on test set...')
            aggregated_metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
            save_metrics(aggregated_metrics, 'client1_aggregated_metrics.csv')
            plot_evaluation(local_metrics, aggregated_metrics)
            torch.save(model.state_dict(), client_model_path)
            print(f'Client 1 updated model saved to {client_model_path} at {datetime.now()}')
        else:
            print(f'Client 1 skipping aggregated model evaluation due to connection failure at {datetime.now()}')
            plot_evaluation(local_metrics)
    else:
        print(f'Client 1 skipping aggregated model evaluation due to send failure at {datetime.now()}')
        plot_evaluation(local_metrics)
    print(f"[CLIENT 1] Client shutdown at {datetime.now()}")


if __name__ == '__main__':
    main()
