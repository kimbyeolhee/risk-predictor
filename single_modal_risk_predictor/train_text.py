import argparse
from omegaconf import OmegaConf
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from typing import Dict, List

class MedicalReportDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_chunk_length: int = 512, max_chunks: int = 10):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length
        self.max_chunks = max_chunks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and chunk the text
        chunks = []
        for i in range(0, len(text), self.max_chunk_length):
            chunk = text[i:i + self.max_chunk_length]
            encoded_chunk = self.tokenizer.encode_plus(
                chunk,
                add_special_tokens=True,
                max_length=self.max_chunk_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            chunks.append({
                'input_ids': encoded_chunk['input_ids'].squeeze(),
                'attention_mask': encoded_chunk['attention_mask'].squeeze()
            })

        # Pad or truncate to max_chunks
        if len(chunks) < self.max_chunks:
            padding = {
                'input_ids': torch.zeros(self.max_chunk_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_chunk_length, dtype=torch.long)
            }
            chunks.extend([padding] * (self.max_chunks - len(chunks)))
        elif len(chunks) > self.max_chunks:
            chunks = chunks[:self.max_chunks]

        return {
            'input_ids': torch.stack([chunk['input_ids'] for chunk in chunks]),
            'attention_mask': torch.stack([chunk['attention_mask'] for chunk in chunks]),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
class HierarchicalBert(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.num_labels = num_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        batch_size, num_chunks, seq_length = input_ids.shape # torch.Size([2, 10, 512])
        input_ids = input_ids.view(-1, seq_length) # torch.Size([20, 512])
        attention_mask = attention_mask.view(-1, seq_length) # torch.Size([20, 512])

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # torch.size([20, 512, 768])
        chunk_embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding # torch.Size([20, 768])
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1) # torch.Size([2, 10, 768])

        lstm_out, _ = self.lstm(chunk_embeddings) # torch.Size([2, 10, 512])
        document_embedding = lstm_out[:, -1, :]  # Use last hidden state # torch.Size([2, 512])

        logits = self.classifier(document_embedding) # torch.Size([2, 2])

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = softmax(logits, axis=1)[:, 1]
    predictions = np.argmax(probabilities, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc
    }

def main(cfg):
    text_column =  '검사결과본문내용'
    label_column = 'Death'

    df = pd.read_csv(cfg.dataset.data_path)
    df['Death'] = df['Death'].fillna(0)

    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = HierarchicalBert(num_labels=2)

    dataset = MedicalReportDataset(texts, labels, tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./train_text_modality_only_logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_text_modal_config")
    args , _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./configs/{args.config}.yaml")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main(cfg)