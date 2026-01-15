import os
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddle.optimizer import AdamW
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.data import JiebaTokenizer, Pad
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import csv

warnings.filterwarnings('ignore')
paddle.seed(2023)


class Config:
    def __init__(self):
        # 数据集路径
        self.data_dir = r"C:\Users\19658\Desktop\数据挖掘项目实践\ChnSentiCorp[dataset 1]\ChnSentiCorp"

        self.epochs = 15
        self.lr = 0.0015
        self.max_seq_len = 128
        self.batch_size = 64
        self.dropout = 0.3
        self.num_class = 2

        # 词向量设置
        self.embedding_name = 'w2v.wiki.target.word-char.dim300'

        # 保存路径
        self.model_save_path = './linear_model.pdparams'
        self.log_dir = './log_linear'


config = Config()

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)


def load_tsv_file(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')

            headers = next(reader)
            num_columns = len(headers)

            if num_columns == 2:
                label_idx = 0
                text_idx = 1
            elif num_columns >= 3:
                label_idx = 1
                text_idx = 2
            else:
                raise ValueError(f"文件列数不足: {file_path} (只有{num_columns}列)")

            for row in reader:
                if len(row) < num_columns:
                    continue

                try:
                    label = int(row[label_idx])
                    text = row[text_idx].strip()
                    if text:
                        data.append((label, text))
                except ValueError:
                    continue
    except Exception as e:
        print(f"读取文件时出错: {file_path}")
        print(f"错误信息: {e}")

    return data


class SentimentDataset(Dataset):
    def __init__(self, file_path, max_seq_len=128):
        super().__init__()
        self.data = load_tsv_file(file_path)
        self.max_seq_len = max_seq_len
        print(f"已加载 {len(self.data)} 条数据: {file_path}")

    def __getitem__(self, idx):
        label, text = self.data[idx]
        return text, label

    def __len__(self):
        return len(self.data)


print("加载数据集...")
train_dataset = SentimentDataset(os.path.join(config.data_dir, 'train.tsv'))
dev_dataset = SentimentDataset(os.path.join(config.data_dir, 'dev.tsv'))
test_dataset = SentimentDataset(os.path.join(config.data_dir, 'test.tsv'))

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(dev_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

print("加载预训练词向量...")
try:
    embedding = TokenEmbedding(
        embedding_name=config.embedding_name,
        unknown_token='[UNK]',
        unknown_token_vector=None,
        extended_vocab_path=None,
        trainable=True,
        keep_extended_vocab_only=False
    )
    print("词向量加载成功!")
except Exception as e:
    print(f"加载词向量失败: {e}")
    print("请确保已下载所需的词向量")
    exit(1)


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = JiebaTokenizer(vocab)
        self.UNK_TOKEN = '[UNK]'
        self.PAD_TOKEN = '[PAD]'
        self.pad_token_id = vocab.token_to_idx.get(self.PAD_TOKEN)

    def text_to_ids(self, text, max_seq_len=128):
        input_ids = []
        unk_token_id = self.vocab[self.UNK_TOKEN]

        tokens = self.tokenizer.cut(text)

        for token in tokens:
            token_id = self.vocab.token_to_idx.get(token, unk_token_id)
            input_ids.append(token_id)

        return input_ids[:max_seq_len]


tokenizer = Tokenizer(embedding.vocab)


def batchify_fn(batch, tokenizer, max_seq_len):
    texts, labels = zip(*batch)

    input_ids = [tokenizer.text_to_ids(text, max_seq_len) for text in texts]

    pad_input_ids = Pad(pad_val=tokenizer.pad_token_id)(input_ids)

    labels = [int(label) for label in labels]

    input_ids_tensor = paddle.to_tensor(pad_input_ids, dtype='int64')
    labels_tensor = paddle.to_tensor(labels, dtype='int64')

    return input_ids_tensor, labels_tensor


train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=lambda batch: batchify_fn(batch, tokenizer, config.max_seq_len)
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=lambda batch: batchify_fn(batch, tokenizer, config.max_seq_len)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=lambda batch: batchify_fn(batch, tokenizer, config.max_seq_len)
)


# 定义线性模型
class LinearModel(nn.Layer):
    def __init__(self, config, embedding):
        super().__init__()
        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim

        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_class)
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        mask = (input_ids != tokenizer.pad_token_id).astype('float32')  # [batch_size, seq_len]
        mask = mask.unsqueeze(2)  # [batch_size, seq_len, 1]
        x = x * mask  # 屏蔽填充部分

        word_count = paddle.sum(mask, axis=1)  # [batch_size, 1]
        word_count = paddle.maximum(word_count, paddle.to_tensor(1.0))  # 避免除以0

        sentence_emb = paddle.sum(x, axis=1) / word_count  # [batch_size, embedding_dim]

        logits = self.fc(sentence_emb)
        return logits


print("创建线性模型...")
model = LinearModel(config, embedding)
print(model)


def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with paddle.no_grad():
        for input_ids, labels in data_loader:
            logits = model(input_ids)
            preds = paddle.argmax(logits, axis=1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='micro')

    model.train()
    return acc, f1


def train(model, train_loader, dev_loader, config):
    optimizer = AdamW(learning_rate=config.lr, parameters=model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    train_losses = []
    val_accs = []
    val_f1s = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.epochs}', leave=True)

        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            logits = model(input_ids)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            epoch_loss += float(loss)
            avg_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        val_acc, val_f1 = evaluate(model, dev_loader)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch}: Loss: {avg_epoch_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            paddle.save(model.state_dict(), config.model_save_path)
            print(f"保存最佳模型，验证集准确率: {val_acc:.4f}")

    print(f"\n训练完成. 最佳验证准确率: {best_acc:.4f}")
    return train_losses, val_accs, val_f1s


print("开始训练线性模型...")
train_losses, val_accs, val_f1s = train(model, train_loader, dev_loader, config)


def test(model, test_loader, config):
    state_dict = paddle.load(config.model_save_path)
    model.set_state_dict(state_dict)

    test_acc, test_f1 = evaluate(model, test_loader)
    print("\n" + "=" * 60)
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集F1分数: {test_f1:.4f}")
    print("=" * 60)
    return test_acc, test_f1


print("在测试集上评估线性模型...")
test_acc, test_f1 = test(model, test_loader, config)

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False
finally:
    plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, config.epochs + 1), val_accs, 'b-o', label='验证集准确率')
plt.plot(range(1, config.epochs + 1), val_f1s, 'r-o', label='验证集F1分数')
plt.title('线性模型在验证集上的性能')
plt.xlabel('训练轮数')
plt.ylabel('分数')
plt.legend()
plt.grid(True)

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, config.epochs + 1), train_losses, 'g-o', label='训练损失')
plt.title('线性模型训练损失变化')
plt.xlabel('训练轮数')
plt.ylabel('损失')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(config.log_dir, 'training_metrics.png'))
plt.show()


# 预测示例
def predict_sentiment(model, text, tokenizer, config):
    # 预处理文本
    input_ids = tokenizer.text_to_ids(text, config.max_seq_len)
    input_ids = paddle.to_tensor([input_ids], dtype='int64')

    # 预测
    model.eval()
    with paddle.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits, axis=1)
        pred = paddle.argmax(probs, axis=1).numpy()[0]
        confidence = probs.numpy()[0][pred]

    sentiment = "正向" if pred == 1 else "负向"
    return sentiment, confidence, probs.numpy()[0]


examples = [
    "酒店环境很好，服务态度也很不错，下次还会来住",
    "房间太小了，而且卫生条件很差，完全不值这个价格",
    "设施比较陈旧，但是位置很方便，总体还可以",
    "早餐种类太少，房间隔音效果差，不会再来了",
    "性价比很高，交通便利，推荐给大家"
]

print("\n情感预测示例:")
for text in examples:
    sentiment, confidence, probs = predict_sentiment(model, text, tokenizer, config)
    print(f"文本: {text}")
    print(f"情感: {sentiment} (置信度: {confidence:.4f}), 概率分布: [负向: {probs[0]:.4f}, 正向: {probs[1]:.4f}]")
    print("-" * 80)

print("\n实验总结:")
print(f"测试集准确率: {test_acc:.4f}")
print(f"测试集F1分数: {test_f1:.4f}")
print(f"最佳验证准确率: {max(val_accs):.4f}")