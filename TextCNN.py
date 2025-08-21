import torch
import pandas as pd
import numpy as np
import ast
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated.")

#读取数据
train_df=pd.read_csv('../Dataset/train_set.csv', sep='\t')
test_df=pd.read_csv("../Dataset/test_a.csv", sep='\t')

#数据预处理，把文本字符串转为数字列表,因为深度学习模型只能处理数字，不能直接读懂“文字”和“字符串”
def str_to_list(text):
    return [int(i) for i in text.split()]
train_df['text'] = train_df['text'].apply(str_to_list)
test_df['text'] = test_df['text'].apply(str_to_list)

MAX_SEQ_LENGTH = 1024

#接下来我们可以创建数据集（Dataset）和数据加载器（DataLoader）了
class NewsDataset(Dataset):
    def __init__(self,texts,labels=None,max_seq_len=1024):
        self.texts = texts
        self.labels = labels
        self.max_seq_len = max_seq_len
    def __len__(self):
        return len(self.texts)

    def __getitem__(self,idx):
        tokens=self.texts[idx]

        #填充或者截断
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        input_ids=torch.tensor(tokens,dtype=torch.long)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx],dtype=torch.long)
            return input_ids,label
        else:
            return input_ids

#自此，已经成功创建PyTorch模型期望的输入格式
#接下来是整个过程最核心的一部分，构建模型：TextCNN


class TextCNN(nn.Module):
    def __init__(self,vocab_size,embed_dim=128,num_classes=14,
                 kernel_sizes=[3,4,5],num_filters=128): #卷积核数量为128，尺寸为3,4,5
        super(TextCNN,self).__init__()

        #1.词嵌入层
        self.embedding=nn.Embedding(num_embeddings=vocab_size,   #词汇表大小
                                    embedding_dim=embed_dim,    #嵌入维度，每个词用128维的向量表示
                                    padding_idx=0,              #ID为0的padding不参与训练，因此设为0
        )

        #2.多尺寸卷积层
        self.convs=nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,      #输入通道：词向量维度
                out_channels=num_filters,   #输出通道：卷积核数量
                kernel_size=k               #卷积核大小
            )for k in kernel_sizes          #卷积核数量为3,4,5
        ])

        #3.分类头
        self.dropout=nn.Dropout(p=0.5)    #防止过拟合，随机丢弃50%的节点
        self.fc=nn.Linear(len(kernel_sizes)*num_filters,num_classes)    #3*128-->14


    #模型前向传播,计算流程：输入 → Embedding → Conv1D ×3 → MaxPool → 拼接 → 全连接 → 输出类别
    def forward(self,x):   #x:[B.L] = [batch_size, seq_len] = [16, 1024]
        #第一步：词嵌入
        x=self.embedding(x)    #[B,L]-->[B,L,D] = [batch_size,seq_len,dim]=[16,1024,128]

        #第二步：转换为Conv1d适应的输入格式
        x=x.transpose(1,2)   #[B,L,D]-->[B,D,L] = [16,128,1024]

        #第三步：多个卷积+池化
        pooled_outputs=[]
        for conv in self.convs:
            conv_out=conv(x)          #[B,D,L]-->[B,F,L-k+1]=[16,128,1024-3+1]
            pooled=torch.max(conv_out,dim=2)[0]     #全局最大池化-->[B,F]
            pooled_outputs.append(pooled)    #每个卷积核输出一个特征向量

        #第四步：拼接多个卷积核的输出
        cat=torch.cat(pooled_outputs,dim=1)   #[B,F]-->[B,F*3]=[16,384]

        #第五步：Dropout+分类
        logits=self.fc(self.dropout(cat))    #[B,F*3]-->[B,14]
        return logits

#初始化模型：
VOCAB_SIZE = 7550

#划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].values,
    train_df['label'].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df['label']  # 保持类别平衡
)

#创建 dataset 和 dataloader
BATCH_SIZE = 16

train_dataset = NewsDataset(train_texts, train_labels, max_seq_len=MAX_SEQ_LENGTH)
val_dataset = NewsDataset(val_texts, val_labels, max_seq_len=MAX_SEQ_LENGTH)
test_dataset = NewsDataset(test_df['text'].values, labels=None, max_seq_len=MAX_SEQ_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


#初始化模型,设备，损失函数，优化器
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=TextCNN(vocab_size=VOCAB_SIZE,
              embed_dim=128,
              num_classes=14,
              kernel_sizes=[3,4,5],
              num_filters=128
)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

#训练函数：
def train(model,data_loader,optimizer,criterion,device):
    model.train()
    total_loss=0
    correct=0
    total=0
    for batch_idx, (input_ids, labels) in enumerate(data_loader):
        input_ids, labels = input_ids.to(device), labels.to(device)

        #前向传播
        logits = model(input_ids)
        loss = criterion(logits, labels)

        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #统计结果
        total_loss += loss.item()
        pred=torch.argmax(logits,dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        #每100个batch打印一次
        if (batch_idx+1) % 100 == 0:
            print(f"Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}")

    avg_loss=total_loss / len(data_loader)
    accuracy = correct / total
    print(f'Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

#验证函数
def evaluate(model,data_loader,device):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for input_ids,labels in data_loader:
            input_ids=input_ids.to(device)
            labels=labels.to(device)
            logits=model(input_ids)
            pred=torch.argmax(logits,dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct/total

#开始训练
EPOCHS=4    #循环选取最优模型
best_val_acc=0.0
save_path='TextCNN_model.pth'
print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    print(f"\n{'=' * 20} Epoch {epoch} / {EPOCHS} {'=' * 20}")

    # 训练
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)

    # 验证
    val_acc = evaluate(model, val_loader, device)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # 学习率调度
    scheduler.step(val_acc)

    # 保存最佳模型
    if val_acc > best_val_acc:
       best_val_acc = val_acc
       torch.save(model.state_dict(), save_path)
       print(f"最佳模型已保存至: {save_path} | 验证准确率: {best_val_acc:.4f}")

print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")


#加载最佳模型，对测试集预测
print("加载最佳模型...")
model.load_state_dict(torch.load(save_path, weights_only=True))
model.eval()

predictions= []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch.to(device)
        logits = model(input_ids)
        pred = torch.argmax(logits, dim=1)
        predictions.extend(pred.cpu().tolist())

submission=pd.DataFrame({'label': predictions})
submission.to_csv('TextCNN_submission.csv', index=False)