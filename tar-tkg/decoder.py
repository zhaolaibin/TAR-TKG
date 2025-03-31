import random

from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
path_dir = os.getcwd()

class TimeConvTransR(nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0.1, hidden_dropout=0.1, feature_map_dropout=0.1, channels=50, kernel_size=3):
        super(TimeConvTransR, self).__init__()

        self.inp_drop = nn.AlphaDropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)

        self.conv1 = nn.Conv1d(4, channels, kernel_size, padding=kernel_size // 2)
        self.bn0 = nn.LayerNorm(embedding_dim)  # 适用于 batch_size=1
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.LayerNorm(embedding_dim)  # 适用于 batch_size=1
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)

        self.register_parameter('b', nn.Parameter(torch.zeros(num_relations * 2)))  # 关系偏置

    def forward(self, embedding, emb_rel, emb_time, triplets, partial_embedding=None):
        """
        :param embedding: 实体嵌入矩阵 (num_entities, embedding_dim)
        :param emb_rel: 关系嵌入矩阵 (num_relations, embedding_dim)
        :param emb_time: (t1, t2)，两个时间步的时间嵌入
        :param triplets: (batch_size, 3)，(head, relation, tail)
        :param partial_embedding: 可选，用于部分评分计算
        """
        batch_size = len(triplets)
        head, relation, tail = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        e1_embedded = torch.tanh(embedding[head]).unsqueeze(1)
        e2_embedded = torch.tanh(embedding[tail]).unsqueeze(1)
        emb_time_1, emb_time_2 = emb_time
        emb_time_1, emb_time_2 = emb_time_1.unsqueeze(1), emb_time_2.unsqueeze(1)

        # 拼接实体和时间信息
        stacked_inputs = torch.stack([e1_embedded, e2_embedded, emb_time_1, emb_time_2], dim=1)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 计算得分
        score = torch.einsum('bd,nd->bn', x, emb_rel)

        if partial_embedding is not None:
            score = score * partial_embedding

        return score


class TimeConvTransE(nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0.1, hidden_dropout=0.1, feature_map_dropout=0.1, channels=50, kernel_size=3):
        super(TimeConvTransE, self).__init__()

        self.inp_drop = nn.AlphaDropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)

        self.conv1 = nn.Conv1d(4, channels, kernel_size, padding=kernel_size // 2)
        self.bn0 = nn.LayerNorm(embedding_dim)  # 适用于 batch_size=1
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.LayerNorm(embedding_dim)  # 适用于 batch_size=1
        self.fc = nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, emb_time, triplets):
        head, relation, tail = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        
        e1_embedded = torch.tanh(embedding[head]).unsqueeze(1)
        rel_embedded = emb_rel[relation].unsqueeze(1)
        emb_time_1, emb_time_2 = emb_time
        emb_time_1, emb_time_2 = emb_time_1.unsqueeze(1), emb_time_2.unsqueeze(1)

        stacked_inputs = torch.stack([e1_embedded, rel_embedded, emb_time_1, emb_time_2], dim=1)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        return torch.einsum('bd,nd->bn', x, embedding)