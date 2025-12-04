import os
import sys

# =========================================================
# [중요] SSL/DLL 로드 오류 해결을 위한 경로 강제 지정
# =========================================================
# 사용자의 Conda 환경 경로에 맞춰 Library\bin을 최우선으로 등록
conda_lib_path = r"C:\conda3\envs\GAT\Library\bin"
os.environ['PATH'] = conda_lib_path + ";" + os.environ['PATH']

import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# 1. 데이터 생성 (튜토리얼 학습용 데이터)
# 7개의 작업(Node)에 대한 4가지 속성(Features) 정의
# 속성: [소요시간(분), CPU코어수, 데이터크기(GB), 중요도(1~5)]
data_source = {
    'task_name': ['Data Load', 'Pre-process', 'Feature Eng', 'Model Train', 'Data Validation', 'Model Eval', 'Deployment'],
    'duration_min': [10, 30, 45, 120, 15, 20, 5],
    'cpu_cores':    [2,  4,  8,  16,  2,  4,  1],
    'data_size_gb': [50, 45, 60, 60,  50, 10, 1],
    'priority':     [5,  3,  4,  5,   2,  5,  5]
}

# Pandas로 보기 좋게 변환
df = pd.DataFrame(data_source)
print("=== Input Node Features (Task Data) ===")
print(df[['task_name', 'duration_min', 'cpu_cores', 'data_size_gb', 'priority']])
print("-" * 60)

# 모델 학습용 Tensor로 변환 (숫자 데이터 4개 컬럼만 사용)
feature_cols = ['duration_min', 'cpu_cores', 'data_size_gb', 'priority']
x = torch.tensor(df[feature_cols].values, dtype=torch.float)

# 노드 개수 및 특징 차원 확인
num_nodes = x.shape[0]      # 7
num_features = x.shape[1]   # 4


# 2. 그래프 구조 정의 (DAG 모델링)
# 노드 번호 매핑 (User: 1~7 -> Code: 0~6)
mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
reverse_mapping = {v:k for k,v in mapping.items()} # 0->1 변환용

edge_index = torch.tensor([
    [mapping[1], mapping[2]], 
    [mapping[2], mapping[3]], 
    [mapping[3], mapping[4]], 
    [mapping[1], mapping[5]], 
    [mapping[5], mapping[6]], 
    [mapping[4], mapping[6]], 
    [mapping[6], mapping[7]]
], dtype=torch.long).t().contiguous()

# PyG 데이터 객체 생성
data = Data(x=x, edge_index=edge_index)

# 3. GAT 모델 정의

class SimpleGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # return_attention_weights=True: 중요도 점수 반환
        out, (edge_index_out, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return out, edge_index_out, alpha

# 모델 생성
model = SimpleGAT(in_channels=num_features, out_channels=4)


# 4. 실행 및 결과 출력

model.eval()
with torch.no_grad():
    out, edges_out, attention_weights = model(data.x, data.edge_index)

print("\n=== Attention Weights Analysis ===")
# 텍스트로 중요도 출력
edge_list = edges_out.t().tolist()
weights = attention_weights.flatten().tolist()

for (src, dst), w in zip(edge_list, weights):
    src_name = f"Node {reverse_mapping[src]}"
    dst_name = f"Node {reverse_mapping[dst]}"
    
    # 자기 자신(Self-loop)인 경우 별도 표기
    if src == dst:
        print(f"[{dst_name} Self-Reflect] Retention Rate : {w:.4f}")
    else:
        print(f"[{src_name} -> {dst_name}] Importance     : {w:.4f}")


# 5. 시각화 (NetworkX)
def visualize_graph():
    G = nx.DiGraph()
    
    # 그래프에 엣지 추가 (자기 자신으로 가는 화살표는 그림에서 제외하여 깔끔하게)
    for (src, dst), w in zip(edge_list, weights):
        if src != dst: 
            G.add_edge(reverse_mapping[src], reverse_mapping[dst], weight=w)

    pos = nx.spring_layout(G, seed=42) # 레이아웃 고정
    
    plt.figure(figsize=(12, 7))
    
    # 1. 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # 2. 엣지 그리기
    nx.draw_networkx_edges(G, pos, width=2, arrowstyle='->', arrowsize=25, edge_color='gray')
    
    # 3. 중요도(Weight) 라벨 표시 (빨간색)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10, label_pos=0.3)
    
    plt.title("DAG Task Scheduling - GAT Attention Weights", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_graph()