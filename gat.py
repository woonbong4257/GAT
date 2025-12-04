import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

# 1. DAG 데이터 모델링 (이미지 구조 반영)
# 노드 1~7을 0~6 인덱스로 매핑 (PyTorch는 0부터 시작)
# 매핑: 1->0, 2->1, 3->2, 4->3, 5->4, 6->5, 7->6

mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
reverse_mapping = {v:k for k,v in mapping.items()}

# 엣지 정의 (Source -> Target) : 이미지의 화살표 방향
# 1->2, 2->3, 3->4, 1->5, 5->6, 4->6, 6->7
edge_index = torch.tensor([
    [mapping[1], mapping[2]], # 1 -> 2
    [mapping[2], mapping[3]], # 2 -> 3
    [mapping[3], mapping[4]], # 3 -> 4
    [mapping[1], mapping[5]], # 1 -> 5
    [mapping[5], mapping[6]], # 5 -> 6
    [mapping[4], mapping[6]], # 4 -> 6 (중요: 병합 지점)
    [mapping[6], mapping[7]]  # 6 -> 7
], dtype=torch.long).t().contiguous()

# 노드 특징(Feature) 정의
# 예제를 위해 각 노드에 8차원의 랜덤 값을 부여합니다.
# 실험 시 각 노드의 특징 (예시: 작업 소요시간, 필요한 자원량, 작업 우선순위 등..)
# 데이터프레임을 이용해서 텐서 변환
num_nodes = 7
num_features = 4
x = torch.randn((num_nodes, num_features))

# PyG 데이터 객체 생성
data = Data(x=x, edge_index=edge_index)

# ---------------------------------------------------------
# 2. GAT 모델 정의
# ---------------------------------------------------------
class SimpleGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGAT, self).__init__()
        # heads=1로 설정하여 직관적으로 가중치 하나만 확인 (멀티헤드 가능)
        # concat=False: 출력 차원 유지
        self.conv1 = GATConv(in_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # return_attention_weights=True로 설정하여 중요도(alpha) 반환
        # out: 노드 임베딩 결과, alpha: (edge_index, attention_weights) 튜플
        out, (edge_index_out, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return out, edge_index_out, alpha

# 모델 초기화
model = SimpleGAT(in_channels=num_features, out_channels=4)

# ---------------------------------------------------------
# 3. 모델 실행 및 중요도 확인
# ---------------------------------------------------------
model.eval()
with torch.no_grad():
    out, edges_out, attention_weights = model(data.x, data.edge_index)

print("\n=== Attention Weights (Edge Importance) ===")
# attention_weights는 각 엣지에 대한 중요도 (0~1 사이 값, 보통 Softmax 됨)
for i in range(edges_out.shape[1]):
    src = edges_out[0][i].item()
    dst = edges_out[1][i].item()
    weight = attention_weights[i].item()
    
    # 노드 6(Index 5)으로 들어오는 정보가 중요하므로 이를 눈여겨봄
    print(f"Node {reverse_mapping[src]} -> Node {reverse_mapping[dst]} : Importance {weight:.4f}")

# ---------------------------------------------------------
# 4. 결과 시각화
# ---------------------------------------------------------
def visualize_graph(edge_index, attention_weights, title="GAT Attention Weights"):
    G = nx.DiGraph()
    
    # 엣지 추가 및 가중치 저장
    edge_list = edge_index.t().tolist()
    weights = attention_weights.flatten().tolist()
    
    for (src, dst), w in zip(edge_list, weights):
        # 시각화할 때 원래 노드 번호(1~7)로 변환
        G.add_edge(reverse_mapping[src], reverse_mapping[dst], weight=w)

    pos = nx.spring_layout(G, seed=42)  # 레이아웃 고정
    
    plt.figure(figsize=(10, 6))
    
    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    
    # 엣지 그리기 (가중치에 따라 두께/색상 다르게 표현 가능)
    # 여기서는 엣지 옆에 가중치(중요도)를 숫자로 표기
    nx.draw_networkx_edges(G, pos, width=2, arrowstyle='->', arrowsize=20)
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title(title)
    plt.axis('off')
    plt.show()

# 시각화 실행
visualize_graph(edges_out, attention_weights)