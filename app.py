import streamlit as st
import torch
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import AttentionalAggregation, GCNConv, global_mean_pool
from typing import Optional

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = Path("processed_data")
MODEL_PATH = Path("best_gnn_model2.pth")
GENES_PATH = Path("top_genes_list.txt")

# Load data
def load_data():
    test_dataset = torch.load(DATA_DIR / "test_dataset.pt", weights_only=False)
    edge_index = torch.load(DATA_DIR / "edge_index.pt", weights_only=False)
    class_mapping = np.load(DATA_DIR / "class_mapping.npy", allow_pickle=True).item()
    with open(GENES_PATH, "r", encoding="utf-8") as f:
        genes = [line.strip() for line in f if line.strip()]
    return test_dataset, edge_index, class_mapping, genes

test_dataset, edge_index, class_mapping, genes = load_data()

# Model definition
class GNN_Model(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        num_hidden_units = 64

        self.conv0 = GCNConv(num_node_features, num_hidden_units)
        self.conv1 = GCNConv(num_hidden_units, num_hidden_units)
        self.conv2 = GCNConv(num_hidden_units, num_hidden_units)
        self.conv3 = GCNConv(num_hidden_units, num_hidden_units)

        self.pool0 = AttentionalAggregation(Linear(num_hidden_units, 1))
        self.pool1 = AttentionalAggregation(Linear(num_hidden_units, 1))
        self.pool2 = AttentionalAggregation(Linear(num_hidden_units, 1))
        self.pool3 = AttentionalAggregation(Linear(num_hidden_units, 1))

        self.lin = Linear(num_hidden_units, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv0(x, edge_index).relu()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        graph_emb = self.pool3(x, batch)
        graph_emb = F.dropout(graph_emb, p=0.5, training=self.training)
        return self.lin(graph_emb)

# Load model
def load_model():
    model = GNN_Model(num_node_features=1, num_classes=len(class_mapping)).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_model()

# Helper functions (copied from notebook)
def _stabilize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return eps * torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

def lrp_linear(x: torch.Tensor,
               weight: torch.Tensor,
               R_out: torch.Tensor,
               gamma: float = 0.0,
               eps: float = 1e-6) -> torch.Tensor:
    Wp = weight + gamma * weight.clamp(min=0.0)
    z = x @ Wp.t()
    s = R_out / (z + _stabilize(z, eps))
    R_in = x * (s @ Wp)
    return R_in

def build_dense_gcn_matrix(edge_index: torch.Tensor, num_nodes: int, dtype=torch.float32, device=None) -> torch.Tensor:
    if device is None:
        device = edge_index.device
    edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=device)
    norm_edge_index, norm_edge_weight = gcn_norm(
        edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        add_self_loops=True,
        dtype=dtype,
    )
    A = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    A.index_put_((norm_edge_index[0], norm_edge_index[1]), norm_edge_weight, accumulate=True)
    return A

def get_gcn_weight(conv: GCNConv) -> torch.Tensor:
    if hasattr(conv, "lin") and hasattr(conv.lin, "weight"):
        return conv.lin.weight
    if hasattr(conv, "weight"):
        return conv.weight
    raise AttributeError("Could not find GCNConv weight parameter.")

def lrp_gcn_layer(x: torch.Tensor,
                  weight: torch.Tensor,
                  adj_norm: torch.Tensor,
                  R_out: torch.Tensor,
                  gamma: float = 0.1,
                  eps: float = 1e-6) -> torch.Tensor:
    Wp = weight + gamma * weight.clamp(min=0.0)
    support = x @ Wp.t()
    z = adj_norm @ support
    s = R_out / (z + _stabilize(z, eps))
    R_support = support * (adj_norm.t() @ s)
    t = R_support / (support + _stabilize(support, eps))
    R_in = x * (t @ Wp)
    return R_in

@torch.no_grad()
def forward_with_cache(model: GNN_Model, data: Data):
    model.eval()
    data = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    adj_norm = build_dense_gcn_matrix(data.edge_index, data.num_nodes, dtype=data.x.dtype, device=device)

    x = data.x
    z1 = model.conv0(x, data.edge_index).relu()
    z2 = model.conv1(z1, data.edge_index).relu()
    z3 = model.conv2(z2, data.edge_index).relu()
    z4 = model.conv3(z3, data.edge_index).relu()
    graph_emb = model.pool3(z4, batch)
    graph_logits = model.lin(F.dropout(graph_emb, p=0.5, training=False))

    cache = {
        "x0": x.detach(),
        "z1": z1.detach(),
        "z2": z2.detach(),
        "z3": z3.detach(),
        "z4": z4.detach(),
        "graph_emb": graph_emb.detach(),
        "graph_logits": graph_logits.detach(),
        "batch": batch.detach(),
        "adj_norm": adj_norm.detach(),
    }
    return cache

def explain_sample_lrp(model: GNN_Model, data: Data, target_class=None, gamma: float = 0.1, eps: float = 1e-6):
    model.eval()
    data = data.to(device)
    cache = forward_with_cache(model, data)

    graph_logits = cache["graph_logits"]
    probs = torch.softmax(graph_logits, dim=-1)[0]

    if target_class is None:
        target_class = int(probs.argmax().item())

    graph_score = float(graph_logits[0, target_class].item())

    R_graph = torch.zeros_like(graph_logits)
    R_graph[0, target_class] = graph_logits[0, target_class]

    R_graph_emb = lrp_linear(cache["graph_emb"], model.lin.weight, R_graph, gamma=0.0, eps=eps)

    R_z4 = R_graph_emb / data.num_nodes  # approximate mean pool inverse

    R_z3 = lrp_gcn_layer(cache["z3"], get_gcn_weight(model.conv3), cache["adj_norm"], R_z4, gamma=gamma, eps=eps)

    R_z2 = lrp_gcn_layer(cache["z2"], get_gcn_weight(model.conv2), cache["adj_norm"], R_z3, gamma=gamma, eps=eps)

    R_z1 = lrp_gcn_layer(cache["z1"], get_gcn_weight(model.conv1), cache["adj_norm"], R_z2, gamma=gamma, eps=eps)

    R_x = lrp_gcn_layer(cache["x0"], get_gcn_weight(model.conv0), cache["adj_norm"], R_z1, gamma=gamma, eps=eps)

    input_relevance = R_x.squeeze(-1).detach().cpu()

    conservation_ratio = float(input_relevance.sum().item() / (graph_score + 1e-12))

    return {
        "cache": cache,
        "probs": probs.detach().cpu(),
        "target_class": target_class,
        "graph_score": graph_score,
        "input_relevance": input_relevance,
        "conservation_ratio": conservation_ratio,
    }

def predict_graph(model, data: Data):
    model.eval()
    data = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    graph_logits = model(data.x, data.edge_index, batch)
    probs = torch.softmax(graph_logits, dim=-1)[0]
    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())
    return {
        "graph_logits": graph_logits.detach(),
        "probs": probs.detach(),
        "pred_class": pred_class,
        "confidence": confidence,
    }

st.write("Model loaded")

# --- INTERFEJS UŻYTKOWNIKA (Streamlit UI) ---

st.title(" GNN Cancer XAI Explorer")
st.markdown("Interpretacja wyników modelu GNN za pomocą **LRP** (Layer-wise Relevance Propagation)")

# 1. Wybór próbki
st.sidebar.header("Ustawienia analizy")
sample_idx = st.sidebar.slider("Wybierz indeks próbki z zestawu testowego", 0, len(test_dataset)-1, 0)

# Pobranie danych dla wybranej próbki
data = test_dataset[sample_idx]

# 2. Przewidywanie
if st.button("Uruchom klasyfikację i LRP"):
    with st.spinner('Analizowanie grafu...'):
        # Predykcja
        res = predict_graph(model, data)
        pred_label = class_mapping[res["pred_class"]]
        
        # Wyjaśnienie LRP
        explanation = explain_sample_lrp(model, data, target_class=res["pred_class"])
        relevance = explanation["input_relevance"]
        
        # Wyświetlanie wyników
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Wynik modelu")
            st.success(f"Przewidziana klasa: **{pred_label}**")
            st.metric("Pewność (Confidence)", f"{res['confidence']:.2%}")
            
        with col2:
            st.subheader("Statystyki LRP")
            st.info(f"Ratio konserwacji: {explanation['conservation_ratio']:.4f}")

        # 3. Wyświetlanie top genów (najważniejszych według LRP)
        st.divider()
        st.subheader(" Najważniejsze geny dla tej decyzji")
        
        # Tworzenie tabeli z wynikami istotności
        gene_importance = pd.DataFrame({
            "Gen": genes,
            "Relevance Score": relevance.numpy()
        }).sort_values(by="Relevance Score", ascending=False)

        st.dataframe(gene_importance.head(10), use_container_width=True)

        # 4. Wizualizacja grafu (Ulepszona)
        st.subheader("🕸️ Topologia Istotności (Top 50 genów)")
        
        top_n = 50
        # Wybieramy indeksy top genów
        top_gene_indices = np.argsort(np.abs(relevance))[-top_n:]
        top_gene_indices_set = set(int(i) for i in top_gene_indices)
        
        G = nx.Graph()
        
        # Dodajemy krawędzie
        edge_list = data.edge_index.t().numpy()
        for edge in edge_list:
            u, v = int(edge[0]), int(edge[1])
            if u in top_gene_indices_set and v in top_gene_indices_set:
                G.add_edge(u, v)
        
        # Dodajemy węzły (również te izolowane)
        for idx in top_gene_indices:
            if idx not in G.nodes():
                G.add_node(int(idx))
        
        # --- KLUCZ DO WYGLĄDU: Spring Layout ---
        # k steruje odległością między węzłami (zwiększ, jeśli jest za ciasno)
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Przygotowanie danych do Plotly
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Linie krawędzi (subtelne, szare)
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none', mode='lines'
        )

        # Węzły
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        for node_idx in G.nodes():
            x, y = pos[node_idx]
            node_x.append(x)
            node_y.append(y)
            
            rel_val = float(relevance[node_idx])
            gene_name = genes[node_idx]
            
            node_text.append(gene_name) # Wyświetlimy to jako etykietę
            node_color.append(rel_val)
            
            # SKALOWANIE ROZMIARU: Im ważniejszy gen, tym większa kropka
            # Używamy pierwiastka lub logarytmu, żeby różnice nie były zbyt drastyczne
            size_base = np.abs(rel_val) * 1000 
            node_size.append(max(8, min(size_base, 25))) # Zakres od 8 do 25

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text', # POKAZUJE TEKST OD RAZU
            text=node_text,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            hoverinfo='text',
            hovertext=[f"Gen: {n}<br>LRP: {c:.4f}" for n, c in zip(node_text, node_color)],
            marker=dict(
                showscale=True,
                colorscale='YlOrRd', # Od żółtego do czerwonego (kojarzy się z "hotspots")
                reversescale=False,
                size=node_size,
                color=node_color,
                line_width=2,
                colorbar=dict(
                    thickness=15,
                    title='Wpływ (LRP)',
                    xanchor='left'
                )
            )
        )

        # Tworzenie figury
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)', # Przezroczyste tło
                        paper_bgcolor='rgba(0,0,0,0)',
                        annotations=[dict(
                            text="Wielkość węzła = Absolutna istotność LRP",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            font=dict(size=10, color="gray")
                        )]
                    )
                )
        
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Kliknij przycisk powyżej, aby przeanalizować wybraną próbkę.")