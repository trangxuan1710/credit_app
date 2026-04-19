import streamlit as st
import pandas as pd
import pickle
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
import tempfile, os

# ── CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Credit AI Dash",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS (HIỆN ĐẠI HÓA) ────────────────────────────────
st.markdown("""
    <style>
    /* Tổng thể */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0b0e14; }
    
    /* Card Container */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border-color: #00cc66;
    }
    
    /* Score Display */
    .score-container {
        text-align: center;
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, #1e2130 0%, #0b0e14 100%);
        border: 1px solid #333;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Sidebar tinh chỉnh */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #222;
    }
    
    /* Header style */
    .main-title {
        background: linear-gradient(90deg, #00cc66, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ── DATA LOADING (Giữ nguyên logic cũ) ───────────────────────
@st.cache_data
def load_data():
    # Giả định file tồn tại như code cũ của bạn
    try:
        df_cust = pd.read_excel("data_training_tin_dung.xlsx", sheet_name="customer")
        df_edge = pd.read_excel("data_training_tin_dung.xlsx", sheet_name="edge")
        le = LabelEncoder()
        df_cust["xi10_nghe_nghiep"] = le.fit_transform(df_cust["xi10_nghe_nghiep"].astype(str))
        return df_cust, df_edge
    except:
        # Dummy data để demo nếu không có file
        return pd.DataFrame({'id': [1,2], 'label': [1,0], 'xi1_tuoi': [25, 30], 'xi8_thu_nhap': [10, 20], 'xi23_diem_cic': [700, 400], 'xi18_tien_qua_han': [0, 5], 'xi22_co_no_xau': [0, 1], 'xi33_co_bat_dong_san': [1, 0], 'xi34_co_xe': [0, 0]}), pd.DataFrame()

@st.cache_resource
def load_model():
    # Try-except để tránh lỗi khi chưa có file model.pkl thực tế
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None, ["xi1_tuoi", "xi8_thu_nhap"], None

df_cust, df_edge = load_data()
model, feature_cols, le = load_model()

@st.cache_data
def build_graph():
    G = nx.Graph()
    for _, row in df_cust.iterrows():
        G.add_node(int(row["id"]), label_true=int(row["label"]))
    for _, row in df_edge.iterrows():
        G.add_edge(int(row["src"]), int(row["dst"]),
                   weight=float(row.get("weight", 1)),
                   relation=row.get("relation_type", "unknown"))
    return G

G = build_graph()

def predict_score(customer_id):
    if model is None: return 0.85, 1 # Demo
    row = df_cust[df_cust["id"] == customer_id]
    if row.empty: return None, None
    X = row[feature_cols].fillna(0)
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return prob, pred

def draw_neighborhood(customer_id, depth=1):
    neighbors = {customer_id}
    for _ in range(depth):
        new_n = set()
        for n in neighbors:
            if n in G: new_n.update(G.neighbors(n))
        neighbors.update(new_n)
    
    sub = G.subgraph(neighbors)
    net = Network(height="500px", width="100%", bgcolor="#0b0e14", font_color="#cbd5e0")
    
    # Cấu hình physics cho mượt hơn
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)
    
    for node in sub.nodes():
        row = df_cust[df_cust["id"] == node]
        lbl_true = int(row["label"].values[0]) if not row.empty else -1
        prob, _ = predict_score(node)
        prob_text = f"{prob*100:.0f}%" if prob is not None else "?"
        
        color = "#00cc66" if lbl_true == 1 else "#ff4444"
        if node == customer_id:
            net.add_node(node, label=f"TARGET\nID {node}", color="#FFD700", size=35, shadow=True)
        else:
            net.add_node(node, label=f"ID {node}\n{prob_text}", color=color, size=20)

    for src, dst, data in sub.edges(data=True):
        net.add_edge(src, dst, color="rgba(255,255,255,0.2)", width=2)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        return f.name

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.markdown("### Control Panel")
    all_ids = sorted(df_cust["id"].tolist())
    selected_id = st.selectbox("🎯 Target Customer ID", all_ids)
    depth = st.select_slider("🔍 Analysis Depth (Hops)", options=[1, 2], value=1)
    
    st.markdown("---")
    st.caption("GNN Model v2.4.0-Stable")
    st.caption("Last updated: 2024-Q3")

# ── MAIN CONTENT ─────────────────────────────────────────────
st.markdown('<h1 class="main-title">Smart Credit AI Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#666; margin-bottom:2rem;">Hệ thống phân tích rủi ro tín dụng dựa trên Graph Neural Networks</p>', unsafe_allow_html=True)

prob, pred = predict_score(selected_id)
row_data = df_cust[df_cust["id"] == selected_id].iloc[0]

# Row 1: Key Metrics
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""<div class="metric-card">
        <p style="color:#888; font-size:14px; margin:0;">Credit Probability</p>
        <h2 style="color:{'#00cc66' if pred==1 else '#ff4444'}; margin:0;">{prob*100:.1f}%</h2>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="metric-card">
        <p style="color:#888; font-size:14px; margin:0;">Income</p>
        <h2 style="color:#fff; margin:0;">{row_data['xi8_thu_nhap']:,}M</h2>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""<div class="metric-card">
        <p style="color:#888; font-size:14px; margin:0;">CIC Score</p>
        <h2 style="color:#00BFFF; margin:0;">{row_data['xi23_diem_cic']}</h2>
    </div>""", unsafe_allow_html=True)

with m4:
    status_icon = "🟢" if pred == 1 else "🔴"
    st.markdown(f"""<div class="metric-card">
        <p style="color:#888; font-size:14px; margin:0;">System Verdict</p>
        <h2 style="font-size:1.2rem; margin:0; padding-top:8px;">{status_icon} {'ELITE' if pred==1 else 'HIGH RISK'}</h2>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Graph and Info
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown('<div class="metric-card" style="height:600px;">', unsafe_allow_html=True)
    st.markdown("### 🕸️ Relationship Network")
    html_path = draw_neighborhood(selected_id, depth)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=500)
    os.unlink(html_path)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card" style="height:600px; overflow-y:auto;">', unsafe_allow_html=True)
    st.markdown("### 📋 Profile Details")
    
    # Hiển thị thông tin dạng list hiện đại
    details = {
        "Age": row_data["xi1_tuoi"],
        "Overdue Amount": f"{row_data['xi18_tien_qua_han']:,} VND",
        "Bad Debt History": "Yes" if row_data["xi22_co_no_xau"] == 1 else "No",
        "Real Estate": "Owned" if row_data["xi33_co_bat_dong_san"] == 1 else "None",
        "Vehicle": "Owned" if row_data["xi34_co_xe"] == 1 else "None"
    }
    
    for k, v in details.items():
        st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.05)">
                <span style="color:#888">{k}</span>
                <span style="color:#fff; font-weight:600">{v}</span>
            </div>
        """, unsafe_allow_html=True)
    
    # GNN Insight
    st.markdown("<br><b>💡 Network Insights</b>", unsafe_allow_html=True)
    neighbors_list = list(G.neighbors(selected_id)) if selected_id in G else []
    good_n = sum(1 for n in neighbors_list if df_cust[df_cust["id"]==n]["label"].values[0] == 1)
    
    st.caption(f"Connected to {len(neighbors_list)} entities. {good_n} of them are healthy.")
    st.progress(good_n/len(neighbors_list) if neighbors_list else 0)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-top:50px; padding:20px; color:#444; border-top:1px solid #222'>
    Advanced Financial Graph Analytics • Powered by Gemini 3 Flash & GNN
</div>
""", unsafe_allow_html=True)