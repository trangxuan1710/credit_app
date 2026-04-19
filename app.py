import streamlit as st
import pandas as pd
import pickle
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
import tempfile, os

# ── Cấu hình trang ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Credit Visualization",
    page_icon="🏦",
    layout="wide"
)

# ── Load dữ liệu ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_cust = pd.read_excel("data_training_tin_dung.xlsx", sheet_name="customer")
    df_edge = pd.read_excel("data_training_tin_dung.xlsx", sheet_name="edge")
    le = LabelEncoder()
    df_cust["xi10_nghe_nghiep"] = le.fit_transform(df_cust["xi10_nghe_nghiep"].astype(str))
    return df_cust, df_edge

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

df_cust, df_edge = load_data()
model, feature_cols, le = load_model()

# ── Xây đồ thị ──────────────────────────────────────────────────
@st.cache_data
def build_graph():
    G = nx.Graph()
    for _, row in df_cust.iterrows():
        G.add_node(int(row["id"]), label_true=int(row["label"]))
    for _, row in df_edge.iterrows():
        G.add_edge(int(row["src"]), int(row["dst"]),
                   weight=float(row["weight"]),
                   relation=row["relation_type"])
    return G

G = build_graph()

# ── Dự báo điểm tín dụng ────────────────────────────────────────
def predict_score(customer_id):
    row = df_cust[df_cust["id"] == customer_id]
    if row.empty:
        return None, None
    X = row[feature_cols].fillna(0)
    prob = model.predict_proba(X)[0][1]   # xác suất tốt
    pred = model.predict(X)[0]
    return prob, pred

# ── Vẽ đồ thị lân cận ───────────────────────────────────────────
def draw_neighborhood(customer_id, depth=1):
    # Lấy các node lân cận
    neighbors = set([customer_id])
    for _ in range(depth):
        new_n = set()
        for n in neighbors:
            if n in G:
                new_n.update(G.neighbors(n))
        neighbors.update(new_n)
    
    sub = G.subgraph(neighbors)
    net = Network(height="520px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut()

    color_map = {1: "#00cc66", 0: "#ff4444"}
    
    for node in sub.nodes():
        row = df_cust[df_cust["id"] == node]
        lbl_true = int(row["label"].values[0]) if not row.empty else -1
        prob, _ = predict_score(node)
        prob_text = f"{prob*100:.0f}%" if prob is not None else "?"
        
        color = color_map.get(lbl_true, "#888888")
        size = 30 if node == customer_id else 18
        border = "#FFD700" if node == customer_id else color
        
        net.add_node(
            node,
            label=f"KH {node}\n{prob_text}",
            color={"background": color, "border": border},
            size=size,
            title=f"Mã KH: {node} | Điểm tốt: {prob_text} | Nhãn thực: {'Tốt ✅' if lbl_true==1 else 'Xấu ❌'}"
        )
    
    relation_color = {"vay_chung": "#FFA500", "chuyen_tien": "#00BFFF"}
    for src, dst, data in sub.edges(data=True):
        net.add_edge(src, dst,
                     color=relation_color.get(data.get("relation",""), "#888"),
                     width=data.get("weight", 0.5) * 4,
                     title=f"{data.get('relation','')} | Trọng số: {data.get('weight',0):.1f}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
        net.save_graph(f.name)
        return f.name

# ── GIAO DIỆN CHÍNH ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#00cc66;'>🏦 Smart Credit Visualization</h1>
<p style='text-align:center; color:#aaa;'>Ứng dụng GNN trong Chấm điểm Tín dụng Thông minh</p>
<hr>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Tra cứu khách hàng")
    all_ids = sorted(df_cust["id"].tolist())
    customer_id = st.selectbox("Chọn Mã Khách Hàng:", all_ids)
    depth = st.slider("Độ sâu lân cận (hops):", 1, 2, 1)
    
    st.markdown("---")
    prob, pred = predict_score(customer_id)
    
    if prob is not None:
        score_color = "#00cc66" if pred == 1 else "#ff4444"
        verdict = "✅ Tín dụng TỐT" if pred == 1 else "❌ Tín dụng XẤU"
        
        st.markdown(f"""
        <div style='background:#1e2130;padding:20px;border-radius:12px;border-left:5px solid {score_color}'>
            <h3 style='color:{score_color}'>{verdict}</h3>
            <h1 style='color:{score_color};font-size:48px'>{prob*100:.1f}%</h1>
            <p style='color:#aaa'>Xác suất tín dụng tốt</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Thông tin khách hàng")
        row = df_cust[df_cust["id"] == customer_id].iloc[0]
        
        info = {
            "📅 Tuổi": row["xi1_tuoi"],
            "💰 Thu nhập": f"{row['xi8_thu_nhap']:,} triệu",
            "🏦 Điểm CIC": row["xi23_diem_cic"],
            "💳 Nợ quá hạn": f"{row['xi18_tien_qua_han']:,}",
            "⚠️ Nợ xấu": "Có" if row["xi22_co_no_xau"] == 1 else "Không",
            "🏠 Bất động sản": "Có" if row["xi33_co_bat_dong_san"] == 1 else "Không",
            "🚗 Xe": "Có" if row["xi34_co_xe"] == 1 else "Không",
        }
        for k, v in info.items():
            st.metric(k, v)

with col2:
    st.subheader("🕸️ Mạng lưới quan hệ tín dụng")
    
    # Chú thích
    st.markdown("""
    <div style='display:flex;gap:20px;margin-bottom:10px;font-size:13px'>
        <span>🟢 Tín dụng tốt</span>
        <span>🔴 Tín dụng xấu</span>
        <span>🟡 Khách hàng đang xem</span>
        <span>🟠 Vay chung</span>
        <span>🔵 Chuyển tiền</span>
    </div>
    """, unsafe_allow_html=True)
    
    html_path = draw_neighborhood(customer_id, depth)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=540, scrolling=False)
    os.unlink(html_path)
    
    # Giải thích GNN
    neighbors = list(G.neighbors(customer_id)) if customer_id in G else []
    good_neighbors = sum(1 for n in neighbors 
                         if not df_cust[df_cust["id"]==n].empty 
                         and df_cust[df_cust["id"]==n]["label"].values[0] == 1)
    
    st.markdown("---")
    st.subheader("💡 Giải thích GNN")
    st.info(f"""
    Khách hàng **{customer_id}** có **{len(neighbors)} mối quan hệ** trong mạng lưới tài chính.
    
    Trong đó **{good_neighbors}/{len(neighbors)} láng giềng** có tín dụng tốt.
    
    {"🟢 Mạng lưới quan hệ **tích cực** → củng cố điểm tín dụng." if good_neighbors > len(neighbors)/2 else "🔴 Mạng lưới quan hệ **rủi ro** → ảnh hưởng xấu đến điểm tín dụng."}
    
    > *Đây là nguyên lý cốt lõi của **Graph Neural Network (GNN)**: điểm tín dụng không chỉ phụ thuộc vào cá nhân mà còn bị ảnh hưởng bởi toàn bộ mạng lưới xã hội tài chính.*
    """)

# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style='text-align:center;color:#666;font-size:12px'>
NCKH: "Ứng dụng GNN và GCD trong Chấm điểm Tín dụng Thông minh cho Ngân hàng Số"
</p>
""", unsafe_allow_html=True)