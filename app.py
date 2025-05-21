import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="BioData ML App", layout="wide")

st.markdown("""
<div style='text-align: center;'>
    <svg width="150" height="150" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="grad" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0%" stop-color="#0ff"/>
        <stop offset="100%" stop-color="#f0f"/>
        </linearGradient>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur"/>
        <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>
        </filter>
    </defs>
    <path d="M 50 30 Q 100 80 50 130 Q 0 80 50 30 Z" fill="url(#grad)" filter="url(#glow)">
        <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="8s" repeatCount="indefinite"/>
    </path>
    </svg>
</div>
""", unsafe_allow_html=True)

# Cyberpunk CSS Styling
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    body {
        background: linear-gradient(135deg, #0d0d0d, #1a1a1a);
        color: #39ff14;
        font-family: 'Share Tech Mono', monospace;
    }
    .main {
        background-color: rgba(20, 20, 20, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 0 30px #00ffff66;
    }
    h1, h2, h3 {
        color: #00ffff;
        text-shadow: 0 0 5px #00ffffaa;
    }
    .sidebar .sidebar-content {
        background-color: #141414;
        color: #00ffff;
        border-right: 1px solid #00ffff33;
    }
    .stButton > button {
        background-color: #ff00ff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 0 0 10px #ff00ff88;
    }
    .stButton > button:hover {
        background-color: #ff33cc;
        box-shadow: 0 0 20px #ff33ccaa;
    }
    input, textarea, select {
        background-color: #222 !important;
        color: #0ff !important;
        border: 1px solid #0ff !important;
    }
    .neon-banner {
        font-size: 2rem;
        color: #39ff14;
        text-align: center;
        padding: 1rem;
        text-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 40px #0ff;
        animation: pulse 2s infinite alternate;
    }
    @keyframes pulse {
        from { text-shadow: 0 0 5px #39ff14; }
        to { text-shadow: 0 0 20px #0ff, 0 0 40px #0ff; }
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #ff00ff; border-radius: 10px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    a { color: #0ff; }
</style>

<div class="neon-banner">
    🧬 Welcome to gene expression analysis app 🌌
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Menu")
page = st.sidebar.radio("Navigation:", ["PCA Analysis", "Clustering", "Volcano Plot", "Team Info"])

uploaded_file = st.sidebar.file_uploader("Upload file (.csv, .tsv, .xlsx, .txt)", type=["csv", "tsv", "xlsx", "txt"])

theme = st.sidebar.radio("🌓 Theme", ["Cyberpunk (Dark)", "Light Lab"])
st.markdown("<style id='theme-style'></style>", unsafe_allow_html=True)

if theme == "Cyberpunk (Dark)":
    st.markdown("""
    <style id='theme-style'>
        body { background-color: #0d0d0d !important; color: #39ff14 !important; }
        .stApp { background-color: #0d0d0d; }
        .main { background-color: rgba(20, 20, 20, 0.95); color: #0ff; }
        h1, h2, h3, h4, h5 { color: #ff00ff !important; }
        .stButton>button { background-color: #ff00ff !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)
elif theme == "Light Lab":
    st.markdown("""
    <style id='theme-style'>
        body { background-color: #ffffff !important; color: #222 !important; }
        .stApp { background-color: #ffffff; }
        .main { background-color: #f9f9f9; color: #111; }
        h1, h2, h3, h4, h5 { color: #2980b9 !important; }
        .stButton>button { background-color: #1abc9c !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# Data loader
@st.cache_data
def load_data_csv(uploaded_file, sep):
    return pd.read_csv(uploaded_file, sep=sep)

@st.cache_data
def load_data_excel(uploaded_file):
    return pd.read_excel(uploaded_file)

df = None
if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext in ["csv", "tsv", "txt"]:
        default_sep = "," if file_ext == "csv" else ("\t" if file_ext in ["tsv", "txt"] else ",")
        sep = st.sidebar.text_input("Separator", value=default_sep)
        try:
            df = load_data_csv(uploaded_file, sep)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    elif file_ext == "xlsx":
        try:
            df = load_data_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to read Excel file: {e}")
    else:
        st.warning("Unsupported file format.")

# PAGE 1: PCA
if page == "PCA Analysis":
    st.title("🔬 PCA Analysis")
    if df is not None:
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        st.write("📊 Data Preview:")
        st.dataframe(df.head())

        if numeric_df.shape[1] >= 2:
            with st.expander("⚙️ PCA Settings"):
                n_comp = st.slider("Number of components", 2, min(5, numeric_df.shape[1]), 2)

            scaled = StandardScaler().fit_transform(numeric_df)
            pca = PCA(n_components=n_comp)
            pcs = pca.fit_transform(scaled)
            pc_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_comp)])

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📈 PCA Plot")
                fig, ax = plt.subplots()
                sns.scatterplot(x="PC1", y="PC2", data=pc_df, ax=ax, palette="pastel")
                st.pyplot(fig)
            with col2:
                st.subheader("📋 Variance Ratios")
                st.write(pca.explained_variance_ratio_)

            # 📤 Export PCA Results
            st.subheader("📤 Export PCA Data")
            pca_export = pc_df.copy()
            pca_export.insert(0, "Sample", df.index)
            csv = pca_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download PCA Results (.csv)",
                data=csv,
                file_name="pca_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("At least 2 numerical columns are required.")
    else:
        st.info("Please upload a file to get started.")

# PAGE 2: Clustering
elif page == "Clustering":
    st.title("🔗 Clustering with KMeans")
    if df is not None:
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        st.dataframe(df.head())

        with st.expander("⚙️ Clustering Settings"):
            k = st.slider("Number of clusters", 2, 10, 3)

        scaled = StandardScaler().fit_transform(numeric_df)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled)
        df["Cluster"] = kmeans.labels_

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)
        pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        pc_df["Cluster"] = kmeans.labels_

        st.subheader("📊 PCA with Clusters")
        fig, ax = plt.subplots()
        sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="Cluster", palette="Set2", ax=ax)
        st.pyplot(fig)

        # 📤 Export Clustering Results
        st.subheader("📤 Export Clustering Data")
        cluster_export = df.copy()
        csv_clusters = cluster_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Clustered Data (.csv)",
            data=csv_clusters,
            file_name="clustered_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Please upload a file first.")

# PAGE 3: Volcano Plot
elif page == "Volcano Plot":
    st.title("🌋 Volcano Plot")
    st.markdown("Example volcano plot with random DEG data")

    np.random.seed(42)
    data = pd.DataFrame({
        "log2FoldChange": np.random.randn(1000),
        "pval": np.random.rand(1000)
    })
    data["-log10(pval)"] = -np.log10(data["pval"])
    data["significant"] = (abs(data["log2FoldChange"]) > 1) & (data["pval"] < 0.05)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="log2FoldChange",
        y="-log10(pval)",
        hue="significant",
        data=data,
        palette={True: "red", False: "gray"},
        ax=ax
    )
    ax.axhline(-np.log10(0.05), ls="--", color="black")
    ax.axvline(-1, ls="--", color="black")
    ax.axvline(1, ls="--", color="black")
    st.pyplot(fig)

    # 📤 Export Volcano Plot Data
    st.subheader("📤 Export Volcano Data")
    volcano_export = data.copy()
    csv_volcano = volcano_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Volcano Plot Data (.csv)",
        data=csv_volcano,
        file_name="volcano_data.csv",
        mime="text/csv"
    )

# PAGE 4: Team Info
elif page == "Team Info":
    st.title("👥 Team Info")
    st.markdown("""
    ### Members:
    - **Member 1 - Sinekidis Alexandros - inf2022186** – PCA Analysis, Volcano Plot  
    - **Member 2 - Nikoletta Hadjiangeli - inf2022241** – Clustering, UI Optimization  
    - **Member 3 - Thomas Nakos - inf2022141** – Dockerization, Responsive Design  
    """)
