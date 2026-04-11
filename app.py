import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# 0. 页面基础设置
# =========================
st.set_page_config(
    page_title="Iris Classification Web Demo",
    page_icon="🌸",
    layout="wide"
)

# =========================
# 1. 全局绘图风格与学术配色定义
# =========================
# 莫兰迪低饱和配色字典
MORANDI_DICT = {
    "Setosa": "#FFB7B2",       # 淡粉
    "Versicolour": "#B2E2F2",  # 淡蓝
    "Virginica": "#B2F2BB"     # 淡绿
}
MORANDI_PALETTE = list(MORANDI_DICT.values())
PRIMARY_COLOR = "#7EA6E0"      # 主题蓝色（用于柱状图、折线等）

plt.style.use("default")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["text.color"] = "#333333"
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#555555"
plt.rcParams["ytick.color"] = "#555555"
plt.rcParams["font.family"] = "Times New Roman"
# 学术风：去掉图表顶部和右侧的边框
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

SPECIES_ORDER = ["Setosa", "Versicolour", "Virginica"]
FEATURE_COLS = ["sepallength", "sepalwidth", "petallength", "petalwidth"]

# =========================
# 2. 数据准备函数
# =========================
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(
        iris.data,
        columns=["sepallength", "sepalwidth", "petallength", "petalwidth"]
    )
    df["species"] = iris.target
    species_map = {0: "Setosa", 1: "Versicolour", 2: "Virginica"}
    df["species"] = df["species"].map(species_map)

    sl_mean = df["sepallength"].mean()
    sl_median = df["sepallength"].median()
    sl_std = df["sepallength"].std()
    pct_5 = np.percentile(df["sepallength"], 5)
    pct_95 = np.percentile(df["sepallength"], 95)

    summary = {
        "samples": len(df),
        "features": 4,
        "classes": df["species"].nunique(),
        "sl_mean": sl_mean,
        "sl_median": sl_median,
        "sl_std": sl_std,
        "sl_pct_5": pct_5,
        "sl_pct_95": pct_95
    }
    return df, summary


def get_train_test_data(df):
    X = df[FEATURE_COLS]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

# =========================
# 3. 可视化与描述生成函数
# =========================
def plot_pca_3d(df):
    """交互式 3D PCA 降维可视化"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
    pca_df['species'] = df['species']
    exp_var_cumul = pca.explained_variance_ratio_.cumsum()

    fig = px.scatter_3d(
        pca_df, x='PC1', y='PC2', z='PC3',
        color='species',
        color_discrete_map=MORANDI_DICT,
        title=f"3D PCA Dimensionality Reduction (Cum. Variance: {exp_var_cumul[-1]:.2%})",
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
        opacity=0.9
    )

    # 去除复杂网格，保留极简学术白底
    fig.update_layout(
        template="simple_white",
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#E5E5E5", showbackground=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#E5E5E5", showbackground=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#E5E5E5", showbackground=False),
        ),
        font=dict(family="Times New Roman", size=12, color="#333333"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="petallength",
        y="petalwidth",
        hue="species",
        hue_order=SPECIES_ORDER,
        s=100,
        palette=MORANDI_PALETTE,
        edgecolor="#777777",
        alpha=0.9,
        ax=ax
    )
    ax.set_title("2D Feature Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.legend(title="Species", loc="upper left", frameon=False)
    plt.tight_layout()
    return fig

def plot_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag", 
        square=True,
        linewidths=1,
        cbar=True,
        ax=ax,
        annot_kws={"size": 11, "color": "#222222"}
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="PuBu",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        linecolor='white',
        ax=ax,
        annot_kws={"size": 12}
    )
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig

def plot_knn_curve(k_values, train_accs, test_accs, best_k):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, train_accs, marker="o", linewidth=2, color="#B2E2F2", label="Training Accuracy")
    ax.plot(k_values, test_accs, marker="s", linewidth=2, color=PRIMARY_COLOR, label="Testing Accuracy")
    ax.axvline(x=best_k, linestyle="--", linewidth=1.5, color="#A0A0A0", label=f"Best k = {best_k}")
    ax.set_title("KNN Performance vs. Parameter k", fontweight="bold", pad=10)
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig

def plot_svm_curve(C_values, train_accs, test_accs, best_c):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = list(range(len(C_values)))
    ax.plot(idx, train_accs, marker="o", linewidth=2, color="#B2E2F2", label="Training Accuracy")
    ax.plot(idx, test_accs, marker="s", linewidth=2, color=PRIMARY_COLOR, label="Testing Accuracy")
    ax.axvline(x=C_values.index(best_c), linestyle="--", linewidth=1.5, color="#A0A0A0", label=f"Best C = {best_c}")
    ax.set_title("SVM Performance vs. Parameter C", fontweight="bold", pad=10)
    ax.set_xlabel("Parameter C")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(idx)
    ax.set_xticklabels(C_values)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_df, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        color=PRIMARY_COLOR,
        alpha=0.85,
        ax=ax
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig

def plot_compare_bar(compare_df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=compare_df,
        x="Model",
        y="Accuracy",
        color=PRIMARY_COLOR,
        alpha=0.85,
        ax=ax
    )
    ax.set_title("Accuracy Baseline of Four Models", fontweight="bold", pad=10)
    ax.set_xlabel("Classification Model")
    ax.set_ylabel("Test Set Accuracy")
    ax.set_ylim(0, 1.1)
    
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    
    for i, value in enumerate(compare_df["Accuracy"]):
        if pd.notna(value):
            ax.text(i, value + 0.02, f"{value:.4f}", ha="center", fontsize=11, fontweight='bold', fontfamily="Times New Roman")
    plt.tight_layout()
    return fig

# 生成动态客观结果描述
def get_objective_cm_text(cm, labels):
    total = np.sum(cm)
    correct = np.trace(cm)
    misclassified = total - correct
    
    desc = f"📊 **测试集分类结果客观统计**：共计测试 **{total}** 个样本，当前模型正确预测了 **{correct}** 个。\n\n"
    
    if misclassified == 0:
        desc += f"当前参数下，模型对测试集的分类准确率达到了 100%，无误判样本。"
    else:
        desc += f"存在 **{misclassified}** 个误判样本。具体错误分布如下："
        errors = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    errors.append(f"将真实类别为 **{labels[i]}** 的样本错误预测为了 **{labels[j]}**（共 {cm[i, j]} 例）")
        desc += "；".join(errors) + "。"
        
    return desc

# =========================
# 4. 模型函数
# =========================
def run_knn(df, k=5):
    X_train, X_test, y_train, y_test = get_train_test_data(df)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    k_values = list(range(1, 21))
    train_accs, test_accs = [], []
    for temp_k in k_values:
        temp_model = KNeighborsClassifier(n_neighbors=temp_k)
        temp_model.fit(X_train_std, y_train)
        train_accs.append(accuracy_score(y_train, temp_model.predict(X_train_std)))
        test_accs.append(accuracy_score(y_test, temp_model.predict(X_test_std)))
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)
    return {"accuracy": acc, "report_df": report_df, 
            "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix (k={k})"),
            "curve_fig": plot_knn_curve(k_values, train_accs, test_accs, k_values[int(np.argmax(test_accs))]),
            "cm": cm, "labels": SPECIES_ORDER}

def run_svm(df, c_value=1.0):
    X_train, X_test, y_train, y_test = get_train_test_data(df)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    C_values = [0.01, 0.1, 1, 10, 100]
    train_accs, test_accs = [], []
    for c in C_values:
        temp_model = SVC(C=c, kernel="rbf", gamma="scale", random_state=42)
        temp_model.fit(X_train_std, y_train)
        train_accs.append(accuracy_score(y_train, temp_model.predict(X_train_std)))
        test_accs.append(accuracy_score(y_test, temp_model.predict(X_test_std)))
        
    model = SVC(C=c_value, kernel="rbf", gamma="scale", random_state=42)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)
    return {"accuracy": acc, "report_df": report_df, 
            "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix (C={c_value})"),
            "curve_fig": plot_svm_curve(C_values, train_accs, test_accs, C_values[int(np.argmax(test_accs))]),
            "cm": cm, "labels": SPECIES_ORDER}

def run_random_forest(df, n_estimators=200):
    X_train, X_test, y_train, y_test = get_train_test_data(df)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)
    importance_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    return {"accuracy": acc, "report_df": report_df, 
            "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix (Trees={n_estimators})"),
            "importance_fig": plot_feature_importance(importance_df, "Feature Importance (RF)"),
            "importance_df": importance_df,
            "cm": cm, "labels": SPECIES_ORDER}

def run_xgboost(df, n_estimators=100, max_depth=3, learning_rate=0.1):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"error": "当前环境未安装 xgboost。请先运行：pip install xgboost"}
    X = df[FEATURE_COLS]
    y = df["species"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=0.9, colsample_bytree=0.9, objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred)
    importance_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    return {"accuracy": acc, "report_df": report_df, 
            "cm_fig": plot_confusion_matrix(cm, label_encoder.classes_, "Confusion Matrix (XGBoost)"),
            "importance_fig": plot_feature_importance(importance_df, "Feature Importance (XGBoost)"),
            "importance_df": importance_df,
            "cm": cm, "labels": label_encoder.classes_}

@st.cache_data
def compare_models(df):
    knn_result = run_knn(df, k=5)
    svm_result = run_svm(df, c_value=1.0)
    rf_result = run_random_forest(df, n_estimators=200)
    xgb_result = run_xgboost(df, n_estimators=100, max_depth=3, learning_rate=0.1)
    data = {
        "Model": ["KNN", "SVM", "Random Forest", "XGBoost"],
        "Accuracy": [
            knn_result["accuracy"], svm_result["accuracy"], rf_result["accuracy"],
            np.nan if "error" in xgb_result else xgb_result["accuracy"]
        ]
    }
    return pd.DataFrame(data)

# =========================
# 5. 页面主体
# =========================
df, summary = load_data()

st.title("🌸 Iris Classification Web System")
st.caption("Python Course Project | Data Visualization + Classical ML + Interactive Demo")

with st.expander("📊 系统功能简介", expanded=True):
    st.markdown("""
    本交互式演示系统集成了数据预处理、可视化探索与模型评估的全流程，包含以下核心模块：
    1. **Data Overview (数据概览)**：展示鸢尾花数据集的基础统计信息与分布特征。
    2. **Visualization (特征可视化)**：集成 PCA 3D 降维展示与特征相关性热力图，直观呈现数据空间分布。
    3. **Model Comparison (模型基准测试)**：宏观对比 KNN、SVM、RF 与 XGBoost 四种经典算法的基准准确率。
    4. **Interactive Demo (交互式演示)**：支持动态调整超参数，实时输出模型评估指标、混淆矩阵及特征分析。
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview", "Visualization", "Model Comparison", "Interactive Model Demo"
])

with tab1:
    st.subheader("1. Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", summary["samples"])
    col2.metric("Features", summary["features"])
    col3.metric("Classes", summary["classes"])

    st.subheader("2. Basic Statistics of Sepal Length")
    stats_df = pd.DataFrame({
        "Statistic": ["Mean", "Median", "Std", "5th Percentile", "95th Percentile"],
        "Value": [
            round(summary["sl_mean"], 2), round(summary["sl_median"], 2), round(summary["sl_std"], 2),
            round(summary["sl_pct_5"], 2), round(summary["sl_pct_95"], 2)
        ]
    })
    st.dataframe(stats_df, use_container_width=True)

    st.subheader("3. Species Count")
    st.dataframe(df["species"].value_counts().rename_axis("Species").reset_index(name="Count"), use_container_width=True)

    st.subheader("4. Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.subheader("Interactive 3D Dimensionality Reduction (PCA)")
    st.plotly_chart(plot_pca_3d(df), use_container_width=True)
    st.info("💡 降维分析：主成分分析（PCA）将 4 维特征映射至 3 维空间。可以看出，Setosa 形成了完全独立的聚簇，而 Versicolour 与 Virginica 的决策边界更为贴近。")
    
    st.divider()
    
    st.subheader("2D Feature Distribution & Correlation")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_scatter(df), use_container_width=True)
        st.success("📊 图表解析：从散点分布可知，Setosa 在花瓣尺寸上明显偏小，与其他两类形成严格的**线性可分**边界；而 Versicolour 和 Virginica 在边界处存在轻微的特征重叠，这往往是大多数分类算法发生误判的主要区域。")
    with col2:
        st.pyplot(plot_heatmap(df), use_container_width=True)
        st.success("📊 图表解析：热力图展示了特征间的皮尔逊相关系数（Pearson Correlation）。花瓣长度、花瓣宽度呈现极强的正相关（r > 0.9），表明存在较高的**特征冗余**。这意味着在工程实践中，仅凭借花瓣特征即可提供大部分分类信息。")

with tab3:
    st.subheader("Accuracy Comparison Baseline")
    compare_df = compare_models(df)
    st.dataframe(compare_df, use_container_width=True)
    st.pyplot(plot_compare_bar(compare_df), use_container_width=True)
    st.info("💡 性能基准：上图展示了四种算法在默认参数配置下的测试集准确率，可作为后续超参数调优的参考基线。")

with tab4:
    st.subheader("Interactive Model Demo")
    st.sidebar.header("⚙️ Model Settings")

    model_name = st.sidebar.selectbox(
        "Choose Classification Algorithm",
        ["KNN", "SVM", "Random Forest", "XGBoost"]
    )

    if model_name == "KNN":
        st.markdown("### K-Nearest Neighbors (KNN) Evaluation")
        k = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=20, value=5, step=1)
        result = run_knn(df, k=k)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["curve_fig"], use_container_width=True)
        with col2:
            st.pyplot(result["cm_fig"], use_container_width=True)
            st.info(get_objective_cm_text(result["cm"], result["labels"]))
        st.subheader("Classification Report")
        st.dataframe(result["report_df"], use_container_width=True)

    elif model_name == "SVM":
        st.markdown("### Support Vector Machine (SVM) Evaluation")
        c_value = st.sidebar.selectbox("Regularization Parameter (C)", [0.01, 0.1, 1, 10, 100], index=2)
        result = run_svm(df, c_value=c_value)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["curve_fig"], use_container_width=True)
        with col2:
            st.pyplot(result["cm_fig"], use_container_width=True)
            st.info(get_objective_cm_text(result["cm"], result["labels"]))
        st.subheader("Classification Report")
        st.dataframe(result["report_df"], use_container_width=True)

    elif model_name == "Random Forest":
        st.markdown("### Random Forest Evaluation")
        n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=500, value=200, step=10)
        result = run_random_forest(df, n_estimators=n_estimators)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["cm_fig"], use_container_width=True)
            st.info(get_objective_cm_text(result["cm"], result["labels"]))
        with col2:
            st.pyplot(result["importance_fig"], use_container_width=True)
        st.subheader("Classification Report")
        st.dataframe(result["report_df"], use_container_width=True)

    else:
        st.markdown("### XGBoost Evaluation")
        n_estimators = st.sidebar.slider("Number of Boosting Rounds", min_value=50, max_value=300, value=100, step=10)
        max_depth = st.sidebar.slider("Maximum Tree Depth", min_value=2, max_value=8, value=3, step=1)
        learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)

        result = run_xgboost(df, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

        if "error" in result:
            st.error(result["error"])
        else:
            st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(result["cm_fig"], use_container_width=True)
                st.info(get_objective_cm_text(result["cm"], result["labels"]))
            with col2:
                st.pyplot(result["importance_fig"], use_container_width=True)
            st.subheader("Classification Report")
            st.dataframe(result["report_df"], use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 操作提示：请在上方下拉菜单中选择分类算法，并通过拖拽滑块实时调整模型超参数以观察分类效果。")
