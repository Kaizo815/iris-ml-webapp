import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

# =========================
# 0. 页面基础设置
# =========================
st.set_page_config(
    page_title="Iris Classification Web Demo",
    page_icon="🌸",
    layout="wide"
)

# 页面宽度与间距优化
st.markdown("""
<style>
.block-container {
    max-width: 1280px;
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
}
div[data-testid="stMetric"] {
    background: #fafafa;
    border: 1px solid #eeeeee;
    border-radius: 12px;
    padding: 10px 14px;
}
div[data-testid="stDataFrame"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 1. 全局绘图风格与配色
# =========================
MORANDI_DICT = {
    "Setosa": "#FFB7B2",
    "Versicolour": "#B2E2F2",
    "Virginica": "#B2F2BB"
}
MORANDI_PALETTE = list(MORANDI_DICT.values())
PRIMARY_COLOR = "#7EA6E0"

plt.style.use("default")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.transparent"] = False
plt.rcParams["text.color"] = "#333333"
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#555555"
plt.rcParams["ytick.color"] = "#555555"
plt.rcParams["font.family"] = "Times New Roman"
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

    iris_data = pd.DataFrame(
        iris.data,
        columns=FEATURE_COLS
    )

    iris_data["species"] = iris.target
    species_map = {
        0: "Setosa",
        1: "Versicolour",
        2: "Virginica"
    }
    iris_data["species"] = iris_data["species"].map(species_map)

    summary = {
        "samples": len(iris_data),
        "features": len(FEATURE_COLS),
        "classes": iris_data["species"].nunique()
    }

    return iris_data, summary


def get_train_test_data(iris_data):
    X = iris_data[FEATURE_COLS]
    y = iris_data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


# =========================
# 3. 图表函数
# =========================
def plot_violin(iris_data):
    # 放大到你当前版本的大约 1.5 倍左右
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.7))
    fig.suptitle(
        "Feature Distribution Analysis",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )

    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i // 2, i % 2]
        sns.violinplot(
            data=iris_data,
            x="species",
            y=col,
            palette=MORANDI_PALETTE,
            inner="quartile",
            linewidth=1.0,
            cut=0,
            ax=ax
        )
        ax.set_title(col, fontweight="bold", pad=4, fontsize=10.5)
        ax.set_xlabel("")
        ax.set_ylabel("cm", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=12)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout(pad=0.5)
    return fig


def plot_scatter(iris_data):
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    sns.scatterplot(
        data=iris_data,
        x="petallength",
        y="petalwidth",
        hue="species",
        hue_order=SPECIES_ORDER,
        s=34,
        palette=MORANDI_PALETTE,
        edgecolor="black",
        alpha=0.8,
        ax=ax
    )

    ax.set_title("2D Scatter Distribution", fontweight="bold", pad=6, fontsize=9.5)
    ax.set_xlabel("Petal Length", fontsize=7.5)
    ax.set_ylabel("Petal Width", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(title="Species", loc="upper left", frameon=False, fontsize=6.5, title_fontsize=7)

    plt.tight_layout(pad=0.25)
    return fig


def plot_heatmap(iris_data):
    corr_matrix = iris_data[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.45,
        cbar=True,
        ax=ax,
        annot_kws={"size": 7.5, "color": "#222222"}
    )

    ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=6, fontsize=9.5)
    ax.set_xlabel("")
    ax.set_ylabel("")

    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
        label.set_fontsize(6.5)

    for label in ax.get_yticklabels():
        label.set_fontsize(6.5)

    plt.tight_layout(pad=0.25)
    return fig


def plot_pca_3d(iris_data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(iris_data[FEATURE_COLS])

    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
    pca_df["species"] = iris_data["species"]
    exp_var_cumul = pca.explained_variance_ratio_.cumsum()

    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="species",
        color_discrete_map=MORANDI_DICT,
        title=f"3D PCA Dimensionality Reduction (Cum. Variance: {exp_var_cumul[-1]:.2%})",
        labels={
            "PC1": "Principal Component 1",
            "PC2": "Principal Component 2",
            "PC3": "Principal Component 3"
        },
        opacity=0.88
    )

    fig.update_traces(marker=dict(size=2.8))

    fig.update_layout(
        template="simple_white",
        height=320,
        font=dict(family="Times New Roman", size=10, color="#333333"),
        margin=dict(l=0, r=0, b=0, t=34),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#E5E5E5", showbackground=False),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#E5E5E5", showbackground=False),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#E5E5E5", showbackground=False),
        )
    )
    return fig


def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 8.5, "color": "#222222"}
    )

    ax.set_title(title, fontweight="bold", pad=5, fontsize=8.8)
    ax.set_xlabel("Predicted Label", fontsize=6.9)
    ax.set_ylabel("True Label", fontsize=6.9)

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(6)

    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontsize(6)

    plt.tight_layout(pad=0.2)
    return fig


def plot_multi_metrics_heatmap(iris_data):
    X = iris_data[FEATURE_COLS]
    y = iris_data["species"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (C=1.0)": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = {}

    for name, model in models.items():
        if name in ["KNN (k=5)", "SVM (C=1.0)"]:
            curr_X_train = X_train_std
            curr_X_test = X_test_std
        else:
            curr_X_train = X_train
            curr_X_test = X_test

        model.fit(curr_X_train, y_train)
        y_pred = model.predict(curr_X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro"
        )

        results[name] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro"
        )

        results["XGBoost"] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

    except Exception:
        pass

    metrics_df = pd.DataFrame(results).T

    # 放大到当前版本的接近 2 倍
    fig, ax = plt.subplots(figsize=(7.8, 5.7))
    sns.heatmap(
        metrics_df,
        annot=True,
        fmt=".4f",
        cmap="PuBu",
        linewidths=0.9,
        linecolor="white",
        cbar=True,
        ax=ax,
        annot_kws={"size": 12, "fontweight": "bold"}
    )

    ax.set_title("Multi-Metric Model Comparison", fontsize=14, fontweight="bold", pad=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout(pad=0.35)
    return fig


def plot_decision_boundary(iris_data, model_name, k=5, c_value=1.0, n_estimators=200, max_depth=3, learning_rate=0.1):
    plot_df = iris_data.copy()

    X_2d = plot_df[["petallength", "petalwidth"]].values
    y_text = plot_df["species"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=k)
        title = f"2D Decision Boundary of KNN (k={k})"

    elif model_name == "SVM":
        model = SVC(C=c_value, kernel="rbf", gamma="scale", random_state=42)
        title = f"2D Decision Boundary of SVM (C={c_value})"

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        title = f"2D Decision Boundary of Random Forest (Trees={n_estimators})"

    else:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42
        )
        title = "2D Decision Boundary of XGBoost"

    model.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FFD1CF", "#D4F0FA", "#D4FADD"])

    # 放大到当前版本的接近 2 倍
    fig, ax = plt.subplots(figsize=(6.3, 4))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=plot_df["species"],
        palette=MORANDI_PALETTE,
        edgecolor="#555555",
        s=42,
        alpha=0.9,
        ax=ax
    )

    ax.set_title(title, fontweight="bold", pad=8, fontsize=15)
    ax.set_xlabel("Petal Length (cm)", fontsize=11)
    ax.set_ylabel("Petal Width (cm)", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(title="True Species", loc="upper left", frameon=False, fontsize=8.5, title_fontsize=9.5)

    plt.tight_layout(pad=0.3)
    return fig


# =========================
# 4. 模型函数
# =========================
def run_knn(iris_data, k=5):
    X_train, X_test, y_train, y_test = get_train_test_data(iris_data)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix of KNN (k={k})"),
        "cm": cm,
        "labels": SPECIES_ORDER
    }


def run_svm(iris_data, c_value=1.0):
    X_train, X_test, y_train, y_test = get_train_test_data(iris_data)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = SVC(C=c_value, kernel="rbf", gamma="scale", random_state=42)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix of SVM (C={c_value})"),
        "cm": cm,
        "labels": SPECIES_ORDER
    }


def run_random_forest(iris_data, n_estimators=200):
    X = iris_data[FEATURE_COLS]
    y = iris_data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix of Random Forest (Trees={n_estimators})"),
        "cm": cm,
        "labels": SPECIES_ORDER
    }


def run_xgboost(iris_data, n_estimators=100, max_depth=3, learning_rate=0.1):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"error": "当前环境未安装 xgboost。请先运行：pip install xgboost"}

    X = iris_data[FEATURE_COLS]
    y = iris_data["species"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
    ).transpose()
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, label_encoder.classes_, "Confusion Matrix of XGBoost"),
        "cm": cm,
        "labels": label_encoder.classes_
    }


def get_objective_cm_text(cm, labels):
    total = np.sum(cm)
    correct = np.trace(cm)
    misclassified = total - correct

    desc = f"📊 **测试集分类结果客观统计**：共计测试 **{total}** 个样本，当前模型正确预测了 **{correct}** 个。\n\n"

    if misclassified == 0:
        desc += "当前参数下，模型对测试集的分类准确率达到了 100%，无误判样本。"
    else:
        desc += f"存在 **{misclassified}** 个误判样本。具体错误分布如下："
        errors = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    errors.append(
                        f"将真实类别为 **{labels[i]}** 的样本错误预测为了 **{labels[j]}**（共 {cm[i, j]} 例）"
                    )
        desc += "；".join(errors) + "。"

    return desc


# =========================
# 6. 页面主体
# =========================
iris_data, summary = load_data()

st.title("🌸 Iris Classification Web System")
st.caption("Python Course Project | Data Visualization + Classical ML + Interactive Demo")

with st.expander("📊 系统功能简介", expanded=True):
    st.markdown("""
    本交互式演示系统集成了数据预处理、可视化探索与模型评估的全流程，包含以下核心模块：
    1. **Data Overview (数据概览)**：展示鸢尾花数据集的基础统计信息与分布特征。
    2. **Feature Engineering (特征可视化)**：展示二维散点图、相关性热力图与 3D PCA 降维结果。
    3. **Model Baseline (模型基准测试)**：从多指标角度综合比较四种经典分类模型。
    4. **Interactive Demo (交互式演练)**：支持动态调整参数，并实时观察决策边界、混淆矩阵与分类指标变化。
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Feature Engineering",
    "Model Baseline",
    "Interactive Demo"
])

# =========================
# Tab 1: 数据概览
# =========================
with tab1:
    st.subheader("1. 数据概览")

    col1, col2, col3 = st.columns(3)
    col1.metric("总样本数", summary["samples"])
    col2.metric("特征数", summary["features"])
    col3.metric("类别数", summary["classes"])

    st.subheader("2. 四特征分布小提琴图")
    st.pyplot(plot_violin(iris_data), use_container_width=False)

    st.success("📊 图表解析：小提琴图展示了四个特征在三类样本中的分布形态。可以直观看出，花瓣长度与花瓣宽度的类别区分度明显高于花萼特征，因此它们通常提供更强的分类信息。")

    st.subheader("3. 原始数据预览")
    st.dataframe(iris_data, use_container_width=True, height=280)

# =========================
# Tab 2: 特征可视化
# =========================
with tab2:
    st.subheader("1. 基础特征可视化")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_scatter(iris_data), use_container_width=False)
        st.success("📊 图表解析：Setosa 在花瓣尺寸上明显更小，与另外两类容易分开；Versicolour 与 Virginica 在边界附近存在一定重叠。")

    with col2:
        st.pyplot(plot_heatmap(iris_data), use_container_width=False)
        st.success("📊 图表解析：热力图展示了四个数值特征之间的相关性。花瓣长度与花瓣宽度相关性较强，说明这两个特征对分类非常关键。")

    st.divider()

    st.subheader("2. 3D PCA 降维展示")
    _, center, _ = st.columns([1, 1.35, 1])
    with center:
        st.plotly_chart(plot_pca_3d(iris_data), use_container_width=False)

    st.info("💡 降维分析：主成分分析（PCA）将 4 维特征映射至 3 维空间。可以看出，Setosa 形成了较为独立的聚簇，而 Versicolour 与 Virginica 的边界更接近。")

# =========================
# Tab 3: 模型基准测试
# =========================
with tab3:
    st.subheader("多指标模型比较热力图")
    st.pyplot(plot_multi_metrics_heatmap(iris_data), use_container_width=False)

    st.info(
        "💡 学术解析：该热力图从 Accuracy、Precision、Recall 与 F1-Score 四个维度同时比较了四种模型的基准性能。整体来看，树模型（Random Forest、XGBoost）通常在综合指标上表现更稳定，而距离/空间模型（KNN、SVM）则更依赖特征空间的分布结构。该结果可以作为后续交互式调参与模型分析的基准参考。"
    )

# =========================
# Tab 4: 交互式演练
# =========================
with tab4:
    st.subheader("交互式演练")
    st.sidebar.header("⚙️ Model Settings")

    model_name = st.sidebar.selectbox(
        "Choose Classification Algorithm",
        ["KNN", "SVM", "Random Forest", "XGBoost"]
    )

    if model_name == "KNN":
        st.markdown("### K-Nearest Neighbors (KNN) Evaluation")
        k = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=20, value=5, step=1)
        result = run_knn(iris_data, k=k)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

        st.subheader("决策边界图")
        st.pyplot(plot_decision_boundary(iris_data, "KNN", k=k), use_container_width=False)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["cm_fig"], use_container_width=False)
            st.info(get_objective_cm_text(result["cm"], result["labels"]))
        with col2:
            st.subheader("Classification Report")
            st.dataframe(result["report_df"], use_container_width=True, height=260)

    elif model_name == "SVM":
        st.markdown("### Support Vector Machine (SVM) Evaluation")
        c_value = st.sidebar.selectbox("Regularization Parameter (C)", [0.01, 0.1, 1, 10, 100], index=2)
        result = run_svm(iris_data, c_value=c_value)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

        st.subheader("决策边界图")
        st.pyplot(plot_decision_boundary(iris_data, "SVM", c_value=c_value), use_container_width=False)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["cm_fig"], use_container_width=False)
            st.info(get_objective_cm_text(result["cm"], result["labels"]))
        with col2:
            st.subheader("Classification Report")
            st.dataframe(result["report_df"], use_container_width=True, height=260)

    elif model_name == "Random Forest":
        st.markdown("### Random Forest Evaluation")
        n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=500, value=200, step=10)
        result = run_random_forest(iris_data, n_estimators=n_estimators)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

        st.subheader("决策边界图")
        st.pyplot(plot_decision_boundary(iris_data, "Random Forest", n_estimators=n_estimators), use_container_width=False)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["cm_fig"], use_container_width=False)
            st.info(get_objective_cm_text(result["cm"], result["labels"]))
        with col2:
            st.subheader("Classification Report")
            st.dataframe(result["report_df"], use_container_width=True, height=260)

    else:
        st.markdown("### XGBoost Evaluation")
        n_estimators = st.sidebar.slider("Number of Boosting Rounds", min_value=50, max_value=300, value=100, step=10)
        max_depth = st.sidebar.slider("Maximum Tree Depth", min_value=2, max_value=8, value=3, step=1)
        learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)

        result = run_xgboost(
            iris_data,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )

        if "error" in result:
            st.error(result["error"])
        else:
            st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

            st.subheader("决策边界图")
            st.pyplot(
                plot_decision_boundary(
                    iris_data,
                    "XGBoost",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate
                ),
                use_container_width=False
            )

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(result["cm_fig"], use_container_width=False)
                st.info(get_objective_cm_text(result["cm"], result["labels"]))
            with col2:
                st.subheader("Classification Report")
                st.dataframe(result["report_df"], use_container_width=True, height=260)

st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 操作提示：请在上方下拉菜单中选择分类算法，并通过拖拽滑块实时调整模型超参数以观察分类效果。")
