import warnings
warnings.filterwarnings("ignore")

from io import BytesIO

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

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# =========================
# 0. 页面基础设置
# =========================
st.set_page_config(
    page_title="Iris Classification Web Demo",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1280px;
    padding-top: 1.1rem;
    padding-bottom: 1.1rem;
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

BEST_K = 9
BEST_C = 10
BEST_RF_N = 50
BEST_XGB_N = 10

# XGBoost 固定参数
FIXED_XGB_MAX_DEPTH = 3
FIXED_XGB_LEARNING_RATE = 0.1


# =========================
# 2. 通用工具
# =========================
def show_fig(fig, use_container_width=False):
    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)


def fig_to_png_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def show_png(png_bytes, use_container_width=False):
    st.image(png_bytes, use_container_width=use_container_width)


# =========================
# 3. 数据准备函数
# =========================
@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def get_prepared_data(iris_data):
    X = iris_data[FEATURE_COLS].copy()
    y_text = iris_data["species"].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_text)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=y_text
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)

    y_train_text = y_text.iloc[train_idx].reset_index(drop=True)
    y_test_text = y_text.iloc[test_idx].reset_index(drop=True)

    y_train_encoded = y_encoded[train_idx]
    y_test_encoded = y_encoded[test_idx]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return {
        "X": X,
        "y_text": y_text,
        "y_encoded": y_encoded,
        "label_classes": list(le.classes_),
        "X_train": X_train,
        "X_test": X_test,
        "y_train_text": y_train_text,
        "y_test_text": y_test_text,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std
    }


@st.cache_data(show_spinner=False)
def get_corr_matrix(iris_data):
    return iris_data[FEATURE_COLS].corr()


@st.cache_data(show_spinner=False)
def get_pca_data(iris_data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(iris_data[FEATURE_COLS])

    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"])
    pca_df["species"] = iris_data["species"]
    exp_var_cumul = pca.explained_variance_ratio_.cumsum()

    return pca_df, float(exp_var_cumul[-1])


# =========================
# 4. 图表函数
# =========================
def plot_violin(iris_data):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.1))
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
    corr_matrix = get_corr_matrix(iris_data)

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
    pca_df, exp_var_cumul = get_pca_data(iris_data)

    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="species",
        color_discrete_map=MORANDI_DICT,
        title=f"3D PCA Dimensionality Reduction (Cum. Variance: {exp_var_cumul:.2%})",
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


# =========================
# 4.1 静态图缓存
# =========================
@st.cache_data(show_spinner=False)
def get_violin_png(iris_data):
    return fig_to_png_bytes(plot_violin(iris_data))


@st.cache_data(show_spinner=False)
def get_scatter_png(iris_data):
    return fig_to_png_bytes(plot_scatter(iris_data))


@st.cache_data(show_spinner=False)
def get_heatmap_png(iris_data):
    return fig_to_png_bytes(plot_heatmap(iris_data))


@st.cache_resource(show_spinner=False)
def get_pca_plotly_fig(iris_data):
    return plot_pca_3d(iris_data)


@st.cache_data(show_spinner=False)
def get_baseline_curve_data(iris_data):
    prepared = get_prepared_data(iris_data)

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train_text = prepared["y_train_text"]
    y_test_text = prepared["y_test_text"]
    X_train_std = prepared["X_train_std"]
    X_test_std = prepared["X_test_std"]
    y_train_encoded = prepared["y_train_encoded"]
    y_test_encoded = prepared["y_test_encoded"]

    # KNN
    k_values = list(range(1, 21))
    knn_acc = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_std, y_train_text)
        knn_acc.append(accuracy_score(y_test_text, model.predict(X_test_std)))

    # SVM
    c_values = [0.01, 0.1, 1, 10, 100]
    svm_acc = []
    for c in c_values:
        model = SVC(C=c, kernel="rbf", gamma="scale", random_state=42)
        model.fit(X_train_std, y_train_text)
        svm_acc.append(accuracy_score(y_test_text, model.predict(X_test_std)))

    # Random Forest
    rf_values = [10, 30, 50, 80, 100, 150, 200, 300]
    rf_acc = []
    for n in rf_values:
        model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train_text)
        rf_acc.append(accuracy_score(y_test_text, model.predict(X_test)))

    # XGBoost（仅调整 n_estimators，其余参数固定）
    xgb_values = [10, 30, 50, 80, 100, 150, 200, 300]
    xgb_acc = None
    if XGB_AVAILABLE:
        xgb_acc = []
        for n in xgb_values:
            model = XGBClassifier(
                n_estimators=n,
                max_depth=FIXED_XGB_MAX_DEPTH,
                learning_rate=FIXED_XGB_LEARNING_RATE,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
            model.fit(X_train, y_train_encoded)
            xgb_acc.append(accuracy_score(y_test_encoded, model.predict(X_test)))

    return {
        "knn": {"params": k_values, "acc": knn_acc, "best_param": BEST_K},
        "svm": {"params": c_values, "acc": svm_acc, "best_param": BEST_C},
        "rf": {"params": rf_values, "acc": rf_acc, "best_param": BEST_RF_N},
        "xgb": {"params": xgb_values, "acc": xgb_acc, "best_param": BEST_XGB_N}
    }


def plot_baseline_accuracy_curves(iris_data):
    curve_data = get_baseline_curve_data(iris_data)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    fig.suptitle(
        "Accuracy Curves of Four Classification Algorithms",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    # KNN
    ax = axes[0, 0]
    knn_params = curve_data["knn"]["params"]
    knn_acc = curve_data["knn"]["acc"]
    best_k = curve_data["knn"]["best_param"]
    best_k_idx = knn_params.index(best_k)

    ax.plot(knn_params, knn_acc, marker="o", linewidth=2)
    ax.axvline(best_k, linestyle="--", color="gray", linewidth=1.2)
    ax.scatter([best_k], [knn_acc[best_k_idx]], s=55, zorder=3)
    ax.set_title("KNN", fontweight="bold", fontsize=12)
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xticks(knn_params)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # SVM
    ax = axes[0, 1]
    svm_params = curve_data["svm"]["params"]
    svm_acc = curve_data["svm"]["acc"]
    best_c = curve_data["svm"]["best_param"]
    svm_x = np.arange(len(svm_params))
    best_c_idx = svm_params.index(best_c)

    ax.plot(svm_x, svm_acc, marker="o", linewidth=2)
    ax.axvline(best_c_idx, linestyle="--", color="gray", linewidth=1.2)
    ax.scatter([best_c_idx], [svm_acc[best_c_idx]], s=55, zorder=3)
    ax.set_title("SVM", fontweight="bold", fontsize=12)
    ax.set_xlabel("C", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xticks(svm_x)
    ax.set_xticklabels(svm_params)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Random Forest
    ax = axes[1, 0]
    rf_params = curve_data["rf"]["params"]
    rf_acc = curve_data["rf"]["acc"]
    best_rf = curve_data["rf"]["best_param"]
    best_rf_idx = rf_params.index(best_rf)

    ax.plot(rf_params, rf_acc, marker="o", linewidth=2)
    ax.axvline(best_rf, linestyle="--", color="gray", linewidth=1.2)
    ax.scatter([best_rf], [rf_acc[best_rf_idx]], s=55, zorder=3)
    ax.set_title("Random Forest", fontweight="bold", fontsize=12)
    ax.set_xlabel("n_estimators", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # XGBoost
    ax = axes[1, 1]
    xgb_params = curve_data["xgb"]["params"]
    xgb_acc = curve_data["xgb"]["acc"]
    best_xgb = curve_data["xgb"]["best_param"]

    if xgb_acc is None:
        ax.text(
            0.5, 0.5,
            "xgboost is not installed.\nPlease run: pip install xgboost",
            ha="center", va="center", fontsize=11
        )
        ax.set_title("XGBoost", fontweight="bold", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        best_xgb_idx = xgb_params.index(best_xgb)
        ax.plot(xgb_params, xgb_acc, marker="o", linewidth=2)
        ax.axvline(best_xgb, linestyle="--", color="gray", linewidth=1.2)
        ax.scatter([best_xgb], [xgb_acc[best_xgb_idx]], s=55, zorder=3)
        ax.set_title("XGBoost", fontweight="bold", fontsize=12)
        ax.set_xlabel("n_estimators", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


@st.cache_data(show_spinner=False)
def get_baseline_curves_png(iris_data):
    return fig_to_png_bytes(plot_baseline_accuracy_curves(iris_data))


@st.cache_data(show_spinner=False)
def get_best_metrics_df(iris_data):
    prepared = get_prepared_data(iris_data)

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    X_train_std = prepared["X_train_std"]
    X_test_std = prepared["X_test_std"]
    y_train_text = prepared["y_train_text"]
    y_test_text = prepared["y_test_text"]
    y_train_encoded = prepared["y_train_encoded"]
    y_test_encoded = prepared["y_test_encoded"]

    results = {}

    knn_model = KNeighborsClassifier(n_neighbors=BEST_K)
    knn_model.fit(X_train_std, y_train_text)
    knn_pred = knn_model.predict(X_test_std)
    knn_precision, knn_recall, knn_f1, _ = precision_recall_fscore_support(
        y_test_text, knn_pred, average="macro"
    )
    results[f"KNN (k={BEST_K})"] = {
        "Accuracy": accuracy_score(y_test_text, knn_pred),
        "Precision": knn_precision,
        "Recall": knn_recall,
        "F1-Score": knn_f1
    }

    svm_model = SVC(C=BEST_C, kernel="rbf", gamma="scale", random_state=42)
    svm_model.fit(X_train_std, y_train_text)
    svm_pred = svm_model.predict(X_test_std)
    svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(
        y_test_text, svm_pred, average="macro"
    )
    results[f"SVM (C={BEST_C})"] = {
        "Accuracy": accuracy_score(y_test_text, svm_pred),
        "Precision": svm_precision,
        "Recall": svm_recall,
        "F1-Score": svm_f1
    }

    rf_model = RandomForestClassifier(n_estimators=BEST_RF_N, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train_text)
    rf_pred = rf_model.predict(X_test)
    rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(
        y_test_text, rf_pred, average="macro"
    )
    results[f"Random Forest\n(n={BEST_RF_N})"] = {
        "Accuracy": accuracy_score(y_test_text, rf_pred),
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1-Score": rf_f1
    }

    if XGB_AVAILABLE:
        xgb_model = XGBClassifier(
            n_estimators=BEST_XGB_N,
            max_depth=FIXED_XGB_MAX_DEPTH,
            learning_rate=FIXED_XGB_LEARNING_RATE,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train_encoded)
        xgb_pred = xgb_model.predict(X_test)
        xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(
            y_test_encoded, xgb_pred, average="macro"
        )
        results[f"XGBoost\n(n={BEST_XGB_N})"] = {
            "Accuracy": accuracy_score(y_test_encoded, xgb_pred),
            "Precision": xgb_precision,
            "Recall": xgb_recall,
            "F1-Score": xgb_f1
        }

    return pd.DataFrame(results).T


def plot_best_model_heatmap(iris_data):
    metrics_df = get_best_metrics_df(iris_data)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    sns.heatmap(
        metrics_df,
        annot=True,
        fmt=".4f",
        cmap="PuBu",
        linewidths=1,
        linecolor="white",
        cbar=True,
        ax=ax,
        annot_kws={"size": 12, "fontweight": "bold"}
    )

    ax.set_title(
        "Performance Comparison of Best-Parameter Models",
        fontsize=14,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout(pad=0.35)
    return fig


@st.cache_data(show_spinner=False)
def get_best_model_heatmap_png(iris_data):
    return fig_to_png_bytes(plot_best_model_heatmap(iris_data))


# =========================
# 4.2 决策边界缓存
# =========================
@st.cache_data(show_spinner=False)
def get_decision_boundary_base(iris_data):
    plot_df = iris_data.copy()

    X_2d = plot_df[["petallength", "petalwidth"]].values
    y_text = plot_df["species"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    return {
        "X_2d": X_2d,
        "y": y,
        "species": plot_df["species"].values,
        "xx": xx,
        "yy": yy,
        "grid_points": grid_points
    }


@st.cache_data(show_spinner=False)
def get_decision_boundary_data(
    iris_data,
    model_name,
    k=5,
    c_value=1.0,
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1
):
    base = get_decision_boundary_base(iris_data)

    X_2d = base["X_2d"]
    y = base["y"]
    xx = base["xx"]
    yy = base["yy"]
    grid_points = base["grid_points"]

    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=k)
        title = f"2D Decision Boundary of KNN (k={k})"

    elif model_name == "SVM":
        model = SVC(C=c_value, kernel="rbf", gamma="scale", random_state=42)
        title = f"2D Decision Boundary of SVM (C={c_value})"

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        title = f"2D Decision Boundary of Random Forest (Trees={n_estimators})"

    else:
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=FIXED_XGB_MAX_DEPTH,
            learning_rate=FIXED_XGB_LEARNING_RATE,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        title = f"2D Decision Boundary of XGBoost (max_depth={FIXED_XGB_MAX_DEPTH}, lr={FIXED_XGB_LEARNING_RATE})"

    model.fit(X_2d, y)
    Z = model.predict(grid_points).reshape(xx.shape)

    return {
        "xx": xx,
        "yy": yy,
        "Z": Z,
        "X_2d": X_2d,
        "species": base["species"],
        "title": title
    }


def plot_decision_boundary(iris_data, model_name, k=5, c_value=1.0, n_estimators=200, max_depth=3, learning_rate=0.1):
    boundary_data = get_decision_boundary_data(
        iris_data,
        model_name,
        k=k,
        c_value=c_value,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    xx = boundary_data["xx"]
    yy = boundary_data["yy"]
    Z = boundary_data["Z"]
    X_2d = boundary_data["X_2d"]
    species = boundary_data["species"]
    title = boundary_data["title"]

    cmap_light = ListedColormap(["#FFD1CF", "#D4F0FA", "#D4FADD"])

    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=species,
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


@st.cache_data(show_spinner=False)
def get_decision_boundary_png(
    iris_data,
    model_name,
    k=5,
    c_value=1.0,
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1
):
    return fig_to_png_bytes(
        plot_decision_boundary(
            iris_data,
            model_name,
            k=k,
            c_value=c_value,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )
    )


@st.cache_data(show_spinner=False)
def get_confusion_matrix_png(cm_tuple, labels_tuple, title):
    cm = np.array(cm_tuple)
    labels = list(labels_tuple)
    return fig_to_png_bytes(plot_confusion_matrix(cm, labels, title))


# =========================
# 5. 模型函数（缓存版）
# =========================
@st.cache_data(show_spinner=False)
def run_knn_cached(iris_data, k=5):
    prepared = get_prepared_data(iris_data)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(prepared["X_train_std"], prepared["y_train_text"])
    y_pred = model.predict(prepared["X_test_std"])

    acc = accuracy_score(prepared["y_test_text"], y_pred)
    report_df = pd.DataFrame(
        classification_report(prepared["y_test_text"], y_pred, output_dict=True)
    ).transpose()
    cm = confusion_matrix(prepared["y_test_text"], y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm": cm,
        "labels": SPECIES_ORDER
    }


@st.cache_data(show_spinner=False)
def run_svm_cached(iris_data, c_value=1.0):
    prepared = get_prepared_data(iris_data)

    model = SVC(C=c_value, kernel="rbf", gamma="scale", random_state=42)
    model.fit(prepared["X_train_std"], prepared["y_train_text"])
    y_pred = model.predict(prepared["X_test_std"])

    acc = accuracy_score(prepared["y_test_text"], y_pred)
    report_df = pd.DataFrame(
        classification_report(prepared["y_test_text"], y_pred, output_dict=True)
    ).transpose()
    cm = confusion_matrix(prepared["y_test_text"], y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm": cm,
        "labels": SPECIES_ORDER
    }


@st.cache_data(show_spinner=False)
def run_random_forest_cached(iris_data, n_estimators=200):
    prepared = get_prepared_data(iris_data)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(prepared["X_train"], prepared["y_train_text"])
    y_pred = model.predict(prepared["X_test"])

    acc = accuracy_score(prepared["y_test_text"], y_pred)
    report_df = pd.DataFrame(
        classification_report(prepared["y_test_text"], y_pred, output_dict=True)
    ).transpose()
    cm = confusion_matrix(prepared["y_test_text"], y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm": cm,
        "labels": SPECIES_ORDER
    }


@st.cache_data(show_spinner=False)
def run_xgboost_cached(iris_data, n_estimators=BEST_XGB_N):
    if not XGB_AVAILABLE:
        return {"error": "当前环境未安装 xgboost。请先运行：pip install xgboost"}

    prepared = get_prepared_data(iris_data)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=FIXED_XGB_MAX_DEPTH,
        learning_rate=FIXED_XGB_LEARNING_RATE,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        verbosity=0
    )
    model.fit(prepared["X_train"], prepared["y_train_encoded"])
    y_pred = model.predict(prepared["X_test"])

    report_df = pd.DataFrame(
        classification_report(
            prepared["y_test_encoded"],
            y_pred,
            target_names=prepared["label_classes"],
            output_dict=True
        )
    ).transpose()
    cm = confusion_matrix(prepared["y_test_encoded"], y_pred)

    return {
        "accuracy": accuracy_score(prepared["y_test_encoded"], y_pred),
        "report_df": report_df,
        "cm": cm,
        "labels": prepared["label_classes"]
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
    st.markdown(f"""
    本交互式演示系统集成了数据预处理、可视化探索与模型评估的全流程，包含以下核心模块：
    1. **Data Overview (数据概览)**：展示鸢尾花数据集的基础统计信息与分布特征。
    2. **Feature Engineering (特征可视化)**：展示二维散点图、相关性热力图与 3D PCA 降维结果。
    3. **Model Comparison (模型比较)**：展示四种算法的参数-准确率曲线，以及最佳参数模型综合热力图。其中 XGBoost 部分仅调整 `n_estimators`，其余参数固定为 `max_depth={FIXED_XGB_MAX_DEPTH}`、`learning_rate={FIXED_XGB_LEARNING_RATE}`。
    4. **Interactive Demo (交互式演练)**：支持动态调整参数，并实时观察决策边界、混淆矩阵与分类指标变化。
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Feature Engineering",
    "Model Comparison",
    "Interactive Demo"
])

# =========================
# Tab 1
# =========================
with tab1:
    st.subheader("1. 数据概览")

    col1, col2, col3 = st.columns(3)
    col1.metric("总样本数", summary["samples"])
    col2.metric("特征数", summary["features"])
    col3.metric("类别数", summary["classes"])

    st.subheader("2. 四特征分布小提琴图")
    show_png(get_violin_png(iris_data), use_container_width=False)

    st.success("📊 图表解析：小提琴图展示了四个特征在三类样本中的分布形态。可以直观看出，花瓣长度与花瓣宽度的类别区分度明显高于花萼特征，因此它们通常提供更强的分类信息。")

    st.subheader("3. 原始数据预览")
    st.dataframe(iris_data, use_container_width=True, height=280)

# =========================
# Tab 2
# =========================
with tab2:
    st.subheader("1. 基础特征可视化")

    col1, col2 = st.columns(2)

    with col1:
        show_png(get_scatter_png(iris_data), use_container_width=False)
        st.success("📊 图表解析：Setosa 在花瓣尺寸上明显更小，与另外两类容易分开；Versicolour 与 Virginica 在边界附近存在一定重叠。")

    with col2:
        show_png(get_heatmap_png(iris_data), use_container_width=False)
        st.success("📊 图表解析：热力图展示了四个数值特征之间的相关性。花瓣长度与花瓣宽度相关性较强，说明这两个特征对分类非常关键。")

    st.divider()

    st.subheader("2. 3D PCA 降维展示")
    st.plotly_chart(get_pca_plotly_fig(iris_data), use_container_width=False)

    st.info("💡 降维分析：主成分分析（PCA）将 4 维特征映射至 3 维空间。可以看出，Setosa 形成了较为独立的聚簇，而 Versicolour 与 Virginica 的边界更接近。")

# =========================
# Tab 3
# =========================
with tab3:
    st.subheader("1. 四种算法准确率随参数变化曲线")
    show_png(get_baseline_curves_png(iris_data), use_container_width=False)

    st.info(
        f"💡 图表说明：上图分别展示了 KNN、SVM、Random Forest 和 XGBoost 的测试集准确率随关键参数变化的趋势，虚线标出了当前核心代码中采用的最佳参数位置。"
        f"其中，XGBoost 部分采用控制变量法，仅调整 n_estimators，其他参数固定为 max_depth={FIXED_XGB_MAX_DEPTH}、learning_rate={FIXED_XGB_LEARNING_RATE}。"
    )

    st.divider()

    st.subheader("2. 最佳参数模型综合性能热力图")
    show_png(get_best_model_heatmap_png(iris_data), use_container_width=False)

    st.info(
        "💡 图表解析：该热力图展示了四种分类算法在最佳参数上的性能对比，其中，KNN 和 SVM 的综合表现最好，两者在四项指标上结果完全一致，说明这两种模型在本实验中的分类能力最强，且整体表现较为稳定。"
    )

# =========================
# Tab 4
# =========================
def render_interactive_demo(iris_data):
    st.subheader("交互式演练")
    st.sidebar.header("⚙️ Model Settings")

    model_name = st.sidebar.selectbox(
        "Choose Classification Algorithm",
        ["KNN", "SVM", "Random Forest", "XGBoost"]
    )

    if model_name == "KNN":
        st.markdown("### K-Nearest Neighbors (KNN) Evaluation")
        k = st.sidebar.slider("Number of Neighbors (k)", min_value=1, max_value=20, value=5, step=1)
        result = run_knn_cached(iris_data, k=k)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

        st.subheader("决策边界图")
        show_png(get_decision_boundary_png(iris_data, "KNN", k=k), use_container_width=False)

        st.subheader("混淆矩阵")
        show_png(
            get_confusion_matrix_png(
                tuple(map(tuple, result["cm"].tolist())),
                tuple(result["labels"]),
                f"Confusion Matrix of KNN (k={k})"
            ),
            use_container_width=False
        )

        st.subheader("错判统计")
        st.info(get_objective_cm_text(result["cm"], result["labels"]))

        st.subheader("分类指标报告")
        st.dataframe(result["report_df"], use_container_width=True, height=260)

    elif model_name == "SVM":
        st.markdown("### Support Vector Machine (SVM) Evaluation")
        c_value = st.sidebar.selectbox("Regularization Parameter (C)", [0.01, 0.1, 1, 10, 100], index=2)
        result = run_svm_cached(iris_data, c_value=c_value)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

        st.subheader("决策边界图")
        show_png(get_decision_boundary_png(iris_data, "SVM", c_value=c_value), use_container_width=False)

        st.subheader("混淆矩阵")
        show_png(
            get_confusion_matrix_png(
                tuple(map(tuple, result["cm"].tolist())),
                tuple(result["labels"]),
                f"Confusion Matrix of SVM (C={c_value})"
            ),
            use_container_width=False
        )

        st.subheader("错判统计")
        st.info(get_objective_cm_text(result["cm"], result["labels"]))

        st.subheader("分类指标报告")
        st.dataframe(result["report_df"], use_container_width=True, height=260)

    elif model_name == "Random Forest":
        st.markdown("### Random Forest Evaluation")
        n_estimators = st.sidebar.slider("Number of Trees", min_value=10, max_value=500, value=200, step=10)
        result = run_random_forest_cached(iris_data, n_estimators=n_estimators)

        st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

        st.subheader("决策边界图")
        show_png(get_decision_boundary_png(iris_data, "Random Forest", n_estimators=n_estimators), use_container_width=False)

        st.subheader("混淆矩阵")
        show_png(
            get_confusion_matrix_png(
                tuple(map(tuple, result["cm"].tolist())),
                tuple(result["labels"]),
                f"Confusion Matrix of Random Forest (Trees={n_estimators})"
            ),
            use_container_width=False
        )

        st.subheader("错判统计")
        st.info(get_objective_cm_text(result["cm"], result["labels"]))

        st.subheader("分类指标报告")
        st.dataframe(result["report_df"], use_container_width=True, height=260)

    else:
        st.markdown("### XGBoost Evaluation")
        n_estimators = st.sidebar.slider(
            "Number of Boosting Rounds",
            min_value=10,
            max_value=300,
            value=BEST_XGB_N,
            step=10
        )

        st.info(
            f"说明：本实验在 XGBoost 部分仅调整 n_estimators，其余参数固定为 max_depth={FIXED_XGB_MAX_DEPTH}，learning_rate={FIXED_XGB_LEARNING_RATE}。"
        )

        result = run_xgboost_cached(iris_data, n_estimators=n_estimators)

        if "error" in result:
            st.error(result["error"])
        else:
            st.metric("Test Set Accuracy", f"{result['accuracy']:.4f}")

            st.subheader("决策边界图")
            show_png(
                get_decision_boundary_png(
                    iris_data,
                    "XGBoost",
                    n_estimators=n_estimators
                ),
                use_container_width=False
            )

            st.subheader("混淆矩阵")
            show_png(
                get_confusion_matrix_png(
                    tuple(map(tuple, result["cm"].tolist())),
                    tuple(result["labels"]),
                    "Confusion Matrix of XGBoost"
                ),
                use_container_width=False
            )

            st.subheader("错判统计")
            st.info(get_objective_cm_text(result["cm"], result["labels"]))

            st.subheader("分类指标报告")
            st.dataframe(result["report_df"], use_container_width=True, height=260)

with tab4:
    render_interactive_demo(iris_data)

st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 操作提示：请在上方下拉菜单中选择分类算法，并通过拖拽滑块实时调整模型超参数以观察分类效果。")
