import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn import datasets
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
# 1. 全局绘图风格
# =========================
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
# 3. 可视化函数
# =========================
def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="petallength",
        y="petalwidth",
        hue="species",
        hue_order=SPECIES_ORDER,
        s=90,
        palette="husl",
        edgecolor="black",
        alpha=0.85,
        ax=ax
    )
    ax.set_title("2D Scatter Distribution", fontweight="bold", pad=12)
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.legend(title="Species", loc="upper left")
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
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar=True,
        ax=ax,
        annot_kws={"size": 10, "color": "#222222"}
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11, "color": "#222222"}
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig


def plot_knn_curve(k_values, train_accs, test_accs, best_k):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, train_accs, marker="o", linewidth=2, label="Training Accuracy")
    ax.plot(k_values, test_accs, marker="s", linewidth=2, label="Testing Accuracy")
    ax.axvline(x=best_k, linestyle="--", linewidth=1.5, color="gray", label=f"Best k = {best_k}")
    ax.set_title("KNN Performance vs. Parameter k", fontweight="bold", pad=10)
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_svm_curve(C_values, train_accs, test_accs, best_c):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = list(range(len(C_values)))
    ax.plot(idx, train_accs, marker="o", linewidth=2, label="Training Accuracy")
    ax.plot(idx, test_accs, marker="s", linewidth=2, label="Testing Accuracy")
    ax.axvline(x=C_values.index(best_c), linestyle="--", linewidth=1.5, color="gray", label=f"Best C = {best_c}")
    ax.set_title("SVM Performance vs. Parameter C", fontweight="bold", pad=10)
    ax.set_xlabel("Parameter C")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(idx)
    ax.set_xticklabels(C_values)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        color="steelblue",
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
        color="steelblue",
        ax=ax
    )
    ax.set_title("Accuracy Comparison of Four Models", fontweight="bold", pad=10)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    for i, value in enumerate(compare_df["Accuracy"]):
        if pd.notna(value):
            ax.text(i, value + 0.01, f"{value:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    return fig

# =========================
# 4. 模型函数
# =========================
def run_knn(df, k=5):
    X_train, X_test, y_train, y_test = get_train_test_data(df)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    k_values = list(range(1, 21))
    train_accs = []
    test_accs = []

    for temp_k in k_values:
        temp_model = KNeighborsClassifier(n_neighbors=temp_k)
        temp_model.fit(X_train_std, y_train)

        y_train_pred = temp_model.predict(X_train_std)
        y_test_pred = temp_model.predict(X_test_std)

        train_accs.append(accuracy_score(y_train, y_train_pred))
        test_accs.append(accuracy_score(y_test, y_test_pred))

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix of KNN (k={k})"),
        "curve_fig": plot_knn_curve(k_values, train_accs, test_accs, k_values[int(np.argmax(test_accs))])
    }


def run_svm(df, c_value=1.0):
    X_train, X_test, y_train, y_test = get_train_test_data(df)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    C_values = [0.01, 0.1, 1, 10, 100]
    train_accs = []
    test_accs = []

    for c in C_values:
        temp_model = SVC(C=c, kernel="rbf", gamma="scale", random_state=42)
        temp_model.fit(X_train_std, y_train)

        y_train_pred = temp_model.predict(X_train_std)
        y_test_pred = temp_model.predict(X_test_std)

        train_accs.append(accuracy_score(y_train, y_train_pred))
        test_accs.append(accuracy_score(y_test, y_test_pred))

    model = SVC(C=c_value, kernel="rbf", gamma="scale", random_state=42)
    model.fit(X_train_std, y_train)
    y_pred = model.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix of SVM (C={c_value})"),
        "curve_fig": plot_svm_curve(C_values, train_accs, test_accs, C_values[int(np.argmax(test_accs))])
    }


def run_random_forest(df, n_estimators=200):
    X_train, X_test, y_train, y_test = get_train_test_data(df)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_test, y_pred, labels=SPECIES_ORDER)

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, SPECIES_ORDER, f"Confusion Matrix of Random Forest ({n_estimators} Trees)"),
        "importance_fig": plot_feature_importance(importance_df, "Feature Importance of Random Forest"),
        "importance_df": importance_df
    }


def run_xgboost(df, n_estimators=100, max_depth=3, learning_rate=0.1):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {
            "error": "当前环境未安装 xgboost。请先运行：pip install xgboost"
        }

    X = df[FEATURE_COLS]
    y = df["species"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
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
    report_dict = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    cm = confusion_matrix(y_test, y_pred)

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return {
        "accuracy": acc,
        "report_df": report_df,
        "cm_fig": plot_confusion_matrix(cm, label_encoder.classes_, "Confusion Matrix of XGBoost"),
        "importance_fig": plot_feature_importance(importance_df, "Feature Importance of XGBoost"),
        "importance_df": importance_df
    }


@st.cache_data
def compare_models(df):
    knn_result = run_knn(df, k=5)
    svm_result = run_svm(df, c_value=1.0)
    rf_result = run_random_forest(df, n_estimators=200)
    xgb_result = run_xgboost(df, n_estimators=100, max_depth=3, learning_rate=0.1)

    data = {
        "Model": ["KNN", "SVM", "Random Forest", "XGBoost"],
        "Accuracy": [
            knn_result["accuracy"],
            svm_result["accuracy"],
            rf_result["accuracy"],
            np.nan if "error" in xgb_result else xgb_result["accuracy"]
        ]
    }
    return pd.DataFrame(data)

# =========================
# 5. 页面主体
# =========================
df, summary = load_data()

st.title("🌸 Iris Classification Web Demo")
st.caption("Python课程项目｜鸢尾花数据分析、可视化与交互式分类系统")

st.markdown("""
### 项目概述
本系统整合了鸢尾花数据分析、可视化展示、模型对比与交互式分类结果演示。
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "数据概览", "可视化分析", "模型对比", "交互式演示"
])

with tab1:
    st.subheader("1. 数据集概览")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", summary["samples"])
    col2.metric("Features", summary["features"])
    col3.metric("Classes", summary["classes"])

    st.subheader("2. 萼片长度基础统计")
    stats_df = pd.DataFrame({
        "Statistic": ["均值", "中位数", "标准差", "5%分位数", "95%分位数"],
        "Value": [
            round(summary["sl_mean"], 2),
            round(summary["sl_median"], 2),
            round(summary["sl_std"], 2),
            round(summary["sl_pct_5"], 2),
            round(summary["sl_pct_95"], 2)
        ]
    })
    st.dataframe(stats_df, use_container_width=True)

    st.subheader("3. 物种数量统计")
    st.dataframe(df["species"].value_counts().rename_axis("物种").reset_index(name="数量"), use_container_width=True)

    st.subheader("4. 原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.subheader("数据可视化")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_scatter(df), use_container_width=True)
        st.info("Setosa 与另外两类区分明显，而 Versicolour 和 Virginica 存在一定重叠。")
    with col2:
        st.pyplot(plot_heatmap(df), use_container_width=True)
        st.info("Petal length 与 petal width 相关性较强，说明花瓣相关特征对分类更重要。")

with tab3:
    st.subheader("准确率对比")
    compare_df = compare_models(df)
    st.dataframe(compare_df, use_container_width=True)
    st.pyplot(plot_compare_bar(compare_df), use_container_width=True)
    st.caption("该图展示了四种分类模型的整体准确率对比结果。")

with tab4:
    st.subheader("交互式模型演示")
    st.sidebar.header("模型设置")

    model_name = st.sidebar.selectbox(
        "选择模型",
        ["KNN", "SVM", "Random Forest", "XGBoost"]
    )

    if model_name == "KNN":
        st.markdown("### KNN 模型演示")
        k = st.sidebar.slider("选择 k 值", min_value=1, max_value=20, value=5, step=1)
        result = run_knn(df, k=k)

        st.metric("准确率", f"{result['accuracy']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["curve_fig"], use_container_width=True)
        with col2:
            st.pyplot(result["cm_fig"], use_container_width=True)
        st.subheader("分类报告")
        st.dataframe(result["report_df"], use_container_width=True)

    elif model_name == "SVM":
        st.markdown("### SVM 模型演示")
        c_value = st.sidebar.selectbox("选择参数 C", [0.01, 0.1, 1, 10, 100], index=2)
        result = run_svm(df, c_value=c_value)

        st.metric("准确率", f"{result['accuracy']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["curve_fig"], use_container_width=True)
        with col2:
            st.pyplot(result["cm_fig"], use_container_width=True)
        st.subheader("分类报告")
        st.dataframe(result["report_df"], use_container_width=True)

    elif model_name == "Random Forest":
        st.markdown("### 随机森林模型演示")
        n_estimators = st.sidebar.slider("树的数量", min_value=10, max_value=500, value=200, step=10)
        result = run_random_forest(df, n_estimators=n_estimators)

        st.metric("准确率", f"{result['accuracy']:.4f}")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(result["cm_fig"], use_container_width=True)
        with col2:
            st.pyplot(result["importance_fig"], use_container_width=True)
        st.subheader("分类报告")
        st.dataframe(result["report_df"], use_container_width=True)
        st.subheader("特征重要性")
        st.dataframe(result["importance_df"], use_container_width=True)

    else:
        st.markdown("### XGBoost 模型演示")
        n_estimators = st.sidebar.slider("XGBoost 树的数量", min_value=50, max_value=300, value=100, step=10)
        max_depth = st.sidebar.slider("最大深度", min_value=2, max_value=8, value=3, step=1)
        learning_rate = st.sidebar.select_slider("学习率", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)

        result = run_xgboost(df, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

        if "error" in result:
            st.error(result["error"])
        else:
            st.metric("准确率", f"{result['accuracy']:.4f}")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(result["cm_fig"], use_container_width=True)
            with col2:
                st.pyplot(result["importance_fig"], use_container_width=True)
            st.subheader("分类报告")
            st.dataframe(result["report_df"], use_container_width=True)
            st.subheader("特征重要性")
            st.dataframe(result["importance_df"], use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("可在左侧边栏调整参数，并实时查看当前模型结果。")
