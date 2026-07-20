# -*- coding: utf-8 -*-
"""
鸢尾花分类识别系统
====================
涉及知识:
    - K 近邻 (K-Nearest Neighbors, KNN)
    - 逻辑回归 (Logistic Regression)
    - 模型评价方法 (准确率、精确率、召回率、F1 值)
    - 数据标准化 (StandardScaler)
    - 特征工程
    - tkinter 图形界面

项目流程:
    1. 加载 sklearn 自带的鸢尾花数据集
    2. 数据探索与基本统计信息
    3. 数据标准化处理 (特征工程)
    4. 划分训练集与测试集
    5. 分别训练 KNN 和 Logistic Regression 模型
    6. 在测试集上评估两个模型的性能
    7. 提供 GUI 界面,允许用户输入 4 个特征进行预测
"""

# ============================================================
# 1. 导入必要的库
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 必须在 import pyplot 之前设置
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris                       # 鸢尾花数据集
from sklearn.preprocessing import StandardScaler            # 数据标准化
from sklearn.model_selection import train_test_split        # 划分训练/测试集
from sklearn.neighbors import KNeighborsClassifier          # K 近邻
from sklearn.linear_model import LogisticRegression         # 逻辑回归
from sklearn.metrics import (
    accuracy_score,                                         # 准确率
    precision_score,                                        # 精确率
    recall_score,                                           # 召回率
    f1_score,                                               # F1 值
    classification_report,                                  # 完整分类报告
    confusion_matrix,                                      # 混淆矩阵
)

import tkinter as tk
from tkinter import ttk, messagebox
import warnings

# 忽略一些 sklearn 的版本警告,保持输出整洁
warnings.filterwarnings("ignore")

# 设置中文字体 (matplotlib)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 加载数据 & 数据探索
# ============================================================
def load_and_explore_data():
    """加载鸢尾花数据集并进行简单的数据探索"""
    iris = load_iris()

    # 特征矩阵 X (150 行 × 4 列): 花萼长/宽、花瓣长/宽
    X = iris.data
    # 标签向量 y (150 个): 0=setosa, 1=versicolor, 2=virginica
    y = iris.target
    # 类别名称
    target_names = iris.target_names
    # 特征名称
    feature_names = iris.feature_names

    print("=" * 60)
    print("【数据探索】鸢尾花数据集基本信息")
    print("=" * 60)
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}  (分别为: {feature_names})")
    print(f"类别数量: {len(target_names)}  (分别为: {list(target_names)})")
    print(f"每个类别的样本数: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 用 pandas 展示前 5 行,方便直观查看
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[i] for i in y]
    print("\n数据前 5 行:")
    print(df.head())
    print("\n数据统计信息:")
    print(df.describe())

    return X, y, feature_names, target_names


# ============================================================
# 3. 特征工程: 数据标准化
# ============================================================
def preprocess_data(X, y, test_size=0.25, random_state=42):
    """
    数据预处理
        1) 划分训练集和测试集
        2) 使用 StandardScaler 进行标准化 (z-score)
    注意: 标准化必须在划分后只 fit 训练集,
          然后用同一个 scaler 去 transform 测试集,避免数据泄露。
    """
    # 划分数据集,stratify=y 保证每个类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 标准化: 让每个特征的均值为 0, 标准差为 1
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)  # 训练集 fit + transform
    X_test_std = scaler.transform(X_test)        # 测试集只 transform

    return X_train, X_test, X_train_std, X_test_std, y_train, y_test, scaler


# ============================================================
# 4. 训练 K 近邻模型
# ============================================================
def train_knn(X_train, y_train, n_neighbors=5):
    """训练 K 近邻分类器,默认 k=5"""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski", p=2)
    knn.fit(X_train, y_train)
    return knn


# ============================================================
# 5. 训练逻辑回归模型
# ============================================================
def train_logistic_regression(X_train, y_train, max_iter=200):
    """
    训练多项式逻辑回归 (sklearn 1.5+ 默认就是 multinomial)
    solver='lbfgs' 适合小数据集,支持多分类
    """
    lr = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train, y_train)
    return lr


# ============================================================
# 6. 模型评价: 准确率 / 精确率 / 召回率 / F1
# ============================================================
def evaluate_model(model, X_test, y_test, target_names, model_name="Model"):
    """
    计算并打印模型的评价指标
        - accuracy  准确率: 预测正确的样本占全部样本的比例
        - precision 精确率 (macro 平均): 各类精确率的算术平均
        - recall    召回率 (macro 平均): 各类召回率的算术平均
        - f1        F1 值  (macro 平均): 精确率和召回率的调和平均
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # average='macro' 适用于类别均衡的数据集 (本数据集就是均衡的)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("=" * 60)
    print(f"【{model_name}】模型评价结果")
    print("=" * 60)
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}  (macro)")
    print(f"召回率 (Recall)   : {rec:.4f}  (macro)")
    print(f"F1 值   (F1-Score): {f1:.4f}  (macro)")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    print("混淆矩阵 (行=真实, 列=预测):")
    print(confusion_matrix(y_test, y_pred))

    return {"name": model_name, "acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ============================================================
# 7. 可视化对比两个模型
# ============================================================
def plot_comparison(metrics_list):
    """用柱状图直观对比两个模型的评价指标"""
    names = [m["name"] for m in metrics_list]
    accs = [m["acc"] for m in metrics_list]
    precs = [m["prec"] for m in metrics_list]
    recs = [m["rec"] for m in metrics_list]
    f1s = [m["f1"] for m in metrics_list]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5 * width, accs, width, label="准确率")
    ax.bar(x - 0.5 * width, precs, width, label="精确率")
    ax.bar(x + 0.5 * width, recs, width, label="召回率")
    ax.bar(x + 1.5 * width, f1s, width, label="F1 值")

    ax.set_ylabel("分数")
    ax.set_title("KNN 与 逻辑回归 在鸢尾花数据集上的性能对比")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.8, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(accs):
        ax.text(i - 1.5 * width, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(precs):
        ax.text(i - 0.5 * width, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(recs):
        ax.text(i + 0.5 * width, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(f1s):
        ax.text(i + 1.5 * width, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=120)
    plt.show()
    print("\n[提示] 对比图已保存为 model_comparison.png")


# ============================================================
# 8. 图形界面 (GUI): 输入 4 个特征 -> 预测花的种类
# ============================================================
class IrisGUI:
    """
    简单的 Tkinter 图形界面
        - 4 个输入框: 花萼长度、花萼宽度、花瓣长度、花瓣宽度
        - 选择模型 (KNN / Logistic Regression)
        - 点击"预测"按钮,输出花的种类以及各类概率
    """

    def __init__(self, knn_model, lr_model, scaler, target_names, feature_names):
        self.knn_model = knn_model
        self.lr_model = lr_model
        self.scaler = scaler
        self.target_names = target_names
        self.feature_names = feature_names

        # 主窗口
        self.root = tk.Tk()
        self.root.title("鸢尾花分类识别系统")
        self.root.geometry("520x420")
        self.root.resizable(False, False)

        # 标题
        title = tk.Label(
            self.root, text="鸢尾花分类识别系统",
            font=("Microsoft YaHei", 16, "bold"), fg="#2c3e50"
        )
        title.pack(pady=10)

        # 模型选择
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=5)
        tk.Label(model_frame, text="选择模型:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        self.model_choice = tk.StringVar(value="KNN")
        ttk.Combobox(
            model_frame, textvariable=self.model_choice,
            values=["KNN", "Logistic Regression"], state="readonly", width=20
        ).pack(side=tk.LEFT, padx=5)

        # 输入框
        self.entries = []
        input_frame = tk.LabelFrame(self.root, text="请输入花的 4 个特征 (cm)",
                                    font=("Microsoft YaHei", 10), padx=10, pady=10)
        input_frame.pack(padx=20, pady=10, fill="x")

        # 默认值: 一个常见的 setosa 样本
        defaults = [5.1, 3.5, 1.4, 0.2]
        for i, (name, default) in enumerate(zip(self.feature_names, defaults)):
            row = tk.Frame(input_frame)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{name}:", width=20, anchor="w",
                     font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
            entry = tk.Entry(row, width=20)
            entry.insert(0, str(default))
            entry.pack(side=tk.LEFT)
            self.entries.append(entry)

        # 按钮
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(
            btn_frame, text="预 测", width=12, command=self.predict,
            bg="#27ae60", fg="white", font=("Microsoft YaHei", 10, "bold")
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame, text="清 空", width=12, command=self.clear,
            bg="#95a5a6", fg="white", font=("Microsoft YaHei", 10, "bold")
        ).pack(side=tk.LEFT, padx=5)

        # 结果显示
        self.result_label = tk.Label(
            self.root, text="", font=("Microsoft YaHei", 12, "bold"),
            fg="#c0392b", wraplength=480, justify="left"
        )
        self.result_label.pack(pady=10)

    def predict(self):
        """点击预测按钮: 读取 4 个输入 -> 标准化 -> 预测 -> 显示结果"""
        try:
            # 1) 读取输入
            values = [float(e.get()) for e in self.entries]
        except ValueError:
            messagebox.showerror("输入错误", "请输入合法的数字！")
            return

        # 2) 转成 numpy 数组并标准化
        x = np.array(values).reshape(1, -1)
        x_std = self.scaler.transform(x)

        # 3) 选择模型
        if self.model_choice.get() == "KNN":
            model = self.knn_model
            model_name = "K 近邻 (KNN)"
        else:
            model = self.lr_model
            model_name = "逻辑回归 (Logistic Regression)"

        # 4) 预测类别
        pred_class = model.predict(x_std)[0]
        pred_species = self.target_names[pred_class]

        # 5) 计算概率 (KNN 用投票比例, LR 用 predict_proba)
        if self.model_choice.get() == "KNN":
            # KNN 的 predict_proba 返回各类别的"近邻占比",可作为置信度
            probs = model.predict_proba(x_std)[0]
        else:
            probs = model.predict_proba(x_std)[0]

        # 6) 拼成漂亮的文字
        result_text = (
            f"使用模型: {model_name}\n"
            f"预测结果: 【{pred_species}】\n"
            f"各类置信度: "
        )
        for name, p in zip(self.target_names, probs):
            result_text += f"{name}={p*100:.1f}%  "
        self.result_label.config(text=result_text)

    def clear(self):
        """清空所有输入框"""
        for e in self.entries:
            e.delete(0, tk.END)

    def run(self):
        """启动 GUI 消息循环"""
        self.root.mainloop()


# ============================================================
# 9. 主函数: 串联整个流程
# ============================================================
def main():
    # 1) 加载并探索数据
    X, y, feature_names, target_names = load_and_explore_data()

    # 2) 数据标准化 + 划分训练/测试集
    X_train, X_test, X_train_std, X_test_std, y_train, y_test, scaler = preprocess_data(X, y)

    # 3) 训练两个模型 (使用标准化后的数据)
    knn = train_knn(X_train_std, y_train, n_neighbors=5)
    lr = train_logistic_regression(X_train_std, y_train)

    # 4) 评价两个模型
    knn_metrics = evaluate_model(knn, X_test_std, y_test, target_names, model_name="K 近邻 (KNN)")
    lr_metrics = evaluate_model(lr, X_test_std, y_test, target_names, model_name="逻辑回归 (Logistic Regression)")

    # 5) 可视化对比
    try:
        plot_comparison([knn_metrics, lr_metrics])
    except Exception as e:
        # 在无 GUI 的服务器上 matplotlib 可能报错,这里兜底
        print(f"[警告] 绘图失败: {e}")

    # 6) 启动 GUI
    print("\n[提示] 即将打开图形界面,输入 4 个特征即可预测花的种类...")
    gui = IrisGUI(knn, lr, scaler, target_names, feature_names)
    gui.run()


if __name__ == "__main__":
    main()
