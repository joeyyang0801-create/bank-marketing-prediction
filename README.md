# 🏦 银行营销预测 (Bank Marketing Prediction)

> **项目简介**：一个基于 XGBoost 的可解释性 AI 模型，用于预测客户是否会订阅定期存款，并结合利润优化和 SHAP 分析。
>
> **Project Overview:** An interpretable AI model for deposit subscription forecasting using XGBoost, focusing on profit optimization and SHAP analysis.

## 📌 项目描述 (Project Description)
本项目旨在根据营销活动的数据，预测客户是否会订阅定期存款（是/否）。与传统的只关注准确率的模型不同，本项目将**商业利润逻辑**整合到了模型评估中，并使用 **SHAP (SHapley Additive exPlanations)** 来确保模型的决策是透明且可解释的。

This project aims to predict whether a client will subscribe to a term deposit (Yes/No) based on marketing campaign data. Unlike traditional models that only focus on accuracy, this project integrates **business profit logic** into the model evaluation and uses **SHAP** to ensure the model's decisions are transparent and interpretable.

## 🚀 核心亮点 (Key Features)
- **数据预处理 (Data Preprocessing)**：对银行营销数据集进行了全面的清洗和特征工程 (`data_clean.py`)。
- **模型构建 (Modeling)**：使用 **XGBoost** 构建高性能分类模型。
- **利润优化 (Profit Optimization)**：不仅使用准确率/F1分数，还通过预估**净利润**来评估模型性能。
- **可解释性 (Interpretability)**：使用 SHAP 值解释全局特征重要性和单个客户的预测结果（力导向图）。

## 🛠️ 如何运行 (How to Run)

请按照以下步骤在本地运行此项目：
Please follow these steps to run the project locally:

### 1. 克隆仓库 (Clone Repository)
```bash
git clone https://github.com/joeyyang0801-create/bank-marketing-prediction.git
cd bank-marketing-prediction
```

### 2. 安装依赖 (Install Dependencies)
确保已安装 Python，然后运行以下命令安装所需的库：
Make sure you have Python installed, then run the following command to install required libraries:
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

### 3. 运行数据清洗 (Run Data Cleaning - Optional)
如果你想自己处理原始数据（可选）：
If you want to process the raw data yourself (Optional):
```bash
python data_clean.py
```

### 4. 运行主模型 (Run Main Model)
执行 XGBoost 训练和分析脚本：
Execute the XGBoost training and analysis script:
```bash
python XGBoost.py
```

📂 文件结构 (File Structure)
XGBoost.py: 用于训练、评估和可视化模型的主脚本。
data_clean.py: 用于数据清洗和预处理的脚本。
bank_cleaned.csv: 处理好的、可用于建模的数据集。
bank-additional-full.csv: 原始的完整数据集。
shap_global_importance.png: 可视化驱动预测的最重要特征。
shap_force_plot_single_customer.png: 单个客户预测结果的 SHAP 力导向图示例。

📊 关键洞察 (Key Insights)
(此处可以添加一句话描述你的图表发现了什么，例如：)
通话时长和上次营销活动的结果被发现是最重要的预测因素。
(Duration of call and previous campaign outcomes were found to be the most significant predictors.)

📈 结果 (Results)
模型 (Model): XGBoost 分类器 (XGBoost Classifier)
性能 (Performance): 在预测定期存款订阅方面表现出高准确率 (High accuracy in predicting term deposit subscriptions).
可解释性 (Interpretability): 生成了 SHAP 图（力导向图和摘要图）来解释特征的重要性 (SHAP plots generated to explain feature importance).
