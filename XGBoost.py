import time
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 忽略 XGBoost 的标签编码器警告 (保持控制台整洁)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. 全局配置 (CONFIG)
# 将所有“魔法数字”提取到这里，方便后续调整参数
# ==========================================
CONFIG = {
    "data_path": "bank_cleaned.csv",
    "target_column": "y",
    "test_size": 0.2,
    "random_state": 42,
    
    # 模型超参数
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    
    # 业务约束
    "target_precision": 0.3, 

    # === 新增：业务利润参数 ===
    "cost_per_call": 5,      # 打一通电话的成本 (元)
    "profit_per_success": 200, # 成功一单的净利润 (元)
}

class BankMarketingModel:
    """
    银行营销预测模型类
    包含数据加载、预处理、训练及基于业务约束的阈值调优逻辑
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.best_threshold = 0.5
        
    def load_and_split_data(self):
        """加载数据并进行训练集/测试集拆分"""
        print(f"📂 正在加载数据: {self.config['data_path']} ...")
        df = pd.read_csv(self.config['data_path'])
        
        y = df[self.config['target_column']]
        X = df.drop(self.config['target_column'], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        print(f"✅ 数据加载完成。训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
        return X_train, X_test, y_train, y_test
    
    def calculate_class_weight(self, y_train: pd.Series) -> float:
        """计算正负样本比例，用于处理数据不平衡"""
        negative_count = len(y_train[y_train == 0])
        positive_count = len(y_train[y_train == 1])
        weight = negative_count / positive_count
        print(f"⚖️ 数据不平衡比率 (scale_pos_weight): {weight:.2f}")
        return weight
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        训练 XGBoost 模型 (包含 GridSearchCV 自动调参)
        """
        print("\n🚀 正在启动 XGBoost 训练与调优...")
        
        scale_pos_weight_val = self.calculate_class_weight(y_train)
        
        # 1. 定义基础模型
        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight_val, # 🔥 核心：平衡权重
            random_state=self.config['random_state'],
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1 # 使用所有CPU核心加速搜索
        )
        
        # 2. 定义参数搜索空间 (根据第四步要求)
        param_grid = {
            'max_depth': [3, 5, 7],       # 树深：控制复杂度
            'learning_rate': [0.05, 0.1, 0.2], # 学习率：控制收敛速度
            'n_estimators': [100, 200]     # 树的数量：配合学习率
        }
        
        print("🔍 正在进行网格搜索 (GridSearchCV)，这可能需要几分钟...")
        
        # 3. 执行搜索 (cv=3 表示3折交叉验证， scoring='roc_auc' 关注排序能力)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='roc_auc', 
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 4. 获取最佳模型并保存到 self.model
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\n✅ 调优完成！最佳参数如下:")
        for param, value in best_params.items():
            print(f"   - {param}: {value}")
        
    def find_optimal_threshold(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        根据业务约束寻找最佳阈值
        目标：在 Precision >= target_precision 的前提下，最大化 Recall
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用 train() 方法。")
            
        print(f"\n🔍 正在寻找最佳阈值 (目标 Precision >= {self.config['target_precision']})...")
        
        # A. 获取预测概率
        y_scores = self.model.predict_proba(X_test)[:, 1]
        
        # B. 计算 P-R 曲线
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
        
        # C. 自动寻找满足条件的最佳阈值
        best_threshold = 0.5  # 默认回退值
        max_recall_at_constraint = 0.0
        
        for i in range(len(thresholds)):
            # 约束条件
            if precisions[i] >= self.config['target_precision']:
                # 优化目标
                if recalls[i] > max_recall_at_constraint:
                    max_recall_at_constraint = recalls[i]
                    best_threshold = thresholds[i]
        
        self.best_threshold = best_threshold
        
        print(f"✅ 找到最佳阈值: {best_threshold:.4f} (默认是 0.5)")
        print(f"   在此阈值下，Recall 提升到了: {max_recall_at_constraint:.4f}")
        
        return y_scores, best_threshold

    def evaluate(self, y_test: pd.Series, y_scores: np.ndarray):
        """使用最佳阈值生成报告"""
        y_pred_new = (y_scores >= self.best_threshold).astype(int)
        
        print("\n📊 调优后的分类报告:")
        print(classification_report(y_test, y_pred_new, target_names=['No Deposit', 'Deposit']))
        
        # 返回预测结果供后续步骤（如利润计算）使用
        return y_pred_new

    # ==========================================
    # 计算利润
    # ==========================================
    
    def calculate_profit(self, y_test, y_pred_default, y_pred_optimized):
        """
        计算并对比不同策略下的预期利润
        :param y_test: 真实标签
        :param y_pred_default: 默认阈值(0.5)的预测结果
        :param y_pred_optimized: 优化阈值后的预测结果
        """
        cost = self.config["cost_per_call"]
        profit = self.config["profit_per_success"]
        total_samples = len(y_test)
        actual_positives = y_test.sum()

        # --- 策略1: 盲打 (Baseline) ---
        # 假设不打模型，对所有人都打电话（或者按历史转化率盲打，这里简化为全量拨打作为最差/最贵基准）
        blind_cost = total_samples * cost
        blind_revenue = actual_positives * profit
        blind_net_profit = blind_revenue - blind_cost

        # --- 策略2: 默认模型 (Threshold 0.5) ---
        pred_count_default = y_pred_default.sum()
        true_pos_default = ((y_pred_default == 1) & (y_test == 1)).sum()
        
        default_cost = pred_count_default * cost
        default_revenue = true_pos_default * profit
        default_net_profit = default_revenue - default_cost

        # --- 策略3: 优化模型 (Optimized Threshold) ---
        pred_count_opt = y_pred_optimized.sum()
        true_pos_opt = ((y_pred_optimized == 1) & (y_test == 1)).sum()

        opt_cost = pred_count_opt * cost
        opt_revenue = true_pos_opt * profit
        opt_net_profit = opt_revenue - opt_cost

        # --- 打印利润报表 ---
        print("\n" + "="*60)
        print(f"💰 业务利润分析报告 (单位: 元)")
        print("="*60)
        print(f"{'策略':<15} | {'拨打人数':<8} | {'成功单数':<8} | {'总成本':<10} | {'总营收':<10} | {'净利润':<10}")
        print("-"*90)
        print(f"{'🛑 盲打 (基准)':<14} | {total_samples:<8} | {actual_positives:<8} | {blind_cost:<10} | {blind_revenue:<10} | {blind_net_profit:<10}")
        print(f"{'⚖️ 默认模型':<14} | {int(pred_count_default):<8} | {true_pos_default:<8} | {default_cost:<10} | {default_revenue:<10} | {default_net_profit:<10}")
        print(f"{'✅ 优化模型':<14} | {int(pred_count_opt):<8} | {true_pos_opt:<8} | {opt_cost:<10} | {opt_revenue:<10} | {opt_net_profit:<10}")
        print("-"*90)
        
        # 计算提升幅度
        lift_vs_blind = ((opt_net_profit - blind_net_profit) / abs(blind_net_profit)) * 100 if blind_net_profit != 0 else 0
        lift_vs_default = ((opt_net_profit - default_net_profit) / default_net_profit) * 100 if default_net_profit != 0 else 0
        
        print(f"🚀 相比盲打，利润提升: {lift_vs_blind:.2f}%")
        print(f"🚀 相比默认模型，利润提升: {lift_vs_default:.2f}%")
        print("="*60 + "\n")

    def explain_model(self, X_test):
        """
        使用 SHAP 进行模型可解释性分析
        包含：1. 全局重要性 (Summary Plot)
              2. 局部解释 (Force Plot) - 针对单个高概率客户
        """
        if self.model is None:
            print("❌ 模型未训练，无法解释！")
            return

        print("\n🔍 正在计算 SHAP 值 (可能需要一点时间)...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        # --- 1. 全局解释 (你已经完成的部分) ---
        print("📊 生成全局特征重要性图...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig("shap_global_importance.png", dpi=300)
        plt.show()
        print("✅ 图片已保存: shap_global_importance.png")

        # --- 2. 局部解释 (你缺失的部分 - 补全这里) ---
        print("\n🎯 正在生成局部解释 (Force Plot)...")
        
        # 选取第一个样本作为演示（或者你可以选一个预测概率最高的）
        # 这里为了演示方便，直接取测试集的第 0 个样本
        sample_index = 0 
        
        # 获取该样本的特征名
        feature_names = X_test.columns
        
        print(f"👉 正在分析第 {sample_index} 号客户的决策逻辑...")
        
        # 绘制 Force Plot
        # 注意：shap.force_plot 需要 matplotlib 后端支持，如果在 Jupyter 外运行通常需要 save_as_png
        shap.force_plot(
            explainer.expected_value, 
            shap_values[sample_index,:], 
            X_test.iloc[sample_index,:],
            matplotlib=True, # 强制使用 matplotlib 渲染，方便保存
            show=False
        )
        
        plt.title(f"SHAP Force Plot for Customer #{sample_index}")
        plt.tight_layout()
        plt.savefig("shap_force_plot_single_customer.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 局部解释图已保存: shap_force_plot_single_customer.png")

class ModelComparator:
    """
    模型利润对比器
    负责对比 LR, RF, XGB 三种模型在特定业务场景下的盈利能力
    """
    def __init__(self, config):
        self.config = config
        self.results = []

    def run_comparison(self, X_train, X_test, y_train, y_test):
        print("\n" + "="*70)
        print("🏆 开始进行三大模型利润大比拼...")
        print("="*70)
        
        # 定义要对比的模型配置
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,            # 调到 2000 彻底杜绝警告
                solver='lbfgs',           # 配合 class_weight 最稳定
                random_state=self.config['random_state'],
                class_weight='balanced' # 自动处理不平衡
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=self.config['random_state'],
                class_weight='balanced'
            ),
            "XGBoost (Ours)": xgb.XGBClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                random_state=self.config['random_state'],
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }

        for name, model in models.items():
            start_time = time.time()
            print(f"\n⏳ 正在训练和评估: {name} ...")
            
            # 1. 训练
            model.fit(X_train, y_train)
            
            # 2. 获取概率
            y_scores = model.predict_proba(X_test)[:, 1]
            
            # 3. 寻找最佳阈值 (复用之前的逻辑，这里简化处理)
            # 注意：为了公平对比，我们允许每个模型有自己的最佳阈值
            best_thresh = self._find_best_threshold_logic(y_test, y_scores)
            
            # 4. 生成预测
            y_pred_opt = (y_scores >= best_thresh).astype(int)
            y_pred_def = (y_scores >= 0.5).astype(int)
            
            # 5. 计算利润 (直接调用你之前写好的逻辑，或者在这里重写简易版)
            profit_data = self._calculate_single_model_profit(y_test, y_pred_def, y_pred_opt, name)
            
            elapsed = time.time() - start_time
            profit_data['Training Time (s)'] = f"{elapsed:.2f}"
            profit_data['Best Threshold'] = f"{best_thresh:.4f}"
            
            self.results.append(profit_data)
            print(f"✅ {name} 完成! 净利润: {profit_data['Optimized Net Profit']}")

        self.print_report()

    def _find_best_threshold_logic(self, y_test, y_scores):
        """简化的阈值寻找逻辑"""
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
        target_p = self.config['target_precision']
        best_t = 0.5
        max_r = 0
        for i in range(len(thresholds)):
            if precisions[i] >= target_p and recalls[i] > max_r:
                max_r = recalls[i]
                best_t = thresholds[i]
        return best_t

    def _calculate_single_model_profit(self, y_test, y_pred_def, y_pred_opt, name):
        """计算单个模型的利润数据"""
        cost = self.config["cost_per_call"]
        profit_unit = self.config["profit_per_success"]
        
        # 优化后策略数据
        opt_count = y_pred_opt.sum()
        opt_tp = ((y_pred_opt == 1) & (y_test == 1)).sum()
        opt_net = (opt_tp * profit_unit) - (opt_count * cost)
        
        return {
            "Model Name": name,
            "Optimized Net Profit": opt_net,
            "Calls Made (Opt)": int(opt_count),
            "Successes (Opt)": int(opt_tp)
        }

    def print_report(self):
        """打印最终的大比拼表格"""
        df_res = pd.DataFrame(self.results)
        
        print("\n\n" + "💰"*30)
        print("📊 三大模型利润大比拼 (Business Value Comparison)")
        print("💰"*30)
        print(df_res.to_string(index=False))
        print("\n💡 结论: ", end="")
        best_model = df_res.loc[df_res['Optimized Net Profit'].idxmax(), 'Model Name']
        print(f"👑 综合利润最高的是 [{best_model}]")
        print("="*70)

# ==========================================
# 主执行流程 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 初始化配置和数据
    project = BankMarketingModel(CONFIG)
    X_train, X_test, y_train, y_test = project.load_and_split_data()
    
    # 2. 训练基础模型
    project.train(X_train, y_train)
    
    # 3. 使用项目自带的逻辑寻找最佳阈值 (这一步会自动更新 project.best_threshold)
    y_scores, best_thresh = project.find_optimal_threshold(X_test, y_test)
    
    # 4. 生成两种预测结果用于对比
    # A. 优化后的预测 (使用我们算出的最佳阈值)
    y_pred_optimized = (y_scores >= best_thresh).astype(int)
    
    # B. 默认模型的预测 (使用标准的 0.5 阈值)
    y_pred_default = (y_scores >= 0.5).astype(int)
    
    # 5. 打印分类报告 (展示技术指标)
    print("\n" + "="*50)
    print("📊 优化后模型的分类报告:")
    print("="*50)
    project.evaluate(y_test, y_scores) 
    
    # 6. 计算并打印利润矩阵 (展示业务价值)
    project.calculate_profit(y_test, y_pred_default, y_pred_optimized)

    # 7. 启动模型利润大比拼（核心！）
    comparator = ModelComparator(CONFIG)
    comparator.run_comparison(X_train, X_test, y_train, y_test)

    # ==========================================
    # 🌟 新增：第8步与第9步
    # ==========================================
    # 8. SHAP 可解释性分析
    project.explain_model(X_test)
    
    # 9.PR 曲线深度对比
    # project.compare_with_lr(X_train, X_test, y_train, y_test)

    # ... 原有的最后几行 ...
    project.explain_model(X_test) 
    
    # === 在这里加上这两行 ===
    import joblib
    joblib.dump(project, 'trained_model.pkl') # 保存整个训练好的项目对象
    print("✅ 模型已保存为 trained_model.pkl")