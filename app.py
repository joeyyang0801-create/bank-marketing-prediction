import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. 页面配置与标题
# ==========================================
st.set_page_config(page_title="银行营销预测系统", layout="wide")
st.title("🏦 银行定期存款营销预测系统")
st.markdown("基于 XGBoost + 利润矩阵优化的智能营销辅助工具")

# ==========================================
# 2. 定义模型类 (为了兼容 joblib 加载)
# ==========================================
class BankMarketingModel:
    def __init__(self):
        self.model = None
        self.best_threshold = 0.5
        # 这里的属性会在 load_model 时被 pkl 文件里的内容覆盖

# ==========================================
# 3. 加载模型
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'trained_model.pkl'
    if not os.path.exists(model_path):
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()

if model is None:
    st.stop() # 如果没模型，直接停止，不往下跑

st.sidebar.success("✅ 模型加载成功！")

# ==========================================
# 4. 侧边栏：上传文件
# ==========================================
st.sidebar.header("1. 数据上传")
uploaded_file = st.sidebar.file_uploader("上传待预测的 CSV 文件", type=["csv"])

# 提供示例下载
if os.path.exists('bank-additional-full.csv'):
    with open('bank-additional-full.csv', 'rb') as f:
        st.sidebar.download_button(
            label="📥 下载示例数据",
            data=f,
            file_name="bank_sample.csv",
            mime="text/csv"
        )

# ==========================================
# 5. 主界面逻辑 (使用 if-else 代替复杂的 try 嵌套)
# ==========================================
if uploaded_file is not None:
    # A. 读取数据
    try:
        df_new = pd.read_csv(uploaded_file, sep=';') 
        st.header("2. 数据预览")
        st.dataframe(df_new.head())
    except Exception as e:
        st.error(f"读取文件失败，请检查分隔符是否为分号(;)。错误信息: {e}")
        st.stop()

    # B. 预测按钮逻辑
    if st.button("🚀 开始预测"):
        st.info("正在处理数据并预测...")
        
        try:
            # --- 数据清洗流程 (必须与训练时一致) ---
            df_process = df_new.copy()

            # 1. 处理目标变量 y
            if 'y' in df_process.columns:
                df_process['y'] = df_process['y'].map({'yes': 1, 'no': 0}).fillna(0)

            # 2. 填充缺失值
            for col in df_process.columns:
                if df_process[col].isnull().sum() > 0:
                    if df_process[col].dtype == 'object':
                        df_process[col].fillna('Unknown', inplace=True)
                    else:
                        df_process[col].fillna(df_process[col].median(), inplace=True)

            # 3. 独热编码
            df_process = pd.get_dummies(df_process, drop_first=True)

            # 4. 删除 duration (防止泄露)
            if 'duration' in df_process.columns:
                df_process.drop('duration', axis=1, inplace=True)

            # 5. 自动对齐特征列 (核心修复点)
            # 尝试从模型获取特征名，如果获取不到，就假设输入数据已经对齐
            if hasattr(model, 'feature_names_in_'):
                target_cols = [c for c in model.feature_names_in_ if c not in ['y', 'y_yes']]
                df_final = df_process.reindex(columns=target_cols, fill_value=0)
            else:
                # 兜底方案：如果没有 feature_names_in_，直接用处理后的数据（可能会报错，取决于模型内部实现）
                # 这里我们尝试去掉 y 列直接预测
                if 'y' in df_process.columns:
                    df_final = df_process.drop('y', axis=1)
                elif 'y_yes' in df_process.columns:
                    df_final = df_process.drop('y_yes', axis=1)
                else:
                    df_final = df_process

            # --- 开始预测 ---
            probabilities = model.model.predict_proba(df_final)[:, 1]
            
            # 获取阈值
            threshold = getattr(model, 'best_threshold', 0.5)
            predictions = (probabilities >= threshold).astype(int)
            
            # --- 展示结果 ---
            st.success(f"✅ 预测完成！共处理 {len(df_final)} 条数据。")
            
            result_df = df_new.copy() 
            result_df['预测概率'] = probabilities.round(4)
            result_df['预测结果'] = pd.Series(predictions).map({1: "会订阅 (Yes)", 0: "不会订阅 (No)"})
            
            # 高亮显示
            styled_df = result_df.style.map(
                lambda x: "background-color: #d4edda" if x == "会订阅 (Yes)" else "",
                subset=["预测结果"]
            )
            st.dataframe(result_df.head(100).style.map(
            lambda x: "background-color: #d4edda" if x == "会订阅 (Yes)" else "",
            subset=["预测结果"]
            ))
            
            # 统计指标
            col1, col2 = st.columns(2)
            col1.metric("预计订阅人数", int(predictions.sum()))
            col2.metric("预计转化率", f"{predictions.mean():.2%}")

            # ==========================================
            # ✅ 【关键修改】把下载按钮放在这里！
            # 必须缩进在 if st.button 内部，确保只有预测成功后才显示
            # ==========================================
            st.divider()
            st.download_button(
                label="📥 下载完整预测结果 (CSV)",
                data=result_df.to_csv(index=False).encode('utf-8-sig'), # 使用 utf-8-sig 防止 Excel 打开中文乱码
                file_name='prediction_result.csv',
                mime='text/csv',
            )


        except Exception as e:
            st.error(f"❌ 预测过程中出错: {str(e)}")
            st.exception(e) # 打印详细报错堆栈，方便调试

else:
    st.info("👈 请在左侧上传 CSV 文件开始预测")

