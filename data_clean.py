import pandas as pd
import numpy as np

# ==========================================
# 第一步：数据导入
# ==========================================

# 1. 设置显示选项，防止数据太多打印不全
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("🚀 程序开始运行...")

try:
    # 2. 读取数据
    # 注意：这里假设 csv 文件和 main.py 在同一个文件夹里
    file_path = 'bank-additional-full.csv' 
    
    print(f"正在读取文件: {file_path} ...")
    df = pd.read_csv(file_path, sep=';') # 这个数据集通常是用分号 ; 分隔的
    
    # 3. 打印基本信息
    print("\n✅ 数据读取成功！")
    print("-" * 30)
    print("数据的前 5 行 (Head):")
    print(df.head())
    
    print("-" * 30)
    print(f"数据集形状 (行数, 列数): {df.shape}")
    
    print("-" * 30)
    print("数据类型概览:")
    print(df.info())

except FileNotFoundError:
    print("❌ 错误：找不到文件 'bank-additional-full.csv'")
    print("请确保该文件和 main.py 在同一个文件夹下。")
except Exception as e:
    print(f"❌ 发生未知错误: {e}")

print("\n🏁 程序运行结束。")

# ==========================================
# 第二步：数据清洗与预处理
# ==========================================
print("\n🧹 开始数据清洗...")

# 1. 处理目标变量 (Target Variable)
# 模型无法理解 "yes" 或 "no"，我们需要把它变成 1 和 0
# yes = 1 (客户订阅了), no = 0 (客户没订阅)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 2. 处理缺失值 (Missing Values) - 优化版
print("正在智能处理缺失值...")

# 设定阈值：缺失率低于 5% 的视为“低缺失”，高于 5% 的视为“高缺失”
MISSING_THRESHOLD = 0.05 

for col in df.columns:
    missing_count = df[col].isnull().sum()
    
    if missing_count > 0:
        missing_ratio = missing_count / len(df)
        
        # --- 针对分类变量 (Object类型) ---
        if df[col].dtype == 'object':
            if missing_ratio < MISSING_THRESHOLD:
                # 策略 A：低缺失率 -> 填充为 'Unknown'，保留其作为独立类别的特征
                # 注意：这比填充众数更能反映真实情况（比如客户故意不填）
                df[col].fillna('Unknown', inplace=True)
                print(f"  [保留特征] {col}: 缺失率 {missing_ratio:.2%} -> 填充为 'Unknown'")
            else:
                # 策略 B：高缺失率 -> 填充众数，避免产生一个巨大的噪声类别破坏分布
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  [众数填充] {col}: 缺失率 {missing_ratio:.2%} -> 填充为众数")
                
        # --- 针对数值变量 (保持原样，通常数值型不适合填 'Unknown') ---
        else: 
            df[col].fillna(df[col].median(), inplace=True)

# 3. 编码分类变量 (Categorical Encoding)
# 模型不认识 "blue-collar", "admin." 这种文字，得变成数字
# 这里使用 One-Hot Encoding (独热编码)，这是最常用的方法
df = pd.get_dummies(df, drop_first=True)

# 4. 删除一些可能引起数据泄露或无用的列
# 比如 'duration' (通话时长)，因为如果你还没打电话，是不可能知道时长的
# 如果 'duration' 在你的列里，建议取消下面这行的注释来删除它
df.drop('duration', axis=1, inplace=True)

print("✅ 清洗完成！")
print(f"现在的列数量: {len(df.columns)}")
print(f"现在的行数: {len(df)}")
print("-" * 30)
print("清洗后的前5行数据预览:")
print(df.head())

# ==========================================
# 第三步：保存清洗后的数据 (存档)
# ==========================================

# 1. 保存为 CSV 文件
# index=False 表示不要把 Pandas 自动生成的 0,1,2... 行号存进去
output_filename = 'bank_cleaned.csv'
df.to_csv(output_filename, index=False)

print(f"✅ 数据已保存为: {output_filename}")
print("你可以随时用 pd.read_csv('bank_cleaned.csv') 重新加载它！")