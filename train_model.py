# train_model.py - 固定绝对路径，避免路径错误
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# ===================== 1. 固定你的Excel文件绝对路径（关键！） =====================
EXCEL_PATH = r"C:\Users\Xiao\Desktop\model_web\xl_cl.xlsx"  # r前缀避免转义字符问题
FEATURE_NAMES = ['Thermal duration', 'Rash duration', 'Vomit', 'Headache', 'N', 'Glu']
FEATURE_INDICES = [3, 5, 6, 7, 10, 23]
LABEL_COLUMN = "Group"  # 替换为你Excel中实际的标签列名（如“脑炎”/“Encephalitis”）

# ===================== 2. 读取并处理数据 =====================
try:
    # 读取Excel（指定engine确保兼容.xlsx）
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    print(f"✅ 数据读取成功！文件路径：{EXCEL_PATH}")
    print(f"数据总行数：{len(df)}，总列数：{len(df.columns)}")
except FileNotFoundError:
    print(f"❌ 错误：未找到文件，请检查路径是否正确：{EXCEL_PATH}")
    exit()

# 按索引提取特征列
X = df.iloc[:, FEATURE_INDICES]
X.columns = FEATURE_NAMES
y = df[LABEL_COLUMN]

# 数据清洗
X = X.dropna()
y = y.loc[X.index]
print(f"✅ 数据清洗完成！有效样本数：{len(X)}")

# ===================== 3. 训练模型 =====================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    n_jobs=-1
)
model.fit(X, y)
print("✅ 模型训练完成！")

# ===================== 4. 保存模型（保存到同目录） =====================
MODEL_SAVE_PATH = r"C:\Users\Xiao\Desktop\model_web\rf_model.pkl"
joblib.dump(model, MODEL_SAVE_PATH)
print(f"✅ 模型已保存至：{MODEL_SAVE_PATH}")

# 验证特征重要性
print("\n📊 特征重要性（验证提取是否正确）：")
for name, imp in zip(FEATURE_NAMES, model.feature_importances_):
    print(f"{name}: {imp:.4f}")