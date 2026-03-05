# train_model.py —— 最终版
# 功能：读取【原始数据】→ 自动归一化 → 训练模型 → 保存模型+归一化工具
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ===================== 路径配置（不用改，我已帮你写好） =====================
EXCEL_PATH      = r"C:\Users\Xiao\Desktop\model_web\xl.xlsx"   # 原始数据
MODEL_PATH      = r"C:\Users\Xiao\Desktop\model_web\rf_model.pkl"
SCALER_PATH     = r"C:\Users\Xiao\Desktop\model_web\scaler.pkl"

FEATURE_NAMES   = ['Thermal duration', 'Rash duration', 'Vomit', 'Headache', 'N', 'Glu']
FEATURE_INDICES = [3,5,6,7,10,23]   # 你Excel里的列位置
LABEL_COLUMN    = "Group"

# ===================== 读取原始数据 =====================
df = pd.read_excel(EXCEL_PATH, engine='openpyxl')

# 提取特征
X_raw = df.iloc[:, FEATURE_INDICES]
X_raw.columns = FEATURE_NAMES
y     = df[LABEL_COLUMN]

# 删缺失值
X_raw = X_raw.dropna()
y = y.loc[X_raw.index]

# ===================== 自动归一化（关键！） =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ===================== 训练模型 =====================
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# ===================== 保存模型 + 归一化工具 =====================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ 训练完成！模型和归一化工具已保存")
print("✅ 现在可以直接用原始数据预测！")