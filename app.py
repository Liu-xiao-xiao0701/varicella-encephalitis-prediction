import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# ===================== 全局配置 =====================
FEATURE_NUM = 6
FEATURE_NAMES = ['Thermal duration', 'Rash duration', 'Vomit', 'Headache', 'N', 'Glu']
MODEL_PATH    = "rf_model.pkl"
SCALER_PATH   = "scaler.pkl"

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="Encephalitis Prediction", page_icon="🏥", layout="wide")

# ===================== 输入界面 =====================
st.title("Pediatric Varicella Encephalitis Prediction")
col1, col2 = st.columns(2)

with col1:
    fever     = st.number_input("Fever duration (d)", 0,20,5)
    rash      = st.number_input("Rash duration (d)", 0,20,5)
    vomit     = 1 if st.selectbox("Vomit", ["No","Yes"]) == "Yes" else 0

with col2:
    headache  = 1 if st.selectbox("Headache", ["No","Yes"]) == "Yes" else 0
    n_pct     = st.number_input("N (%)", 0.0,100.0,50.0,0.1)
    glu       = st.number_input("Glu (mmol/L)", 0.0,20.0,5.0,0.1)

# 构建原始输入
input_raw = pd.DataFrame(
    [[fever, rash, vomit, headache, n_pct, glu]],
    columns=FEATURE_NAMES
)

# ===================== 预测 + SHAP分析 =====================
if st.button("Predict", type="primary"):
    # 1. 加载模型 + 归一化工具（删除成功提示）
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # 这里删除了 st.success 提示
    except FileNotFoundError as e:
        st.error(f"❌ 未找到文件：{e.filename}")
        st.error("请先运行 train_model.py 生成模型和scaler文件")
        st.stop()
    except Exception as e:
        st.error(f"❌ 加载失败：{str(e)}")
        st.stop()

    # 2. 自动归一化
    input_scaled = scaler.transform(input_raw)

    # 3. 预测概率
    prob = model.predict_proba(input_scaled)[0][1]

    # 4. 风险等级判断
    if prob < 0.613:
        risk = "Low Risk"
        risk_color = "#155724"
        risk_bg = "#d4edda"
    else:
        risk = "High Risk"
        risk_color = "#721c24"
        risk_bg = "#f8d7da"

    # 5. 显示结果：先显示概率，再显示风险等级色块
    st.subheader("Result")
    st.metric("Encephalitis Probability", f"{prob:.4f}")
    # 风险等级放在概率下方
    st.markdown(f"""
    <div style="background-color: {risk_bg}; color: {risk_color}; padding: 10px; border-radius: 5px; font-size: 20px; font-weight: bold; margin-top: 10px;">
    Risk Level: {risk}
    </div>
    """, unsafe_allow_html=True)

    # 6. SHAP分析（彻底解决重叠问题）
    st.subheader("SHAP Feature Contribution Analysis")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_scaled)
    shap_arr = np.array(shap_vals)

    # 处理二分类SHAP值
    if shap_arr.ndim == 3 and shap_arr.shape[0] == 2:
        shap_arr = shap_arr[1]
    elif shap_arr.ndim > 2:
        shap_arr = shap_arr.squeeze()

    # 强制对齐6维特征
    shap_flat = shap_arr.ravel()
    if len(shap_flat) > FEATURE_NUM:
        shap_final = shap_flat[:FEATURE_NUM]
    elif len(shap_flat) < FEATURE_NUM:
        shap_final = np.pad(shap_flat, (0, FEATURE_NUM - len(shap_flat)), mode='constant')
    else:
        shap_final = shap_flat

    # 构建SHAP数据框
    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "SHAP_Value": shap_final
    }).sort_values(by="SHAP_Value", ascending=False)

# 绘制SHAP条形图（终极版：确保所有数值清晰显示）
fig, ax = plt.subplots(figsize=(12, 6))

# 优化条形颜色：加深蓝色，让白色文字更醒目
colors = ["#E83A8F" if val > 0 else "#3399FF" for val in shap_df["SHAP_Value"]]
text_colors = ["#FFFFFF" for _ in shap_df["SHAP_Value"]]  # 全部白色文字

bars = ax.barh(
    y=range(len(shap_df)),
    width=shap_df["SHAP_Value"],
    color=colors,
    edgecolor="white",
    linewidth=1
)

# 基础样式
ax.invert_yaxis()
ax.set_yticks(range(len(shap_df)))
ax.set_yticklabels(shap_df["Feature"], fontsize=14)
ax.set_xlabel("mean (SHAP value)", fontsize=14)
ax.set_title("Feature Contributions to Encephalitis Prediction", fontsize=16)
ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

# 核心标注逻辑：
# 1. 计算每个条形的中心位置，把数值放在正中间
# 2. 加粗文字，增大字号，确保在深色背景下清晰
for i, bar in enumerate(bars):
    val = bar.get_width()
    # 计算条形的中心x坐标
    bar_center_x = bar.get_x() + bar.get_width() / 2
    bar_center_y = bar.get_y() + bar.get_height() / 2
    
    sign = "+" if val > 0 else "-"
    label_text = f"{sign}{abs(val):.3f}"
    
    # 标注在条形正中心，白色加粗，字号12
    ax.text(
        bar_center_x,
        bar_center_y,
        label_text,
        va="center",
        ha="center",
        color="#FFFFFF",
        fontweight="bold",
        fontsize=12,
    )

# 自动适配x轴范围
left_limit = min(shap_final) - 0.02
right_limit = max(shap_final) + 0.02
ax.set_xlim(left=left_limit, right=right_limit)

plt.tight_layout()
st.pyplot(fig)
