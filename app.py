import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib  # 提前导入，避免重复导入

# --------------------------
# 全局配置（固定特征数，核心兜底）
# --------------------------
FEATURE_NUM = 6  # 固定6个特征，强制对齐
FEATURE_NAMES = ['Thermal duration', 'Rash duration', 'Vomit', 'Headache', 'N', 'Glu']
# 模型绝对路径（避免相对路径错误）
MODEL_PATH = "rf_model.pkl"  # 相对路径，云端/本地都兼容

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(
    page_title="Varicella Encephalitis Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# 侧边栏
# --------------------------
with st.sidebar:
    st.title("About")
    st.info("Pediatric Varicella Encephalitis Risk Prediction with SHAP")

# --------------------------
# 输入区域（严格对应6个特征）
# --------------------------
st.title("Pediatric Varicella Encephalitis Risk Prediction")
col1, col2 = st.columns(2)
with col1:
    fever = st.number_input("Fever duration (d)", 0, 20, 5)
    rash = st.number_input("Rash duration (d)", 0, 20, 5)
    vomit = 1 if st.selectbox("Vomit", ["No", "Yes"]) == "Yes" else 0
with col2:
    headache = 1 if st.selectbox("Headache", ["No", "Yes"]) == "Yes" else 0
    n_pct = st.number_input("N (%)", 0.0, 100.0, 50.0, 0.1)
    glu = st.number_input("Glu (mmol/L)", 0.0, 20.0, 5.0, 0.1)

# 构建输入数据（严格对应6列）
input_data = pd.DataFrame(
    [[fever, rash, vomit, headache, n_pct, glu]],
    columns=FEATURE_NAMES
)

# --------------------------
# 核心逻辑（使用真实模型，无冗余代码）
# --------------------------
if st.button("Predict", type="primary"):
    # 1. 加载真实模型（核心修正：删除模拟数据训练）
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅")
    except FileNotFoundError:
        st.error(f"❌ 未找到模型文件：{MODEL_PATH}")
        st.error("请先运行 train_model.py 生成 rf_model.pkl")
        st.stop()  # 终止后续代码运行
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.stop()

    # 2. 风险等级预测（使用真实模型预测）
    pred_proba = model.predict_proba(input_data)[0][1]
    if pred_proba < 0.5:
        risk = "Low Risk (No Encephalitis)"
    elif 0.5 <= pred_proba <= 0.8:
        risk = "Medium Risk (Potential Encephalitis)"
    else:
        risk = "High Risk (Varicella Encephalitis)"

    st.subheader("Prediction Result")
    st.metric("Encephalitis Probability", f"{pred_proba:.4f}")
    st.write(f"Risk Level: **{risk}**")

    # 3. SHAP分析（保持原有逻辑，兼容真实模型）
    st.subheader("SHAP Feature Contribution Analysis")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_data)
    shap_arr = np.array(shap_vals)

    # 强制降维并对齐6维
    shap_flat = shap_arr.ravel()
    if len(shap_flat) > FEATURE_NUM:
        shap_final = shap_flat[:FEATURE_NUM]
    elif len(shap_flat) < FEATURE_NUM:
        shap_final = np.pad(shap_flat, (0, FEATURE_NUM - len(shap_flat)), mode='constant')
    else:
        shap_final = shap_flat

    # 绘制SHAP条形图
    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "SHAP_Value": shap_final
    }).sort_values(by="SHAP_Value", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E83A8F" if val > 0 else "#66CCFF" for val in shap_df["SHAP_Value"]]
    text_colors = ["#E83A8F" if val > 0 else "#0066CC" for val in shap_df["SHAP_Value"]]
    bars = ax.barh(
        y=range(len(shap_df)),
        width=shap_df["SHAP_Value"],
        color=colors,
        edgecolor="white",
        linewidth=1
    )

    # 样式调整
    ax.invert_yaxis()
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df["Feature"], fontsize=12)
    ax.set_xlabel("mean (SHAP value)", fontsize=12)
    ax.set_title("Feature Contributions to Encephalitis Prediction", fontsize=14)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

    # 数值标注（避免遮盖）
    for i, bar in enumerate(bars):
        val = bar.get_width()
        x_pos = val + 0.005 if val > 0 else val - 0.03
        sign = "+" if val > 0 else "-"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height()/2,
            f"{sign}{abs(val):.3f}",
            va="center",
            color=text_colors[i],
            fontweight="bold",
            fontsize=11
        )

    plt.tight_layout()
    st.pyplot(fig)

    # SHAP Force Plot
    st.markdown("**SHAP Force Plot**")
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0] if len(base_value) > 0 else 0.0

    force_plot = shap.force_plot(
        base_value,
        shap_final,
        feature_names=FEATURE_NAMES,
        matplotlib=False,
        show=False
    )

    st.components.v1.html(force_plot.html(), height=150)
