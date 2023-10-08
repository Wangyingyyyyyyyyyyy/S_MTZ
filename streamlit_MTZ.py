# 导入需要的库
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

if 'model' not in st.session_state:
    model = joblib.load('model.pkl')
    st.session_state["model"] = model
else:
    model = st.session_state["model"]

st.set_page_config(layout="wide")


def set_background():
    page_bg_img = '''
    <style>
    .css-1nnpbs {width: 100vw}
    h1 {padding: 0.75rem 0px 0.75rem;margin-top: 2rem;box-shadow: 0px 3px 5px gray;}
    h2 {background-color: #B266FF;margin-top: 2vh;border-left: red solid 0.6vh}
    .css-1avcm0n {background: rgb(14, 17, 23, 0)}
    .css-18ni7ap {background: #B266FF;z-index:1;height:3rem}
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    .css-z5fcl4 {padding: 0 1rem 1rem}
    .css-12ttj6m {box-shadow: 0.05rem 0.05rem 0.2rem 0.1rem rgb(192, 192, 192);margin:0 calc(20% + 0.5rem);}
    .css-1cbqeqj {text-align: center;}
    .css-1x8cf1d {background: #00800082}
    .css-1x8cf1d:hover {background: #00800033}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background()
st.markdown("<h1 style='text-align: center'>Helicobacter pylori Metronidazole Resistance Prediction Based on a Machine Learning Model</h1>", unsafe_allow_html=True)
for i in range(3):
    st.write("")
with st.form("my_form"):
    col7, col8 = st.columns([5, 5])
    with col7:
        a = st.selectbox("HP_0040 Val241Ile", ("presence", "absence"))
        b = st.selectbox("HP_0642 Ala70fs", ("presence", "absence"))
        c = st.selectbox("HP_0887 Thr439Ser", ("presence", "absence"))
        d = st.selectbox("group_913", ("presence", "absence"))
        e = st.selectbox("HP_0060 Glu475Thr", ("presence", "absence"))
    with col8:
        f = st.selectbox("tdhF Ala217Thr", ("presence", "absence"))
        g = st.selectbox("group_1110", ("presence", "absence"))
        h = st.selectbox("group_641", ("presence", "absence"))
        i = st.selectbox("group_435", ("presence", "absence"))
        j = st.selectbox("nolK Ser97Leu", ("presence", "absence"))
    col4, col5, col6 = st.columns([2, 2, 6])
    with col4:
        submitted = st.form_submit_button("Predict")
    with col5:
        reset = st.form_submit_button("Reset")


    # 如果按下按钮
    if submitted:  # 显示按钮
        # 将输入存储DataFrame
        X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j]],
                         columns=['HP_0040 Val241Ile', 'HP_0642 Ala70fs', 'HP_0887 Thr439Ser', 'group_913', 'HP_0060 Glu475Thr', 'tdhF Ala217Thr', 'group_1110', 'group_641', 'group_435', 'nolK Ser97Leu'])

        X = X.replace(["presence", "absence"], [1, 0])
    
    
        # 进行预测
        prediction = model.predict(X)[0]
        Predict_proba = model.predict_proba(X)[:, 1][0]
        # 输出预测结果
        if prediction == 0:
            st.subheader(f"The predicted result of Hp MTZ:  Sensitive")
        else:
            st.subheader(f"The predicted result of Hp MTZ:  Resistant")
        # 输出概率
        st.subheader(f"The probability of Hp MTZ:  {'%.2f' % float(Predict_proba * 100) + '%'}")

        with st.spinner('force plot generation, please wait...'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0].values, feature_names=['HP_0040 Val241Ile', 'HP_0642 Ala70fs', 'HP_0887 Thr439Ser', 'group_913', 'HP_0060 Glu475Thr',
                                                                                                       'tdhF Ala217Thr', 'group_1110', 'group_641', 'group_435', 'nolK Ser97Leu'], matplotlib=True, show=False, figsize=(20, 5))
            plt.xticks(fontproperties='Arial', size=16)
            plt.yticks(fontproperties='Arial', size=16)
            plt.tight_layout()
            plt.savefig('force.png', dpi=600)
            st.image('force.png')
