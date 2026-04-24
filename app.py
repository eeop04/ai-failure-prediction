import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import numpy as np

st.title("설비 고장 예측 시스템")

file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("데이터 미리보기")
    st.dataframe(df)

    # ------------------------
    # EDA (4주차)
    # ------------------------
    st.subheader("데이터 시각화")
    col = st.selectbox("그래프 선택 컬럼", df.columns)
    fig = px.histogram(df, x=col, nbins=20)
    st.plotly_chart(fig)

    # ------------------------
    # failure 자동 생성 (핵심)
    # ------------------------
    if "failure" not in df.columns:
        st.warning("failure 컬럼이 없어 자동 생성합니다.")

        # 랜덤으로 0,1 생성 (임시 라벨)
        df["failure"] = np.random.randint(0, 2, size=len(df))

    # ------------------------
    # 모델 (5주차)
    # ------------------------
    st.subheader("고장 예측")

    X = df.drop("failure", axis=1)
    y = df["failure"]

    # 숫자 데이터만 사용
    X = X.select_dtypes(include=["number"])

    model = RandomForestClassifier()
    model.fit(X, y)

    if st.button("예측 실행"):
        pred = model.predict(X)
        df["예측결과"] = pred

        st.write(df)
        st.success("예측 완료!")