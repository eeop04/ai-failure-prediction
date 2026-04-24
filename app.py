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
    # failure 자동 생성 (먼저 이동)
    # ------------------------
    if "failure" not in df.columns:
        st.warning("failure 컬럼이 없어 자동 생성합니다.")
        df["failure"] = np.random.randint(0, 2, size=len(df))

    # ------------------------
    # EDA (4주차)
    # ------------------------
    st.subheader("데이터 시각화")

    # 숫자 컬럼만 가져오기
    num_cols = df.select_dtypes(include=["number"]).columns

    valid_cols = []

    for c in num_cols:
        unique_ratio = df[c].nunique() / len(df)

        # ID 같은 컬럼 제거
        if unique_ratio < 0.9:
            valid_cols.append(c)

    if len(valid_cols) == 0:
        st.warning("시각화할 수 있는 적절한 컬럼이 없습니다.")
    else:
        col = st.selectbox("그래프 선택 컬럼", valid_cols)

        chart_type = st.radio("그래프 종류 선택", ["히스토그램", "박스플롯", "라인그래프"])

        # ------------------------
        # 그래프 생성 (UI 개선 핵심)
        # ------------------------
        if chart_type == "히스토그램":
            fig = px.histogram(
                df,
                x=col,
                nbins=20,
                color="failure",
                barmode="group",   # 패딩 느낌
                opacity=0.75,
                color_discrete_map={
                    0: "blue",
                    1: "red"
                }
            )

        elif chart_type == "박스플롯":
            fig = px.box(
                df,
                x="failure",
                y=col,
                color="failure",
                color_discrete_map={
                    0: "blue",
                    1: "red"
                }
            )

        elif chart_type == "라인그래프":
            fig = px.line(
                df,
                y=col,
                color="failure"
            )

        # ------------------------
        # 레이아웃 개선
        # ------------------------
        fig.update_layout(
            title=f"{col} vs Failure 분석",
            xaxis_title=col,
            yaxis_title="값",
            legend_title="Failure 여부",
            bargap=0.2   # 막대 간격
        )

        st.plotly_chart(fig, use_container_width=True)

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