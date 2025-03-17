import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def show_churn(df):
    """
    이탈여부 / 비이탈여부 파이차트 그리는 함수

    parameter 설명 
    df는 dataframe 입니당 
    """
    fig, ax = plt.subplots(figsize=(7,7))
    count = Counter(df['이탈여부'])
    labels = ["잔류 고객", "이탈 고객"]
    ax.pie(count.values(), labels = labels, autopct = lambda p: f'{p:.2f}%', colors=["skyblue", "salmon"])
    ax.set_title('고객 이탈 비율')
    plt.show()

# 기존 goo 함수  
def corr_analysis(df, categorical_columns, filter_threshold=True):
    """
    이탈여부와 컬럼들의 상관관계를 분석하는 함수 (주피터 환경에서 실행 가능)
    
    Parameters:
        df (DataFrame) : 분석할 데이터프레임
        categorical_columns (list) : 범주형 변수 컬럼 리스트
        filter_threshold (bool) : True이면 상관계수 |0.3| 이상만 출력, False이면 전체 출력
    
    Returns:
        DataFrame : 상관계수 정렬된 결과
    """

    # 타겟 변수
    target_column = "이탈여부"
    
    # 제외할 컬럼
    exclude_columns = ["기준년월", "발급회원번호"]
    df = df.drop(columns=[col for col in exclude_columns if col in df.columns], errors="ignore")
    
    # 범주형 변수를 문자열로 변환
    df[categorical_columns] = df[categorical_columns].astype(str)
    
    # 원-핫 인코딩 적용 (범주형 변수)
    df_encoded = pd.get_dummies(df[categorical_columns], drop_first=True)
    
    # 전체 데이터셋에서 원본 숫자형 변수들과 결합
    df_final = pd.concat([df.drop(columns=categorical_columns), df_encoded], axis=1)
    
    # 결측값(NaN)을 0으로 채움
    df_final = df_final.fillna(0)
    
    # 표준편차가 0인 열(값이 모두 동일한 열) 제거
    df_final = df_final.loc[:, df_final.std() != 0]
    
    # 이탈 여부와의 상관관계 계산
    correlations = df_final.corrwith(df[target_column])
    
    # 결과를 데이터프레임으로 변환 및 내림차순 정렬
    correlations_df = correlations.reset_index()
    correlations_df.columns = ["변수", "상관계수"]
    correlations_df = correlations_df.sort_values(by="상관계수", ascending=False)
    
    # **사용자가 필터링 여부를 선택할 수 있도록 조건 추가**
    if filter_threshold:
        correlations_df = correlations_df[
            (correlations_df["상관계수"] >= 0.3) | (correlations_df["상관계수"] <= -0.3)
        ]
    
    # Jupyter에서 모든 행을 출력 (생략 없이)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    
    # 주피터에서 결과 출력
    display(correlations_df)

    return correlations_df  # 데이터프레임 반환

