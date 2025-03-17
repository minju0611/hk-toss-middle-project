import pandas as pd

def preprocessing_customer_data(df_21, df_22):
    """
    이탈 고객 라벨링 + 카드 건수 0인행 제거하는 공통 전처리 함수 
    """
    # 카드 건수(PYE_C1M210000) 0인 행 제거 (df_22 기준)
    df_22 = df_22[df_22["PYE_C1M210000"] > 0]

    # 원본 데이터 변경 방지 (SettingWithCopyWarning)
    df_22 = df_22.copy()

    # 이탈 여부 라벨링
    df_22["exited"] = (df_21["PYE_C18233005"] - df_22["PYE_C18233005"] > 0).astype(int)

    return df_22



