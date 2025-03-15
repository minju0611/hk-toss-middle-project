import pandas as pd

def preprocessing_customer_data(df):
    """
    이탈 고객 라벨링 + 카드 건수 0인행 제거하는 공통 전처리 함수 
    """
    df = df[df["PYE_C1M210000"] > 0]
    df["이탈여부"] = (df["PYE_C18233003"] - df["PYE_C18233003"] > 0).astype(int)
    return df


