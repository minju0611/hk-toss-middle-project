import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve

from bayes_opt import BayesianOptimization
import lightgbm as lgb
import numpy as np
import seaborn as sns

"""
1. set_macos_font():
    - mac 폰트 깨짐 방지
2. set_window_font():
    - widnow 폰트 깨짐 방지
3. create_df(df_target, target_col, df_member, df_credit, df_sales, df_bill)
    - 타겟과 부른 데이터프레임들 합치기 
4. balanced_sampling(df, target_col, sample_size_1, sample_size_0, random_state=42)
    - 균등 추출
5. show_target_piechart(df, target_col)
    - 타겟의 파이차트 그리기

"""

# mac 체제 폰트 깨짐 방지
def set_macos_font():
    """
    MacOS 환경에서 matplotlib의 한글 폰트 깨짐을 방지하는 설정 적용 함수
    
    Returns:
       - None (설정이 적용됨)
    """
    plt.rcParams['font.family'] = 'AppleGothic'  # Mac 기본 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False   # 마이너스 부호 깨짐 방지

# window 체제 폰트 깨짐 방지
def set_window_font():
    plt.rc('font', family='Malgun Gothic')

def balanced_sampling(df, target_col, sample_size_1, sample_size_0, random_state=42):
    """
    특정 타겟 컬럼의 각 클래스에서 지정한 개수만큼 샘플링한 후, 결합하는 함수.

    Parameters:
        df (pd.DataFrame): 균등 추출에 사용할 데이터프레임
        target_col (str): 균등 추출 기준이 될 컬럼명 (예: '이탈여부')
        sample_size_1 (int): target_col1 인 데이터에서 추출할 샘플 개수
        sample_size_0 (int): target_col1 0인 데이터에서 추출할 샘플 개수
        random_state (int): 랜덤 시드 (재현성 보장을 위함)

    Returns:
        pd.DataFrame: 균등 샘플링된 데이터프레임
    """
    # 이탈여부 1과 0 각각 샘플링
    df_1_sampled = df[df[target_col] == 1].sample(n=sample_size_1, random_state=random_state)
    df_0_sampled = df[df[target_col] == 0].sample(n=sample_size_0, random_state=random_state)
    
    # 두 개 합치고 섞기
    df_balanced = pd.concat([df_1_sampled, df_0_sampled]).sample(frac=1, random_state=random_state)

    # 이탈여부 분포 출력
    print(df_balanced[target_col].value_counts())

    return df_balanced


def show_target_piechart(df, target_col):
    """
    특정 타겟 컬럼의 분포를 원형 그래프로 시각화하는 함수.

    Parameters:
        df (pd.DataFrame): 분석할 데이터프레임
        target_col (str): 분포를 확인할 타겟 컬럼명 (예: '이탈여부')

    Returns:
        None (파이차트 그래프)
    """
    # 타겟 컬럼 값 개수 계산
    count = Counter(df[target_col])
    
    # 라벨 설정
    labels = [f"{target_col} {key}" for key in count.keys()]
    
    # 원형 그래프 그리기
    fig, ax = plt.subplots(figsize=(7,7))
    ax.pie(count.values(), labels=labels, autopct=lambda p: f'{p:.2f}%', colors=["skyblue", "salmon"])
    ax.set_title(f"'{target_col}' 분포")
    plt.show()

def create_df(df_target, target_col, df_member, df_credit, df_sales, df_bill):
    """
    타겟을 정할 컬럼을 지정하여 새 데이터프레임에 저장
    '발급회원번호' 기준으로 병합
    타겟 저장된 데이터프레임과 

    Parameters:
    
        1. df_target: 이탈여부 컬럼 추가를 위한 타겟 컬럼, 발급회원번호를 읽은 데이터프레임
        2. target_col: 이탈여부를 정할 컬럼 
        3. df_member: 1.회원정보 데이터프레임
        4. df_credit: 2.신용정보 데이터프레임
        5. df_sales: 3.카드승인매출 데이터프레임 
        6. df_bill: 4.카드청구정보 데이터프레임

        # - 3.카드승인매출 csv 파일을 읽을 때 '기준년월' 컬럼 제외해야 함
         
    Returns:
        기본 전처리된 데이터프레임

    """

    # target_col이 존재하는지 확인 
    if target_col in df_target.columns:  
        # 타겟으로 잡은 컬럼을 0,1로 분리
        df_target[target_col] = (df_target[target_col] == 0).astype(int)
    else:
        raise KeyError("df_target에 '이용건수_신용_R6M' 컬럼이 없는디?")

    # 필수 컬럼 존재 여부 체크
    required_cols = ['발급회원번호']
    for df_name, df in zip(['df_member', 'df_credit', 'df_sales', 'df_bill'], [df_member, df_credit, df_sales, df_bill]):
        # 01 ~ 04 데이터 불러온 거 
        if not all(col in df.columns for col in required_cols):
            raise KeyError(f"{df_name}에 '발급회원번호' 컬럼이 없는디?")

    # Merge dataframe
    df_merged = df_member.copy()
    for df in [df_credit, df_sales, df_bill]:
        df_merged = pd.merge(df_merged, df, on='발급회원번호')

    # Merge target dataframe
    df_merged = pd.merge(df_merged, df_target[['발급회원번호', target_col]], on='발급회원번호')

    # 데이터프레임 합친 후, Drop columns (존재하지 않는 컬럼 있어도 에러 방지)
    df_merged = df_merged.drop(columns=['기준년월', '발급회원번호', '연령', 'VIP등급코드'], errors='ignore')

    return df_merged

def ml_prepare_data(df, target_col, test_size=0.2, random_state=42):
    """
    데이터셋을 학습용과 테스트용으로 분리하는 함수

    Parameters:
        - target_col: 타겟 변수(이탈여부 등)의 컬럼명
        - test_size: 테스트 데이터 비율 (기본값: 0.2)
        - random_state: 랜덤 시드 고정
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])  # 타겟 컬럼 제거
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def train_lgbm(X_train, y_train, params=None):
    """
    LightGBM 모델을 학습하는 함수

    Parameters:
        - X_train, y_train: 학습 데이터
        - params: 사용자 지정 하이퍼파라미터 딕셔너리 (기본값 None -> Default 설정 사용)
    
    Returns:
        return: 학습된 모델
    """
    # 기본 하이퍼파라미터 설정
    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_depth": -1,
        "min_child_samples": 10,
        "min_child_weight": 1e-3,
        "scale_pos_weight": 1.0,
    }
    
    # 사용자 정의 하이퍼파라미터 적용 (기본값 업데이트)
    if params:
        default_params.update(params)

    # 모델 생성 및 학습
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    학습된 모델 평가지표

    Parameters:
        - model: 학습된 LightGBM 모델
        - X_test, y_test: 테스트 데이터
    
    return: 
        - 평가 지표 (Accuracy, AUC-ROC, Precision, Recall)
    """
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # 결과 출력
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return accuracy, roc_auc, precision, recall


def bayesian_optimization(X_train, y_train, init_points=5, n_iter=15, random_state=42):
    """
    LightGBM 모델의 하이퍼파라미터를 베이지안 최적화를 통해 최적화하는 함수.

    Parameters:
        - X_train: 훈련 데이터 (Feature)
        - y_train: 훈련 데이터 (Target)
        - init_points: 초기 탐색 포인트 개수
        - n_iter: 최적화 반복 횟수
        - random_state: 랜덤 시드 값

    Returns:
        - 최적의 하이퍼파라미터 딕셔너리 (best_params)
    """

    # 목표 함수 정의
    def lgbm_eval(learning_rate, max_depth, num_leaves, scale_pos_weight, min_child_samples, subsample):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': learning_rate,
            'max_depth': int(max_depth),
            'num_leaves': int(num_leaves),
            'scale_pos_weight': scale_pos_weight,
            'min_child_samples': int(min_child_samples),
            'subsample': subsample,
            'n_estimators': 500
        }

        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
        return np.mean(scores)  # AUC 값의 평균 반환

    # 최적화할 하이퍼파라미터의 범위 설정
    param_bounds = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'num_leaves': (10, 50),
        'scale_pos_weight': (1.0, 5.0),
        'min_child_samples': (10, 100),
        'subsample': (0.5, 1.0)
    }

    # 베이지안 최적화 실행
    optimizer = BayesianOptimization(f=lgbm_eval, pbounds=param_bounds, random_state=random_state)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # 최적 하이퍼파라미터 출력 및 반환
    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])  # 정수형 변환
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    
    print("\n 베스트 파라미터 출력 ~~:", best_params)
    return best_params

# 최적 파라미터 lgbm에 학습시키는 함수 
def train_best_lgbm(X_train, y_train, best_params):
    """
    베이지안 최적화로 찾은 최적 하이퍼파라미터를 적용해 LGBM 모델을 학습하는 함수.

    Parameters:
        - X_train: 훈련 데이터 (Feature)
        - y_train: 훈련 데이터 (Target)
        - best_params: 최적의 하이퍼파라미터 딕셔너리

    Returns:
        - 학습된 LGBM 모델
    """

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    return model

############# 예측 확률값 확인하는 그래프
def plot_prediction_proba(model, X_test):
    """
    모델이 예측한 확률값의 분포를 시각화하는 함수

    Parameters:
    - model: 학습된 LightGBM 모델
    - X_test: 테스트 데이터 (특징 변수)
    """
    proba = model.predict_proba(X_test)[:, 1]  # 클래스 1의 확률값 가져오기

    plt.figure(figsize=(8, 5))
    sns.histplot(proba, bins=50, kde=True)  # KDE 그래프 포함
    plt.title("예측 확률값 분포", fontsize=14)
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True)
    plt.show()


#################### 여러 임계값 적용해서 재현/정밀 변화를 보는 시각화 그래프
def plot_precision_recall_vs_threshold(y_test, y_pred_proba):
    """
    여러 임계값을 적용하면서 Precision과 Recall이 어떻게 변하는지 시각화하는 함수.

    Parameters:
    - y_test: 실제 라벨값
    - y_pred_proba: 모델이 예측한 확률값 (클래스 1일 확률)

    Returns:
    - 그래프 출력 (X축: 임계값, Y축: Precision & Recall)
    """

    thresholds = np.arange(0.1, 0.9, 0.05)  # 0.1 ~ 0.9까지 0.05 간격으로 탐색
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred_adjusted))
        recalls.append(recall_score(y_test, y_pred_adjusted))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision", marker="o")
    plt.plot(thresholds, recalls, label="Recall", marker="s")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs. Threshold")
    plt.legend()
    plt.grid()
    plt.show()

def find_best_threshold(model, X_test, y_test):
    """
    Precision-Recall Curve를 활용하여 최적의 임계값(Threshold)을 찾고, 
    해당 임계값을 적용한 Precision과 Recall을 출력하는 함수.

    Parameters:
        model: 학습된 LightGBM 모델
        X_test: 테스트 데이터 (Feature)
        y_test: 테스트 데이터 (정답 Label)

    Returns:
        best_threshold: 최적의 Threshold 값
        new_precision: 최적 Threshold 적용 후 Precision
        new_recall: 최적 Threshold 적용 후 Recall
    """
    
    # 모델의 예측 확률값 가져오기
    y_proba = model.predict_proba(X_test)[:, 1]

    # Precision-Recall Curve 계산
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # F1-score 계산 및 최적 Threshold 선택
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1_scores.argmax()]  # F1-score가 가장 높은 Threshold 선택

    print(f"Best Threshold: {best_threshold:.4f}")

    # 최적 Threshold 기준으로 예측값 변환
    y_pred_adj = (y_proba >= best_threshold).astype(int)

    # Precision & Recall 출력
    new_precision = precision_score(y_test, y_pred_adj)
    new_recall = recall_score(y_test, y_pred_adj)

    print(f"New Precision: {new_precision:.4f}")
    print(f"New Recall: {new_recall:.4f}")

    return best_threshold, new_precision, new_recall