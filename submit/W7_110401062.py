# -*- coding: utf-8 -*-
# W6 Titanic Preprocessing Template
# 僅可修改 TODO 區塊，其餘部分請勿更動

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 任務 1：載入資料
def load_data(file_path):
    # TODO 1.1: 讀取 CSV
    df = pd.read_csv(file_path)
    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isna().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    # TODO 2.2: 以 Embarked 眾數填補
    if 'Embarked' in df.columns:
        mode_val = df['Embarked'].mode().iloc[0]
        df['Embarked'] = df['Embarked'].fillna(mode_val)
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    if 'Fare' in df.columns:
        # 確保 Fare 為數值
        fare = pd.to_numeric(df['Fare'], errors='coerce')
        fare_mean = fare.mean()
        fare_std = fare.std()  # pandas 樣本標準差 (ddof=1)
        threshold = fare_mean + 3 * fare_std
        # TODO 3.2: 移除 Fare > mean + 3*std 的資料
        df = df[fare <= threshold].copy()
        df.reset_index(drop=True, inplace=True)
    return df



# 任務 4：類別變數編碼
def remove_outliers(df):
    # 重複篩選，直到沒有新異常值被移除
    if 'Fare' in df.columns:
        df = df.copy()
        df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

        prev_len = -1
        # 當資料筆數持續變化（表示仍有異常值被移除）時，繼續迴圈
        while prev_len != len(df):
            prev_len = len(df)
            fare_mean = df['Fare'].mean()
            fare_std = df['Fare'].std()  # 樣本標準差 ddof=1
            threshold = fare_mean + 3 * fare_std
            df = df[df['Fare'] <= threshold].copy()
            df.reset_index(drop=True, inplace=True)
    return df



# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    df_scaled = df.copy()
    cols = [c for c in ['Age', 'Fare'] if c in df_scaled.columns]
    if cols:
        df_scaled[cols] = scaler.fit_transform(df_scaled[cols])
    return df_scaled

# 任務 6：資料切割
def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    if 'Survived' not in df.columns:
        raise ValueError("缺少 'Survived' 欄位")
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

# 主程式流程（請勿修改）
if __name__ == "__main__":
    input_path = "data/titanic.csv"
    output_path = "data/titanic_processed.csv"

    df, missing_count = load_data(input_path)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(df, output_path)

    print("Titanic 資料前處理完成")
