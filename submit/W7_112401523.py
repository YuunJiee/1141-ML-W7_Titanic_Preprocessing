# -*- coding: utf-8 -*-
# Titanic 資料前處理簡化版

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 任務 1：載入資料
def load_data(file_path):
    # 1.1 讀取 CSV
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    # 1.2 統一欄位首字母大寫
    df.columns = [c.capitalize() for c in df.columns]
    # 計算缺失值總數
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # 2.1 用 Age 的中位數填補
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # 2.2 用 Embarked 的眾數填補
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # 3.1 計算 Fare 平均與標準差
    fare_mean = df['Fare'].mean()
    fare_std = df['Fare'].std()
    # 3.2 移除 Fare > 平均 + 3*標準差 的資料
    df = df[df['Fare'] <= fare_mean + 3 * fare_std]
    return df


# 任務 4：類別變數編碼
def encode_features(df):
    # 使用 One-hot 對 Sex、Embarked 編碼
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df


# 任務 5：數值標準化
def scale_features(df):
    scaler = StandardScaler()
    # 建立一個副本
    df_scaled = df.copy()
    # 標準化 Age 和 Fare 欄位
    df_scaled[['Age', 'Fare']] = scaler.fit_transform(df_scaled[['Age', 'Fare']])
    return df_scaled


# 任務 6：資料切割
def split_data(df):
    # 6.1 y 是 Survived，X 是其餘欄位
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    # 6.2 切割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


# 主程式流程
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

    print("Titanic 資料前處理完成 ✅")

