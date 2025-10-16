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
    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df = pd.read_csv(file_path) # 使用傳入的 file_path
    df.columns = [c.capitalize() for c in df.columns]
    # 'Cabin' 缺失值較多，在此先移除
    df = df.drop(['Cabin', 'Ticket', 'Name', 'Passengerid'], axis=1) # 移除不相關或太多缺失值的欄位
    missing_count = df.isnull().values.sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    # TODO 2.2: 以 Embarked 眾數填補
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    Fare_mean = df['Fare'].mean()
    Fare_std = df['Fare'].std()
    outlier_threshold = Fare_mean + 3 * Fare_std
    df = df[df['Fare'] <= outlier_threshold]
    return df


# 任務 4：類別變數編碼
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    # 這裡選擇移除原始欄位，並將 Pclass 視為類別變數進行編碼
    df['Pclass'] = df['Pclass'].astype('category')
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], prefix=['Sex', 'Embarked', 'Pclass'], drop_first=True) # drop_first=True 避免共線性
    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    # 標準化後，將結果放回 DataFrame
    scaler = StandardScaler()
    features_to_scale = ['Age', 'Fare']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df # 返回 DataFrame


# 任務 6：資料切割
def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    # 確保 Survived 欄位存在
    if 'Survived' not in df.columns:
        # 這應該不會發生，除非前一個步驟有誤
        raise ValueError("DataFrame 缺少 'Survived' 欄位")

    X = df.drop('Survived', axis=1)
    Y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將處理後的 DataFrame 存成 CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


# 主程式流程（請勿修改）
if __name__ == "__main__":
    input_path = "data/titanic.csv"
    output_path = "data/titanic_processed.csv"

    # 注意：由於您在任務 1 中硬編碼了 'data/titanic.csv'，如果實際路徑不同，請修改 load_data
    df, missing_count = load_data(input_path) 
    print(f"原始缺失值數量 (Age, Embarked, Fare, Cabin...): {missing_count}")

    # 必須先處理缺失值，否則 remove_outliers 和 scale_features 可能出錯
    df = handle_missing(df)
    
    # 在編碼和標準化前移除 Fare 異常值
    df = remove_outliers(df)
    
    # 在標準化前進行編碼，以便標準化只作用於數值特徵
    df = encode_features(df)
    
    # 執行數值標準化 (返回 DataFrame)
    df = scale_features(df)
    
    # 執行資料切割 (需傳入 DataFrame)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 輸出最終處理後的 DataFrame (需要保留所有特徵)
    save_data(df, output_path)

    print("Titanic 資料前處理完成")
    print(f"最終處理後的資料集形狀: {df.shape}")
    print(f"訓練集形狀: {X_train.shape}")
    print(f"測試集形狀: {X_test.shape}")