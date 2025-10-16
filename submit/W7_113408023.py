# -*- coding: utf-8 -*-
# W6 Titanic Preprocessing Template
# 僅可修改 TODO 區塊，其餘部分請勿更動

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # [風格修正] 移至檔案開頭


# 任務 1：載入資料
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median) # 建議寫法，避免 FutureWarning
    
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode) # 建議寫法
    
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # [觀念補充] 雖然這次作業不能改函式參數，但未來可以設計成 remove_outliers(df, column, n_std)
    # 這樣就可以對任何欄位移除 n 倍標準差外的異常值，增加程式碼的重複使用性。
    fare_mean = df['Fare'].mean()
    fare_std = df['Fare'].std()
    
    outlier_threshold = fare_mean + 3 * fare_std
    df = df[df['Fare'] <= outlier_threshold]
    
    return df


# 任務 4：類別變數編碼
def encode_features(df):
    # [主要修正] 移除 drop_first=True，以符合自動測試的要求，保留所有 one-hot 編碼後的欄位
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    
    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    numerical_cols = ['Age', 'Fare']
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_scaled


# 任務 6：資料切割
def split_data(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # [品質提升] 在儲存前增加資料驗證，確保沒有缺失值
    assert df.isnull().sum().sum() == 0, "處理後的資料仍有缺失值！"
    
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
    print(f"原始資料缺失值總數: {missing_count}")
    print("處理後資料前五筆:")
    print(df.head())
    print(f"\n訓練集資料維度 (X_train): {X_train.shape}")
    print(f"測試集資料維度 (X_test): {X_test.shape}")
