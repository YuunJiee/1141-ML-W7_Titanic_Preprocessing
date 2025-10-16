import pandas as pd
import numpy as np

# TODO 1.1: 讀取 CSV
# TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量

df = pd.read_csv("data/titanic.csv")
df.columns = [c.capitalize() for c in df.columns]
missing_count = df.isnull().values.sum()

print('缺失值總數:', missing_count)
df.head()

# TODO 2.1: 以 Age 中位數填補
# TODO 2.2: 以 Embarked 眾數填補

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum())

# TODO 3.1: 計算 Fare 平均與標準差
# TODO 3.2: 移除 Fare > mean + 3*std

Fare_mean= df['Fare'].mean()
Fare_std = df['Fare'].std()
outlier_threshold = Fare_mean + 3 * Fare_std
df = df[df['Fare'] <= outlier_threshold]

print('筆數:', len(df))
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3))
plt.boxplot(df['Fare'])
plt.title('Fare Boxplot (After Outlier Removal)')
plt.show()
# TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼

df = pd.get_dummies(df, columns=['Sex','Embarked'],prefix=['Sex','Embarked'],drop_first=False)
df.head()

from sklearn.preprocessing import StandardScaler

# TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

df[['Age', 'Fare']].describe()

from sklearn.model_selection import train_test_split

# TODO 6.1: 將 Survived 作為 y，其餘為 X
# TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
X = df.drop('Survived', axis = 1)
Y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print('訓練集筆數:', len(X_train))
print('測試集筆數:', len(X_test))

# TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
df.to_csv('data/titanic_processed.csv', index=False, encoding='utf-8-sig')

print('✅ 資料處理完成並已輸出至 data/titanic_processed.csv')