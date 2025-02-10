# House Price Prediction

## 📌 Project Overview

House Price Prediction adalah proyek machine learning yang bertujuan untuk memprediksi harga rumah berdasarkan berbagai fitur seperti jumlah kamar tidur, luas bangunan, lokasi, dan kondisi rumah. Dataset yang digunakan berasal dari **House Sales in King County, USA**, yang berisi informasi harga rumah di daerah tersebut.

Dataset Link : https\://www\.kaggle.com/datasets/harlfoxem/housesalesprediction

---

## 📂 Dataset

Dataset yang digunakan memiliki **21 fitur** dan **21.613 sampel**. Berikut adalah beberapa kolom penting dalam dataset:

- `price` - Harga rumah (target)
- `bedrooms` - Jumlah kamar tidur
- `bathrooms` - Jumlah kamar mandi
- `sqft_living` - Luas bangunan (sqft)
- `floors` - Jumlah lantai
- `waterfront` - Apakah rumah memiliki pemandangan air (1 = ya, 0 = tidak)
- `condition` - Kondisi rumah (1-5)
- `grade` - Kualitas konstruksi dan desain rumah (1-13)
- `zipcode` - Kode pos lokasi rumah
- `lat, long` - Koordinat geografis rumah

---

## ⚙️ Data Preprocessing

1. **Import Library**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   ```
2. **Load Dataset**
   ```python
   df = pd.read_csv('Datasets/kc_house_data.csv')
   ```
3. **Memeriksa Data**
   ```python
   df.info()
   df.isnull().sum()  # Mengecek missing values
   ```
4. **Menghapus Fitur yang Tidak Diperlukan**
   ```python
   X = df.drop(columns=['id', 'date', 'price'], axis=1)
   y = df['price']
   ```
5. **Split Dataset**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   ```

---

## 🔍 Model yang Digunakan

Proyek ini menggunakan beberapa model regresi untuk membandingkan performa:

### 1️⃣ **Multiple Linear Regression**

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_pred = lin_reg.predict(X_test)
```

**R² Score: 0.69**

### 2️⃣ **Polynomial Regression**

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
poly_reg_pred = poly_reg.predict(poly.transform(X_test))
```

**R² Score: 0.38** (Kurang optimal karena overfitting)

### 3️⃣ **Decision Tree Regression**

```python
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X_train, y_train)
dec_tree_reg_pred = dec_tree_reg.predict(X_test)
```

**R² Score: 0.78**

### 4️⃣ **Random Forest Regression**

```python
from sklearn.ensemble import RandomForestRegressor
rand_for_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rand_for_reg.fit(X_train, y_train)
rand_for_reg_pred = rand_for_reg.predict(X_test)
```

**R² Score: 0.88** (Model terbaik dalam proyek ini)

### 5️⃣ **Support Vector Regression (SVR)**

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
y_train_scaled = sc_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
svr_pred = sc_y.inverse_transform(svr.predict(sc_X.transform(X_test)).reshape(-1, 1))
```

**R² Score: 0.80**

---

## 📊 Hasil Perbandingan Model

Berikut adalah hasil **R² Score** dari masing-masing model:

| Model                     | R² Score |
| ------------------------- | -------- |
| Linear Regression         | 0.69     |
| Polynomial Regression     | 0.38     |
| Decision Tree Regression  | 0.78     |
| Random Forest Regression  | 0.88     |
| Support Vector Regression | 0.80     |

**Kesimpulan:** Model **Random Forest Regression** memiliki performa terbaik dengan **R² Score 0.88**.

---

## 📌 Visualisasi Hasil Model

```python
import matplotlib.pyplot as plt

model_scores = {
    'Linear Regression': lin_reg_score,
    'Polynomial Regression': poly_reg_score,
    'Decision Tree Regression': dec_tree_reg_score,
    'Random Forest Regression': rand_for_reg_score,
    'Support Vector Regression': svr_score
}

plt.figure(figsize=(8, 5))
bars = plt.bar(model_scores.keys(), model_scores.values(), color=['blue', 'green', 'red', 'orange', 'purple'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', fontsize=10)

plt.xlabel('Model')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()
```

---

## 🔥 Kesimpulan

1. **Random Forest Regression** adalah model terbaik untuk prediksi harga rumah dengan **R² Score 0.88**.
2. **Polynomial Regression** kurang efektif karena menyebabkan overfitting.
3. **Feature Scaling** meningkatkan performa SVR, tetapi masih kalah dengan Random Forest.
4. **Keakuratan model dapat ditingkatkan** dengan **Feature Engineering** dan **Hyperparameter Tuning**.
