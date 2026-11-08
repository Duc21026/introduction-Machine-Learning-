# Quy trình học Code và Dữ liệu Machine Learning (Chi tiết, đầy đủ các dạng)

> Tài liệu dành cho sinh viên Khoa học Dữ liệu. Đi từ nền tảng → dữ liệu → mô hình → đánh giá → triển khai, kèm code Python mẫu cho từng giai đoạn.

---

## Mục lục

1. [Bức tranh tổng thể](#1-bức-tranh-tổng-thể)
2. [Chuẩn bị môi trường và nền tảng](#2-chuẩn-bị-môi-trường-và-nền-tảng)
3. [Các dạng dữ liệu trong ML](#3-các-dạng-dữ-liệu-trong-ml)
4. [Quy trình ML end-to-end (10 bước)](#4-quy-trình-ml-end-to-end-10-bước)
5. [Bước 1-2: Xác định bài toán và thu thập dữ liệu](#5-bước-1-2-xác-định-bài-toán-và-thu-thập-dữ-liệu)
6. [Bước 3: Khám phá dữ liệu (EDA)](#6-bước-3-khám-phá-dữ-liệu-eda)
7. [Bước 4: Làm sạch và tiền xử lý](#7-bước-4-làm-sạch-và-tiền-xử-lý)
8. [Bước 5: Feature Engineering và Feature Selection](#8-bước-5-feature-engineering-và-feature-selection)
9. [Bước 6: Chia dữ liệu và Cross-validation](#9-bước-6-chia-dữ-liệu-và-cross-validation)
10. [Bước 7: Chọn và huấn luyện mô hình](#10-bước-7-chọn-và-huấn-luyện-mô-hình)
11. [Bước 8: Đánh giá mô hình (Metrics)](#11-bước-8-đánh-giá-mô-hình-metrics)
12. [Bước 9: Tinh chỉnh và chống Overfitting](#12-bước-9-tinh-chỉnh-và-chống-overfitting)
13. [Bước 10: Giải thích, lưu và triển khai](#13-bước-10-giải-thích-lưu-và-triển-khai)
14. [Quy trình riêng theo từng dạng dữ liệu](#14-quy-trình-riêng-theo-từng-dạng-dữ-liệu)
15. [Lộ trình học theo tuần](#15-lộ-trình-học-theo-tuần)
16. [Dự án thực hành gợi ý](#16-dự-án-thực-hành-gợi-ý)
17. [Checklist và lỗi thường gặp](#17-checklist-và-lỗi-thường-gặp)
18. [Tài nguyên học tập](#18-tài-nguyên-học-tập)

---

## 1. Bức tranh tổng thể

```
Bài toán thực tế
      │
      ▼
Thu thập dữ liệu ──► EDA ──► Làm sạch ──► Feature Engineering
                                                │
                                                ▼
        Triển khai ◄── Tinh chỉnh ◄── Đánh giá ◄── Huấn luyện ◄── Chia train/val/test
            │
            ▼
   Giám sát (monitoring) ──► Thu thập dữ liệu mới ──► Huấn luyện lại
```

**Ba nguyên tắc cốt lõi:**

1. **Dữ liệu quan trọng hơn thuật toán.** Dữ liệu sạch + feature tốt thường thắng mô hình phức tạp.
2. **Luôn đo trên dữ liệu chưa thấy.** Điểm trên tập train không có ý nghĩa đánh giá.
3. **Bắt đầu đơn giản.** Có baseline (mô hình đơn giản) trước rồi mới nâng cấp.

### Phân loại ML

| Loại | Dữ liệu | Mục tiêu | Ví dụ thuật toán |
|---|---|---|---|
| **Học có giám sát** | Có label | Dự đoán label | Hồi quy, Cây, SVM, Nơ-ron |
| **Học không giám sát** | Không label | Tìm cấu trúc ẩn | K-Means, PCA, DBSCAN |
| **Bán giám sát** | Ít label + nhiều không label | Tận dụng cả hai | Self-training, Label Propagation |
| **Tự giám sát** | Không label, tự tạo label | Học biểu diễn | BERT, SimCLR |
| **Học tăng cường** | Môi trường + phần thưởng | Tối đa hóa phần thưởng | Q-Learning, PPO |

---

## 2. Chuẩn bị môi trường và nền tảng

### 2.1. Cài đặt môi trường

```bash
# Tạo môi trường ảo
python -m venv ml_env
source ml_env/bin/activate        # Windows: ml_env\Scripts\activate

# Thư viện cốt lõi
pip install numpy pandas scipy matplotlib seaborn plotly
pip install scikit-learn statsmodels
pip install xgboost lightgbm catboost
pip install jupyterlab ipykernel

# Tùy chọn theo nhu cầu
pip install torch torchvision torchaudio        # Deep learning (PyTorch)
pip install tensorflow                          # Hoặc TensorFlow/Keras
pip install imbalanced-learn optuna shap joblib
pip install nltk spacy transformers             # NLP
pip install opencv-python pillow                # Ảnh
pip install librosa                             # Âm thanh
pip install prophet                             # Chuỗi thời gian
```

**Công cụ khuyên dùng:** JupyterLab / VS Code, Git + GitHub, Google Colab (có GPU miễn phí), Kaggle Notebooks.

### 2.2. Kiến thức nền tảng cần có

| Mảng | Nội dung cần nắm | Dùng ở đâu |
|---|---|---|
| **Python** | List/dict, hàm, class, list comprehension, xử lý file | Mọi nơi |
| **NumPy** | Mảng, broadcasting, phép toán ma trận | Tính toán số |
| **Pandas** | DataFrame, lọc, groupby, merge, pivot | Xử lý dữ liệu bảng |
| **Xác suất** | Phân phối chuẩn, kỳ vọng, phương sai, Bayes | Naive Bayes, hiểu nhiễu |
| **Thống kê** | Trung bình, trung vị, tương quan, kiểm định giả thuyết, p-value | EDA, hồi quy |
| **Đại số tuyến tính** | Vector, ma trận, nhân ma trận, giá trị riêng | PCA, nơ-ron |
| **Giải tích** | Đạo hàm, gradient, quy tắc chuỗi | Gradient Descent, backprop |
| **SQL** | SELECT, JOIN, GROUP BY, window function | Lấy dữ liệu thực tế |

### 2.3. Ôn nhanh Pandas (dùng hàng ngày)

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

# Xem tổng quan
df.head()
df.info()
df.describe()
df.shape

# Chọn và lọc
df["cot_a"]
df[["cot_a", "cot_b"]]
df[df["tuoi"] > 30]
df.loc[df["quan"] == "Quan 7", ["gia", "dien_tich"]]
df.iloc[0:5, 0:3]

# Nhóm và tổng hợp
df.groupby("quan")["gia"].agg(["mean", "median", "count"])

# Nối bảng
pd.merge(df1, df2, on="id", how="left")
pd.concat([df1, df2], axis=0)

# Xoay bảng
pd.pivot_table(df, values="gia", index="quan", columns="loai_nha", aggfunc="mean")

# Hàm áp dụng
df["gia_m2"] = df["gia"] / df["dien_tich"]
df["nhom_tuoi"] = pd.cut(df["tuoi"], bins=[0, 18, 35, 60, 100], labels=["tre", "tre_truong_thanh", "trung_nien", "cao_tuoi"])
```

---

## 3. Các dạng dữ liệu trong ML

| Dạng dữ liệu | Mô tả | Ví dụ | Công cụ/Mô hình điển hình |
|---|---|---|---|
| **Bảng (tabular)** | Hàng = mẫu, cột = feature | Hồ sơ khách hàng, giá nhà | Pandas, Random Forest, XGBoost |
| **Chuỗi thời gian** | Dữ liệu theo thứ tự thời gian | Giá cổ phiếu, doanh thu ngày | ARIMA, Prophet, LSTM |
| **Văn bản (text)** | Chuỗi ký tự | Bình luận, email, tin tức | TF-IDF, Word2Vec, BERT |
| **Ảnh (image)** | Ma trận pixel | Ảnh X-quang, ảnh sản phẩm | CNN, ResNet, ViT |
| **Âm thanh (audio)** | Dạng sóng | Giọng nói, nhạc | Spectrogram, MFCC, CNN/RNN |
| **Video** | Chuỗi ảnh + âm thanh | Camera giám sát | 3D CNN, Video Transformer |
| **Đồ thị (graph)** | Nút + cạnh | Mạng xã hội, phân tử | GNN, Node2Vec |
| **Không gian/địa lý** | Tọa độ | Vị trí giao hàng | GeoPandas, clustering |
| **Đa phương thức** | Kết hợp nhiều dạng | Ảnh + mô tả sản phẩm | CLIP, mô hình đa phương thức |

### 3.1. Phân loại theo cấu trúc

- **Có cấu trúc:** bảng, CSV, cơ sở dữ liệu SQL
- **Bán cấu trúc:** JSON, XML, log
- **Phi cấu trúc:** văn bản tự do, ảnh, âm thanh, video

### 3.2. Phân loại theo kiểu giá trị của feature

| Kiểu | Con | Ví dụ | Cách mã hóa |
|---|---|---|---|
| Định lượng | Liên tục | Diện tích, tuổi | Chuẩn hóa |
| Định lượng | Rời rạc | Số phòng | Giữ nguyên |
| Định tính | Danh nghĩa | Quận, màu | One-Hot |
| Định tính | Thứ tự | Học vấn, mức hài lòng | Ordinal |
| Nhị phân | | Có/Không | 0/1 |
| Thời gian | | Ngày giờ | Tách thành thành phần |

---

## 4. Quy trình ML end-to-end (10 bước)

| # | Bước | Câu hỏi chính | Sản phẩm đầu ra |
|---|---|---|---|
| 1 | Xác định bài toán | Dự đoán gì? Đo thành công bằng gì? | Mô tả bài toán, metric mục tiêu |
| 2 | Thu thập dữ liệu | Dữ liệu ở đâu? Đủ chưa? | Dataset thô |
| 3 | EDA | Dữ liệu trông thế nào? | Biểu đồ, nhận xét, danh sách vấn đề |
| 4 | Làm sạch | Thiếu, trùng, ngoại lai, sai? | Dataset sạch |
| 5 | Feature engineering | Tạo/chọn feature nào? | Ma trận X, vector y |
| 6 | Chia dữ liệu | Chia sao để đánh giá công bằng? | Train / Validation / Test |
| 7 | Huấn luyện | Dùng thuật toán nào? | Baseline + các mô hình ứng viên |
| 8 | Đánh giá | Mô hình tốt đến đâu? Sai ở đâu? | Bảng metric, phân tích lỗi |
| 9 | Tinh chỉnh | Làm sao tốt hơn, không overfit? | Mô hình tối ưu |
| 10 | Triển khai | Đưa vào dùng thế nào? Giám sát ra sao? | API/ứng dụng, monitoring |

> **Lưu ý:** quy trình là **vòng lặp**, không phải đường thẳng. Thường quay lại bước 3-5 nhiều lần sau khi đánh giá.

---

## 5. Bước 1-2: Xác định bài toán và thu thập dữ liệu

### 5.1. Xác định bài toán

Trả lời đủ các câu hỏi sau trước khi viết dòng code nào:

1. **Mục tiêu kinh doanh/nghiên cứu** là gì?
2. **Label** là gì? Có sẵn hay phải tạo?
3. **Loại bài toán:** hồi quy / phân loại / phân cụm / chuỗi thời gian...?
4. **Metric thành công:** RMSE? F1? Recall? (chọn theo hậu quả của sai lầm)
5. **Ràng buộc:** thời gian dự đoán, giải thích được, dữ liệu nhạy cảm?
6. **Baseline:** hiện tại người ta làm thế nào? (trung bình, luật đơn giản)

**Ví dụ:**

| Bài toán | Label | Loại | Metric gợi ý |
|---|---|---|---|
| Dự đoán giá nhà | Giá | Hồi quy | RMSE, MAE |
| Phát hiện email spam | Spam/Không | Phân loại nhị phân | Precision, F1 |
| Chẩn đoán bệnh | Bệnh/Không | Phân loại (ưu tiên Recall) | Recall, AUC |
| Phân khúc khách hàng | Không có | Phân cụm | Silhouette |
| Dự báo doanh thu tuần tới | Doanh thu | Chuỗi thời gian | MAPE, RMSE |

### 5.2. Nguồn dữ liệu

| Nguồn | Cách lấy | Ghi chú |
|---|---|---|
| File CSV/Excel | `pd.read_csv`, `pd.read_excel` | Phổ biến nhất |
| Cơ sở dữ liệu | `pd.read_sql` + SQLAlchemy | Dữ liệu doanh nghiệp |
| API | `requests` | JSON |
| Web scraping | `BeautifulSoup`, `Scrapy`, `Selenium` | Kiểm tra điều khoản website |
| Dataset công khai | Kaggle, UCI, Hugging Face Datasets, Google Dataset Search | Học tập |
| Tự thu thập | Khảo sát, cảm biến, log | Cần thiết kế cẩn thận |

```python
# CSV / Excel / JSON
df = pd.read_csv("data.csv", encoding="utf-8")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_json("data.json")

# SQL
from sqlalchemy import create_engine
engine = create_engine("sqlite:///mydb.db")
df = pd.read_sql("SELECT * FROM khach_hang WHERE tuoi > 18", engine)

# API
import requests
res = requests.get("https://api.example.com/data", params={"limit": 100})
df = pd.DataFrame(res.json())

# Dataset có sẵn trong scikit-learn
from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
```

### 5.3. Dataset học tập gợi ý theo dạng

| Dạng | Dataset |
|---|---|
| Bảng, hồi quy | California Housing, Boston→Ames Housing, Diabetes |
| Bảng, phân loại | Titanic, Iris, Breast Cancer, Adult Income |
| Chuỗi thời gian | Air Passengers, Store Sales (Kaggle) |
| Văn bản | IMDB Reviews, 20 Newsgroups, SMS Spam |
| Ảnh | MNIST, Fashion-MNIST, CIFAR-10, Cats vs Dogs |
| Âm thanh | Speech Commands, UrbanSound8K |
| Gợi ý | MovieLens |

### 5.4. Kiểm tra chất lượng dữ liệu ban đầu

- Đủ **số mẫu**? (quy tắc thô: ít nhất 10-20 mẫu cho mỗi feature với mô hình đơn giản)
- Label có **đáng tin**? Ai gán nhãn, nhất quán không?
- Dữ liệu có **đại diện** cho tình huống thực tế không? (tránh thiên lệch lấy mẫu)
- Có vấn đề **pháp lý/đạo đức/quyền riêng tư**? (dữ liệu cá nhân, y tế)

---

## 6. Bước 3: Khám phá dữ liệu (EDA)

Mục tiêu: **hiểu dữ liệu trước khi mô hình hóa**, phát hiện vấn đề, hình thành giả thuyết.

### 6.1. Quy trình EDA

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Kích thước và kiểu dữ liệu
print(df.shape)
print(df.dtypes)
df.info()

# 2. Thống kê mô tả
df.describe()                       # cột số
df.describe(include="object")       # cột chữ

# 3. Giá trị thiếu
thieu = df.isnull().sum().sort_values(ascending=False)
ty_le = (thieu / len(df) * 100).round(2)
print(pd.DataFrame({"thieu": thieu, "ty_le_%": ty_le}))

# 4. Dòng trùng lặp
print("Số dòng trùng:", df.duplicated().sum())

# 5. Phân phối label
sns.histplot(df["gia"], kde=True)                 # hồi quy
sns.countplot(x="nhan", data=df)                  # phân loại (kiểm tra mất cân bằng)
df["nhan"].value_counts(normalize=True)

# 6. Phân phối từng feature số
df.hist(figsize=(14, 10), bins=30)

# 7. Feature phân loại
for c in df.select_dtypes("object").columns:
    print(df[c].value_counts().head(10), "\n")

# 8. Tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0)

# 9. Quan hệ feature - label
sns.scatterplot(x="dien_tich", y="gia", data=df)
sns.boxplot(x="quan", y="gia", data=df)
sns.pairplot(df[["gia", "dien_tich", "so_phong"]])
```

### 6.2. Bảng chọn biểu đồ

| Muốn xem | Biểu đồ |
|---|---|
| Phân phối 1 biến số | Histogram, KDE, Boxplot |
| Phân phối 1 biến phân loại | Bar chart, Count plot |
| Quan hệ 2 biến số | Scatter plot |
| Biến phân loại vs biến số | Boxplot, Violin plot |
| 2 biến phân loại | Bảng chéo (crosstab), Stacked bar |
| Tương quan nhiều biến | Heatmap, Pair plot |
| Xu hướng thời gian | Line plot |
| Phân bố địa lý | Bản đồ (folium, geopandas) |

### 6.3. Những điều cần tìm trong EDA

| Quan sát | Hàm ý |
|---|---|
| Phân phối lệch phải (giá, thu nhập) | Cân nhắc biến đổi log |
| Ngoại lai (outlier) | Kiểm tra: lỗi nhập liệu hay giá trị thật? |
| Thiếu dữ liệu nhiều | Quyết định điền/bỏ cột/bỏ dòng |
| Hai feature tương quan > 0.9 | Đa cộng tuyến, cân nhắc bỏ một |
| Label mất cân bằng | Dùng metric phù hợp, resampling, class_weight |
| Feature gần như hằng số | Bỏ |
| Cột ID, tên | Không dùng làm feature |
| Quan hệ phi tuyến x-y | Dùng đa thức hoặc mô hình phi tuyến |
| Feature có thể rò rỉ label | Loại bỏ (data leakage) |

### 6.4. Công cụ EDA tự động (tham khảo)

```bash
pip install ydata-profiling sweetviz
```
```python
from ydata_profiling import ProfileReport
ProfileReport(df, title="Báo cáo EDA").to_file("eda.html")
```

> Dùng để tham khảo nhanh, nhưng **vẫn phải tự phân tích** để hiểu dữ liệu.

---

## 7. Bước 4: Làm sạch và tiền xử lý

### 7.1. Xử lý giá trị thiếu

| Cách | Khi nào dùng | Code |
|---|---|---|
| Xóa dòng | Thiếu rất ít, ngẫu nhiên | `df.dropna(subset=["cot"])` |
| Xóa cột | Thiếu > 50-60%, ít giá trị | `df.drop(columns=["cot"])` |
| Điền trung bình | Số, phân phối đối xứng | `fillna(df["c"].mean())` |
| Điền trung vị | Số, có ngoại lai/lệch | `fillna(df["c"].median())` |
| Điền yếu vị (mode) | Phân loại | `fillna(df["c"].mode()[0])` |
| Điền "Unknown" + cột cờ | Thiếu **có ý nghĩa** | `df["c_thieu"] = df["c"].isna()` |
| Nội suy | Chuỗi thời gian | `df["c"].interpolate()` |
| KNN / Iterative Imputer | Muốn điền thông minh | `KNNImputer`, `IterativeImputer` |

```python
from sklearn.impute import SimpleImputer, KNNImputer

imp = SimpleImputer(strategy="median")
X_train[cot_so] = imp.fit_transform(X_train[cot_so])   # fit trên train
X_test[cot_so]  = imp.transform(X_test[cot_so])        # chỉ transform trên test
```

> **Quan trọng:** `fit` bộ điền trên **tập train**, rồi `transform` cho test, để tránh rò rỉ dữ liệu.

### 7.2. Xử lý dữ liệu trùng và sai

```python
df = df.drop_duplicates()

# Chuẩn hóa chuỗi
df["quan"] = df["quan"].str.strip().str.lower()
df["quan"] = df["quan"].replace({"q7": "quan 7", "quận 7": "quan 7"})

# Sửa kiểu dữ liệu
df["ngay"] = pd.to_datetime(df["ngay"], errors="coerce")
df["gia"] = pd.to_numeric(df["gia"], errors="coerce")

# Giá trị vô lý
df = df[(df["tuoi"] >= 0) & (df["tuoi"] <= 120)]
```

### 7.3. Xử lý ngoại lai (outlier)

**Phát hiện:**

```python
# IQR
Q1, Q3 = df["gia"].quantile([0.25, 0.75])
IQR = Q3 - Q1
thap, cao = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outlier = df[(df["gia"] < thap) | (df["gia"] > cao)]

# Z-score
from scipy import stats
z = np.abs(stats.zscore(df["gia"]))
outlier = df[z > 3]
```

**Xử lý:**

| Cách | Khi nào |
|---|---|
| Giữ nguyên | Là giá trị thật và quan trọng (ví dụ gian lận) |
| Xóa | Chắc chắn là lỗi |
| Giới hạn (winsorize/clip) | Muốn giảm ảnh hưởng nhưng giữ dòng |
| Biến đổi log | Phân phối lệch nặng |
| Dùng mô hình bền vững | Cây, Huber Regression |

```python
df["gia"] = df["gia"].clip(lower=thap, upper=cao)   # clip
df["gia_log"] = np.log1p(df["gia"])                  # log
```

### 7.4. Mã hóa feature phân loại

| Phương pháp | Dùng khi | Lưu ý |
|---|---|---|
| **One-Hot** | Danh nghĩa, ít giá trị (< ~15) | Tăng số cột |
| **Ordinal** | Có thứ tự | Phải đặt đúng thứ tự |
| **Label Encoding** | Chỉ cho **label** hoặc mô hình cây | Không dùng cho mô hình tuyến tính với feature danh nghĩa |
| **Target Encoding** | Nhiều giá trị (mã bưu điện) | Dễ rò rỉ, phải làm trong cross-validation |
| **Frequency Encoding** | Nhiều giá trị | Thay bằng tần suất xuất hiện |
| **Hashing** | Cực nhiều giá trị | Có va chạm |
| **Embedding** | Deep learning | Học được biểu diễn |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal = OrdinalEncoder(categories=[["THPT", "Dai hoc", "Thac si"]])
```

### 7.5. Chuẩn hóa / co giãn thang đo (Scaling)

| Phương pháp | Công thức | Dùng khi |
|---|---|---|
| **StandardScaler** | (x − μ) / σ | Phổ biến nhất, phân phối gần chuẩn |
| **MinMaxScaler** | (x − min)/(max − min) | Cần khoảng [0,1], nơ-ron, ảnh |
| **RobustScaler** | (x − median)/IQR | Có nhiều ngoại lai |
| **Log / Power Transform** | log(x), Box-Cox, Yeo-Johnson | Dữ liệu lệch |

**Mô hình nào cần scaling?**

| Cần | Không cần |
|---|---|
| Hồi quy tuyến tính/logistic có regularization, KNN, SVM, K-Means, PCA, Nơ-ron | Cây quyết định, Random Forest, XGBoost, LightGBM |

### 7.6. Xử lý dữ liệu mất cân bằng

| Kỹ thuật | Mô tả |
|---|---|
| `class_weight="balanced"` | Cho lớp hiếm trọng số cao hơn (đơn giản nhất, nên thử trước) |
| **Oversampling (SMOTE)** | Sinh thêm mẫu lớp hiếm |
| **Undersampling** | Giảm mẫu lớp đông |
| Đổi ngưỡng phân loại | Hạ ngưỡng dưới 0.5 để tăng Recall |
| Metric phù hợp | PR-AUC, F1, Recall thay vì Accuracy |

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)   # CHỈ áp dụng trên train
```

### 7.7. Pipeline: gom tiền xử lý để tránh rò rỉ

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

cot_so = ["dien_tich", "so_phong", "so_nam_xay"]
cot_pl = ["quan", "loai_nha"]

so_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
pl_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

tien_xu_ly = ColumnTransformer([
    ("so", so_pipe, cot_so),
    ("pl", pl_pipe, cot_pl),
])
```

---

## 8. Bước 5: Feature Engineering và Feature Selection

### 8.1. Feature Engineering: tạo feature mới

| Kỹ thuật | Ví dụ |
|---|---|
| **Tổ hợp số học** | `gia_m2 = gia/dien_tich`, `BMI = can_nang/chieu_cao²` |
| **Tương tác** | `dien_tich × so_phong` |
| **Đa thức** | `x²`, `x³` (`PolynomialFeatures`) |
| **Biến đổi** | `log(x)`, `sqrt(x)`, `1/x` |
| **Rời rạc hóa (binning)** | Nhóm tuổi, nhóm thu nhập |
| **Thời gian** | Năm, tháng, thứ, giờ, cuối tuần, ngày lễ; dạng tuần hoàn dùng sin/cos |
| **Tổng hợp theo nhóm** | Chi tiêu trung bình của mỗi khách hàng, số giao dịch 30 ngày gần nhất |
| **Độ trễ (lag), cửa sổ trượt** | Doanh thu hôm qua, trung bình 7 ngày |
| **Văn bản** | Độ dài, số từ, TF-IDF, embedding |
| **Địa lý** | Khoảng cách tới trung tâm, cụm vị trí |

```python
# Thời gian
df["nam"] = df["ngay"].dt.year
df["thang"] = df["ngay"].dt.month
df["thu"] = df["ngay"].dt.dayofweek
df["cuoi_tuan"] = df["thu"].isin([5, 6]).astype(int)
df["thang_sin"] = np.sin(2 * np.pi * df["thang"] / 12)
df["thang_cos"] = np.cos(2 * np.pi * df["thang"] / 12)

# Tổng hợp theo nhóm
tb_kh = df.groupby("ma_kh")["so_tien"].transform("mean")
df["so_tien_vs_tb"] = df["so_tien"] / tb_kh

# Lag và rolling (chuỗi thời gian)
df = df.sort_values("ngay")
df["lag_1"] = df["doanh_thu"].shift(1)
df["tb_7ngay"] = df["doanh_thu"].shift(1).rolling(7).mean()   # shift(1) để không dùng thông tin tương lai
```

> **Cảnh báo rò rỉ:** feature phải **tính được tại thời điểm dự đoán**. Với chuỗi thời gian, luôn `shift` trước khi `rolling`.

### 8.2. Feature Selection: chọn feature

| Nhóm | Phương pháp | Mô tả |
|---|---|---|
| **Filter** | Variance Threshold, tương quan, `SelectKBest` (chi², F-test, mutual info) | Nhanh, độc lập với mô hình |
| **Wrapper** | RFE (Recursive Feature Elimination), Forward/Backward selection | Chính xác hơn, tốn thời gian |
| **Embedded** | Lasso (L1), feature importance của cây | Chọn ngay trong lúc huấn luyện |
| **Giảm chiều** | PCA, LDA, t-SNE/UMAP (trực quan) | Nén thông tin |

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Filter
sel = SelectKBest(score_func=f_regression, k=10).fit(X_train, y_train)

# Wrapper
rfe = RFE(LinearRegression(), n_features_to_select=8).fit(X_train, y_train)

# Embedded: Lasso
lasso = LassoCV(cv=5).fit(X_train, y_train)
quan_trong = pd.Series(lasso.coef_, index=X_train.columns)
print(quan_trong[quan_trong != 0])

# Embedded: Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Đa cộng tuyến: VIF (> 5-10 là cao)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
```

---

## 9. Bước 6: Chia dữ liệu và Cross-validation

### 9.1. Tại sao chia 3 tập?

| Tập | Tỷ lệ gợi ý | Vai trò |
|---|---|---|
| **Train** | 60-80% | Mô hình học tham số |
| **Validation** | 10-20% | Chọn mô hình, tinh chỉnh siêu tham số |
| **Test** | 10-20% | Đánh giá cuối cùng, **chỉ dùng 1 lần** |

> Nếu nhìn tập test nhiều lần để chỉnh mô hình, test không còn "chưa thấy" nữa.

### 9.2. Các cách chia

```python
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     TimeSeriesSplit, GroupKFold, cross_val_score)

# Chia ngẫu nhiên
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Phân loại mất cân bằng: giữ tỷ lệ lớp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Chuỗi thời gian: KHÔNG xáo trộn, quá khứ để train, tương lai để test
cat = int(len(df) * 0.8)
train, test = df.iloc[:cat], df.iloc[cat:]
```

| Tình huống | Cách chia |
|---|---|
| Dữ liệu bảng thông thường | `KFold` |
| Phân loại, mất cân bằng | `StratifiedKFold` |
| Chuỗi thời gian | `TimeSeriesSplit` (không xáo trộn) |
| Nhiều mẫu cùng một đối tượng (cùng bệnh nhân, cùng khách hàng) | `GroupKFold` (tránh cùng đối tượng ở cả train và test) |
| Dữ liệu rất ít | Leave-One-Out, Repeated K-Fold |

### 9.3. Cross-validation

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
diem = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
print(f"F1 = {diem.mean():.3f} ± {diem.std():.3f}")
```

Ý nghĩa: chia train thành K phần, lần lượt dùng 1 phần làm validation, huấn luyện K lần, lấy trung bình để ước lượng ổn định hơn.

---

## 10. Bước 7: Chọn và huấn luyện mô hình

### 10.1. Quy tắc làm việc

1. **Baseline trước:** mô hình ngu ngốc nhất (dự đoán trung bình / lớp phổ biến nhất).
2. **Mô hình đơn giản:** hồi quy tuyến tính/logistic.
3. **Mô hình mạnh hơn:** Random Forest, Gradient Boosting.
4. **Deep learning:** chỉ khi dữ liệu lớn hoặc là ảnh/văn bản/âm thanh.

```python
from sklearn.dummy import DummyRegressor, DummyClassifier
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
```

### 10.2. Bảng chọn mô hình theo bài toán

| Bài toán | Mô hình thử theo thứ tự |
|---|---|
| **Hồi quy (bảng)** | Linear/Ridge/Lasso → Random Forest → XGBoost/LightGBM → MLP |
| **Phân loại (bảng)** | Logistic → Cây → Random Forest → XGBoost/LightGBM/CatBoost |
| **Phân loại văn bản** | TF-IDF + Logistic/SVM/Naive Bayes → BERT/PhoBERT |
| **Ảnh** | CNN nhỏ → Transfer learning (ResNet, EfficientNet) → ViT |
| **Chuỗi thời gian** | Naive/Seasonal Naive → ARIMA/ETS/Prophet → XGBoost với lag → LSTM/Transformer |
| **Phân cụm** | K-Means → Hierarchical → DBSCAN → Gaussian Mixture |
| **Phát hiện bất thường** | Isolation Forest → One-Class SVM → Autoencoder |
| **Gợi ý** | Popularity → Collaborative Filtering → Matrix Factorization → Neural CF |
| **Giảm chiều** | PCA → t-SNE/UMAP (chỉ để trực quan) |

### 10.3. Các thuật toán chính: tham số quan trọng

| Thuật toán | Lớp trong sklearn | Siêu tham số quan trọng |
|---|---|---|
| Hồi quy tuyến tính | `LinearRegression` | (không có) |
| Ridge / Lasso / ElasticNet | `Ridge`, `Lasso`, `ElasticNet` | `alpha`, `l1_ratio` |
| Hồi quy logistic | `LogisticRegression` | `C`, `penalty`, `class_weight` |
| Cây quyết định | `DecisionTreeClassifier/Regressor` | `max_depth`, `min_samples_leaf`, `criterion` |
| Random Forest | `RandomForestClassifier/Regressor` | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| Gradient Boosting | `GradientBoosting*`, `XGBoost`, `LightGBM` | `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree` |
| KNN | `KNeighborsClassifier` | `n_neighbors`, `weights`, `metric` |
| SVM | `SVC`, `SVR` | `C`, `kernel`, `gamma` |
| Naive Bayes | `GaussianNB`, `MultinomialNB` | `alpha` (Multinomial) |
| K-Means | `KMeans` | `n_clusters`, `n_init` |
| DBSCAN | `DBSCAN` | `eps`, `min_samples` |
| PCA | `PCA` | `n_components` |
| MLP | `MLPClassifier/Regressor` | `hidden_layer_sizes`, `alpha`, `learning_rate_init` |

### 10.3.1. Hồi quy: chọn dạng công thức

| Hình dạng dữ liệu | Công thức |
|---|---|
| Đường thẳng | `y = wx + b` |
| Đường cong | `y = w₂x² + w₁x + b` |
| Tăng chậm dần | `y = w·ln(x) + b` |
| Tăng/giảm theo cấp số nhân | `ln(y) = wx + b` |
| Nhiều biến, đa cộng tuyến | Ridge |
| Nhiều biến, muốn loại biến thừa | Lasso |
| Y là số đếm | Poisson |
| Y là xác suất / 0-1 | Logistic |

Chọn dựa trên: scatter plot, residual plot, cross-validation (RMSE, R²), AIC/BIC, nguyên tắc "đơn giản hơn nếu ngang nhau".

### 10.4. Code so sánh nhiều mô hình

```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ung_vien = {
    "Logistic":     make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN":          make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "SVM":          make_pipeline(StandardScaler(), SVC()),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost":      XGBClassifier(n_estimators=300, learning_rate=0.05, eval_metric="logloss"),
}

ket_qua = []
for ten, m in ung_vien.items():
    cv = cross_validate(m, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    ket_qua.append({
        "Mo hinh": ten,
        "Accuracy": cv["test_accuracy"].mean(),
        "F1": cv["test_f1"].mean(),
        "AUC": cv["test_roc_auc"].mean(),
    })
print(pd.DataFrame(ket_qua).sort_values("F1", ascending=False))
```

### 10.5. Gradient Descent: hiểu cách mô hình học

```python
# Hồi quy tuyến tính bằng Gradient Descent từ đầu (để hiểu bản chất)
import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        y_hat = X @ w + b
        loi = y_hat - y
        dw = (2 / n) * X.T @ loi          # đạo hàm MSE theo w
        db = (2 / n) * loi.sum()          # đạo hàm MSE theo b
        w -= lr * dw
        b -= lr * db
    return w, b
```

Khái niệm đi kèm: **hàm mất mát (loss)**, **learning rate**, **epoch**, **batch / mini-batch / SGD**.

---

## 11. Bước 8: Đánh giá mô hình (Metrics)

### 11.1. Hồi quy

| Metric | Ý nghĩa | Ghi chú |
|---|---|---|
| **MAE** | Sai số tuyệt đối trung bình | Dễ hiểu, bền với ngoại lai |
| **MSE / RMSE** | Sai số bình phương (căn) | Phạt nặng sai số lớn |
| **R²** | Tỷ lệ phương sai giải thích được (1 là tốt nhất) | Có thể âm nếu tệ hơn baseline |
| **Adjusted R²** | R² có phạt số biến | So sánh mô hình khác số feature |
| **MAPE** | Sai số phần trăm | Lỗi nếu y gần 0 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R²  :", r2_score(y_test, y_pred))
```

### 11.2. Phân loại

**Ma trận nhầm lẫn (Confusion Matrix):**

|  | Dự đoán Dương | Dự đoán Âm |
|---|---|---|
| **Thực Dương** | TP | FN |
| **Thực Âm** | FP | TN |

| Metric | Công thức | Dùng khi |
|---|---|---|
| **Accuracy** | (TP+TN)/Tổng | Lớp cân bằng |
| **Precision** | TP/(TP+FP) | Báo nhầm tốn kém (spam) |
| **Recall** | TP/(TP+FN) | Bỏ sót tốn kém (bệnh, gian lận) |
| **F1** | 2·P·R/(P+R) | Cần cân bằng P và R |
| **ROC-AUC** | Diện tích dưới đường ROC | Đánh giá khả năng xếp hạng |
| **PR-AUC** | Diện tích dưới đường Precision-Recall | Lớp rất hiếm |
| **Log-loss** | Mất mát entropy chéo | Cần xác suất chuẩn |

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
```

**Đa lớp:** dùng `average="macro"` (mỗi lớp như nhau), `"weighted"` (theo số mẫu), `"micro"` (gộp toàn bộ).

### 11.3. Phân cụm

| Metric | Có label thật? | Ý nghĩa |
|---|---|---|
| Silhouette | Không | −1 đến 1, càng cao càng tách biệt |
| Davies-Bouldin | Không | Càng thấp càng tốt |
| Elbow (inertia) | Không | Chọn K tại điểm "khuỷu tay" |
| Adjusted Rand Index | Có | So với phân nhóm thật |

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    print(k, km.inertia_, silhouette_score(X_scaled, km.labels_))
```

### 11.4. Chuỗi thời gian

MAE, RMSE, MAPE, sMAPE, và so sánh với **baseline naive** (dự đoán bằng giá trị kỳ trước).

### 11.5. Phân tích lỗi (không bỏ qua!)

- Xem **các mẫu bị dự đoán sai nhiều nhất**: có điểm chung không?
- Tính metric **theo nhóm con** (theo quận, giới tính, độ tuổi) để phát hiện thiên lệch.
- **Residual plot** (hồi quy): phần dư nên ngẫu nhiên quanh 0.
- **Learning curve**: điểm train/validation theo số mẫu, để biết nên thêm dữ liệu hay đổi mô hình.
- **Calibration**: xác suất dự đoán 0.8 có thật sự đúng ~80%?

---

## 12. Bước 9: Tinh chỉnh và chống Overfitting

### 12.1. Chẩn đoán Bias-Variance

| Triệu chứng | Chẩn đoán | Hướng xử lý |
|---|---|---|
| Train thấp, test thấp | **Underfitting** (bias cao) | Mô hình phức tạp hơn, thêm feature, giảm regularization, huấn luyện lâu hơn |
| Train cao, test thấp | **Overfitting** (variance cao) | Thêm dữ liệu, regularization, giảm độ phức tạp, bớt feature, early stopping, dropout |
| Train ≈ test, đều tốt | Tốt | Kiểm tra lại leakage nếu quá tốt để là thật |

### 12.2. Kỹ thuật chống overfitting

| Kỹ thuật | Áp dụng cho |
|---|---|
| Thêm dữ liệu / data augmentation | Mọi mô hình, đặc biệt ảnh/văn bản |
| **L1/L2 regularization** | Hồi quy, nơ-ron |
| Giới hạn `max_depth`, `min_samples_leaf` | Cây |
| Pruning | Cây |
| Ensemble (bagging) | Random Forest |
| **Early stopping** | Boosting, nơ-ron |
| **Dropout, Batch Norm** | Nơ-ron |
| Giảm số feature | Mọi mô hình |
| Cross-validation | Mọi mô hình |

### 12.3. Tinh chỉnh siêu tham số

| Phương pháp | Đặc điểm |
|---|---|
| **Grid Search** | Thử mọi tổ hợp, chậm khi nhiều tham số |
| **Random Search** | Thử ngẫu nhiên, thường hiệu quả hơn cùng ngân sách |
| **Bayesian (Optuna)** | Chọn thông minh dựa trên các lần thử trước |

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

tham_so = {
    "n_estimators": randint(100, 600),
    "max_depth": randint(3, 20),
    "min_samples_leaf": randint(1, 20),
    "max_features": uniform(0.3, 0.7),
}
tim = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    tham_so, n_iter=40, cv=5, scoring="f1", n_jobs=-1, random_state=42,
)
tim.fit(X_train, y_train)
print(tim.best_params_, tim.best_score_)
model_tot_nhat = tim.best_estimator_
```

```python
# Optuna
import optuna
def muc_tieu(trial):
    m = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
    )
    return cross_val_score(m, X_train, y_train, cv=5, scoring="f1").mean()

study = optuna.create_study(direction="maximize")
study.optimize(muc_tieu, n_trials=50)
```

### 12.4. Ensemble

| Kỹ thuật | Ý tưởng |
|---|---|
| **Bagging** | Nhiều mô hình độc lập trên mẫu bootstrap (Random Forest) |
| **Boosting** | Mô hình sau sửa lỗi mô hình trước (XGBoost) |
| **Voting** | Bỏ phiếu giữa các mô hình khác loại |
| **Stacking** | Mô hình cấp 2 học cách kết hợp dự đoán của mô hình cấp 1 |

---

## 13. Bước 10: Giải thích, lưu và triển khai

### 13.1. Giải thích mô hình

| Công cụ | Mục đích |
|---|---|
| Hệ số (coefficients) | Mô hình tuyến tính |
| Feature importance | Cây, Random Forest |
| **Permutation Importance** | Mọi mô hình, đo mức giảm hiệu suất khi xáo trộn feature |
| **SHAP** | Đóng góp của từng feature cho từng dự đoán |
| **LIME** | Giải thích cục bộ |
| Partial Dependence Plot | Ảnh hưởng của một feature lên dự đoán |

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

### 13.2. Đánh giá cuối trên tập test

```python
# Chỉ làm 1 lần, sau khi đã chốt mô hình
y_pred = model_tot_nhat.predict(X_test)
print(classification_report(y_test, y_pred))
```

Sau đó có thể **huấn luyện lại trên train + validation** với tham số đã chốt trước khi triển khai.

### 13.3. Lưu và nạp mô hình

```python
import joblib
joblib.dump(model_tot_nhat, "model.joblib")        # lưu cả Pipeline (gồm tiền xử lý)
model = joblib.load("model.joblib")
model.predict(du_lieu_moi)
```

### 13.4. Triển khai API đơn giản (FastAPI)

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI()
model = joblib.load("model.joblib")

class NhaInput(BaseModel):
    dien_tich: float
    so_phong: int
    quan: str
    so_nam_xay: int

@app.post("/du-doan")
def du_doan(x: NhaInput):
    df = pd.DataFrame([x.model_dump()])
    return {"gia_du_doan": float(model.predict(df)[0])}
```

```bash
pip install fastapi uvicorn
uvicorn app:app --reload
```

Lựa chọn khác: **Streamlit** hoặc **Gradio** (demo giao diện nhanh), **Docker** (đóng gói), dịch vụ cloud (AWS SageMaker, GCP Vertex AI, Azure ML).

### 13.5. MLOps cơ bản

| Hạng mục | Công cụ/ý tưởng |
|---|---|
| Quản lý phiên bản code | Git |
| Quản lý phiên bản dữ liệu | DVC |
| Theo dõi thí nghiệm | MLflow, Weights & Biases |
| Tái lập kết quả | Cố định `random_state`, ghi lại phiên bản thư viện (`requirements.txt`) |
| Giám sát | Theo dõi **data drift** (phân phối đầu vào đổi), **concept drift** (quan hệ X-y đổi), hiệu suất theo thời gian |
| Huấn luyện lại | Định kỳ hoặc khi hiệu suất giảm |

---

## 14. Quy trình riêng theo từng dạng dữ liệu

### 14.1. Dữ liệu bảng (Tabular)

1. EDA → làm sạch → mã hóa → (scale nếu cần)
2. Baseline → Linear/Logistic → Random Forest → XGBoost/LightGBM/CatBoost
3. Feature engineering là đòn bẩy lớn nhất
4. CatBoost/LightGBM xử lý feature phân loại tốt, ít cần One-Hot

### 14.2. Chuỗi thời gian

**Đặc thù:** thứ tự thời gian quan trọng, **không xáo trộn khi chia**.

```python
# 1. Kiểm tra xu hướng, mùa vụ
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(series, model="additive", period=12).plot()

# 2. Kiểm tra tính dừng (ADF test)
from statsmodels.tsa.stattools import adfuller
print(adfuller(series)[1])     # p < 0.05 → dừng

# 3. Mô hình thống kê
from statsmodels.tsa.arima.model import ARIMA
fit = ARIMA(train, order=(1, 1, 1)).fit()
du_bao = fit.forecast(steps=12)

# 4. Hoặc ML với feature lag/rolling + TimeSeriesSplit
```

Quy trình: baseline naive → phân rã → ARIMA/ETS/Prophet → ML với lag feature → deep learning (LSTM, Temporal Fusion Transformer). Chia bằng **TimeSeriesSplit** hoặc walk-forward validation.

### 14.3. Văn bản (NLP)

**Tiền xử lý:** chuẩn hóa chữ thường, bỏ ký tự thừa, tách từ (tiếng Việt dùng `underthesea`, `pyvi`), bỏ stopword (tùy bài).

**Biểu diễn văn bản:**

| Cấp | Phương pháp | Ghi chú |
|---|---|---|
| 1 | Bag-of-Words, **TF-IDF** | Nhanh, baseline mạnh |
| 2 | Word2Vec, GloVe, FastText | Vector từ tĩnh |
| 3 | **BERT, PhoBERT, XLM-R** (Transformer) | Hiểu ngữ cảnh, fine-tune |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train_text, y_train)
```

```python
# Fine-tune Transformer (rút gọn)
from transformers import pipeline
clf = pipeline("sentiment-analysis", model="vinai/phobert-base")   # ví dụ, cần mô hình đã fine-tune phù hợp
```

Bài toán NLP: phân loại văn bản, phân tích cảm xúc, nhận dạng thực thể (NER), tóm tắt, dịch, hỏi đáp, sinh văn bản.

### 14.4. Ảnh (Computer Vision)

**Tiền xử lý:** resize, chuẩn hóa pixel (0-1 hoặc theo mean/std của ImageNet), **data augmentation** (lật, xoay, cắt, đổi màu).

**Lộ trình mô hình:** MLP trên pixel → **CNN** tự xây → **Transfer Learning** → Fine-tuning.

```python
import torch, torch.nn as nn
from torchvision import models, transforms, datasets

# Transfer learning với ResNet18
model = models.resnet18(weights="DEFAULT")
for p in model.parameters():
    p.requires_grad = False                       # đóng băng
model.fc = nn.Linear(model.fc.in_features, so_lop)  # thay lớp cuối

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

Bài toán: phân loại ảnh, phát hiện đối tượng (YOLO), phân đoạn (U-Net), nhận dạng khuôn mặt, OCR.

### 14.5. Âm thanh

Biến sóng âm thành **spectrogram / mel-spectrogram / MFCC** rồi xử lý như ảnh bằng CNN hoặc dùng mô hình dựa trên Transformer (Wav2Vec2, Whisper).

```python
import librosa
y, sr = librosa.load("audio.wav", sr=16000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
```

### 14.6. Mạng nơ-ron: quy trình huấn luyện với PyTorch

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

X_t = torch.tensor(X_train.values, dtype=torch.float32)
y_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(X_t.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1),
)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(50):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()          # lan truyền ngược
        opt.step()               # cập nhật trọng số
    # Nhớ: đánh giá validation mỗi epoch, dùng early stopping
```

Khái niệm cần nắm: nơ-ron, lớp, hàm kích hoạt (ReLU, Sigmoid, Softmax), forward/backward, loss, optimizer (SGD, Adam), batch, epoch, dropout, batch norm, vanishing gradient.

### 14.7. Học không giám sát

| Mục tiêu | Thuật toán | Lưu ý |
|---|---|---|
| Phân cụm | K-Means, Hierarchical, DBSCAN, GMM | Scale dữ liệu; chọn K bằng Elbow/Silhouette |
| Giảm chiều | PCA, t-SNE, UMAP | PCA giữ ~90-95% phương sai; t-SNE/UMAP chỉ để vẽ |
| Bất thường | Isolation Forest, One-Class SVM, Autoencoder | Ít/không có nhãn bất thường |
| Luật kết hợp | Apriori, FP-Growth | Giỏ hàng: "mua A thường mua B" |

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95).fit(X_scaled)     # giữ 95% phương sai
print(pca.n_components_, pca.explained_variance_ratio_.cumsum())
```

### 14.8. Hệ thống gợi ý

1. **Popularity:** gợi ý mục phổ biến (baseline)
2. **Content-based:** dựa trên đặc điểm món đồ (TF-IDF mô tả, thể loại)
3. **Collaborative Filtering:** user-user, item-item (cosine similarity)
4. **Matrix Factorization:** SVD, ALS
5. **Deep learning:** Neural CF, Two-Tower
Metric: Precision@K, Recall@K, MAP, NDCG.

### 14.9. Học tăng cường (giới thiệu)

Khái niệm: **agent, environment, state, action, reward, policy**. Thuật toán: Q-Learning → DQN → Policy Gradient/PPO. Công cụ: `gymnasium`, `stable-baselines3`.

---

## 15. Lộ trình học theo tuần

| Tuần | Chủ đề | Việc cần làm |
|---|---|---|
| 1-2 | Python, NumPy, Pandas | Làm 20+ bài tập xử lý bảng; SQL cơ bản |
| 3 | Thống kê và trực quan hóa | Matplotlib/Seaborn; mô tả, tương quan, kiểm định |
| 4 | Tổng quan ML, feature/label, train/test | Mini-project với Iris, Titanic |
| 5-6 | Hồi quy tuyến tính, đa thức, Ridge/Lasso, logistic | Dự đoán giá nhà; tự cài Gradient Descent |
| 7 | Đánh giá, Cross-validation, Overfitting/Underfitting | Vẽ learning curve; so sánh metric |
| 8 | Cây quyết định, Random Forest | Dự đoán khách rời bỏ |
| 9 | Boosting (XGBoost/LightGBM), tinh chỉnh | Tham gia 1 cuộc thi Kaggle "Getting Started" |
| 10 | KNN, SVM, Naive Bayes | So sánh toàn bộ trên 1 dataset |
| 11 | Học không giám sát: K-Means, PCA | Phân khúc khách hàng |
| 12 | Feature engineering, Pipeline, xử lý mất cân bằng | Dự án phát hiện gian lận |
| 13-14 | Mạng nơ-ron, PyTorch/Keras | MNIST, tự viết vòng lặp huấn luyện |
| 15 | CNN, Transfer learning | Phân loại ảnh Cats vs Dogs |
| 16 | NLP: TF-IDF, Transformer | Phân tích cảm xúc bình luận tiếng Việt |
| 17 | Chuỗi thời gian | Dự báo doanh thu |
| 18 | Giải thích mô hình (SHAP), MLOps cơ bản | Đóng gói bằng FastAPI/Streamlit |
| 19-20 | **Dự án cuối kỳ end-to-end** | Đưa lên GitHub, viết README, demo |

> Thời gian chỉ mang tính tham khảo. Quan trọng là **làm dự án thật**, không chỉ xem video.

---

## 16. Dự án thực hành gợi ý

| Mức | Dự án | Dạng dữ liệu | Kỹ năng |
|---|---|---|---|
| Cơ bản | Dự đoán sống sót Titanic | Bảng | EDA, mã hóa, phân loại |
| Cơ bản | Dự đoán giá nhà | Bảng | Hồi quy, feature engineering |
| Cơ bản | Phân loại hoa Iris | Bảng | Toàn bộ quy trình nhỏ |
| Trung bình | Dự đoán khách hàng rời bỏ | Bảng, mất cân bằng | Metric, SMOTE, SHAP |
| Trung bình | Phân khúc khách hàng (RFM) | Bảng | K-Means, PCA |
| Trung bình | Lọc spam SMS | Văn bản | TF-IDF, Naive Bayes |
| Trung bình | Dự báo doanh thu cửa hàng | Chuỗi thời gian | Lag feature, TimeSeriesSplit |
| Nâng cao | Phân loại ảnh bệnh trên lá cây | Ảnh | CNN, Transfer learning, augmentation |
| Nâng cao | Phân tích cảm xúc tiếng Việt | Văn bản | PhoBERT, fine-tuning |
| Nâng cao | Hệ thống gợi ý phim | Tương tác | Collaborative filtering |
| Nâng cao | Triển khai mô hình thành web app | Bất kỳ | FastAPI/Streamlit, Docker |

**Cấu trúc thư mục dự án chuẩn:**

```
du_an_ml/
├── data/
│   ├── raw/              # dữ liệu gốc, không sửa
│   └── processed/        # dữ liệu đã xử lý
├── notebooks/            # EDA, thử nghiệm
├── src/
│   ├── data.py           # nạp và làm sạch
│   ├── features.py       # feature engineering
│   ├── train.py          # huấn luyện
│   └── predict.py        # dự đoán
├── models/               # mô hình đã lưu
├── reports/              # biểu đồ, báo cáo
├── requirements.txt
└── README.md             # mô tả bài toán, cách chạy, kết quả
```

---

## 17. Checklist và lỗi thường gặp

### 17.1. Checklist trước khi huấn luyện

- [ ] Đã xác định rõ label, loại bài toán, metric
- [ ] Đã chia train/test **trước** mọi bước học từ dữ liệu (impute, scale, encode, chọn feature)
- [ ] Đã kiểm tra thiếu, trùng, ngoại lai, kiểu dữ liệu
- [ ] Đã loại cột ID và cột có thể rò rỉ label
- [ ] Đã xử lý mất cân bằng (nếu có)
- [ ] Đã có baseline
- [ ] Dùng `Pipeline` và cố định `random_state`

### 17.2. Checklist sau huấn luyện

- [ ] Đo bằng cross-validation, không chỉ một lần chia
- [ ] So sánh train vs validation (overfit/underfit?)
- [ ] Phân tích lỗi, kiểm tra theo nhóm con
- [ ] Mô hình tốt hơn baseline có ý nghĩa?
- [ ] Test set chỉ dùng một lần cuối
- [ ] Đã lưu mô hình, code, phiên bản thư viện

### 17.3. Lỗi thường gặp

| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| **Data leakage** (scale/impute trên toàn bộ dữ liệu, feature chứa thông tin tương lai) | Điểm cao giả, thất bại khi triển khai | Pipeline, chia trước, kiểm tra feature importance bất thường |
| Chọn mô hình dựa vào test set | Đánh giá lạc quan | Dùng validation/CV |
| Dùng Accuracy với dữ liệu mất cân bằng | Mô hình vô dụng nhưng trông tốt | F1, Recall, PR-AUC |
| Xáo trộn dữ liệu chuỗi thời gian | Nhìn thấy tương lai | `TimeSeriesSplit` |
| Cùng một đối tượng ở cả train và test | Điểm cao giả | `GroupKFold` |
| Không scale với KNN/SVM/nơ-ron | Kết quả kém | `StandardScaler` trong Pipeline |
| One-Hot cho feature hàng nghìn giá trị | Bùng nổ số chiều | Target/Frequency encoding, CatBoost |
| Tinh chỉnh quá nhiều trên validation | Overfit validation | Nested CV, giữ test riêng |
| Bỏ qua EDA | Không hiểu dữ liệu | Luôn EDA trước |
| Không đặt `random_state` | Không tái lập | Cố định seed |
| Mô hình quá phức tạp ngay từ đầu | Khó debug, dễ overfit | Bắt đầu từ baseline |
| Quên giám sát sau triển khai | Mô hình xuống cấp âm thầm | Monitoring, drift detection |

---

## 18. Tài nguyên học tập

### Khóa học
- **Machine Learning Specialization** (Andrew Ng, Coursera): nền tảng tốt nhất cho người mới
- **Deep Learning Specialization** (Andrew Ng)
- **fast.ai** (Practical Deep Learning): học theo hướng thực hành
- **CS229 / CS231n / CS224n** (Stanford): chuyên sâu, miễn phí trên YouTube
- **Kaggle Learn**: các khóa ngắn miễn phí (Pandas, ML, Feature Engineering)

### Sách
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Aurélien Géron)
- *An Introduction to Statistical Learning (ISLR)* (miễn phí)
- *Pattern Recognition and Machine Learning* (Bishop): nâng cao
- *Machine Learning cơ bản* (Vũ Hữu Tiệp, blog `machinelearningcoban.com`): tiếng Việt

### Thực hành
- **Kaggle**: dataset, notebook, cuộc thi
- **Google Colab**: GPU miễn phí
- **Hugging Face**: mô hình và dataset NLP/ảnh/âm thanh
- **GitHub**: đưa dự án lên để xây dựng portfolio

### Tài liệu chính thức
- scikit-learn: <https://scikit-learn.org>
- Pandas: <https://pandas.pydata.org>
- PyTorch: <https://pytorch.org/tutorials>
- XGBoost / LightGBM / CatBoost: tài liệu trên GitHub của từng thư viện

---

> **Lời khuyên cuối:** Mỗi khái niệm hãy theo vòng **Hiểu → Tự code từ đầu (bản nhỏ) → Dùng thư viện → Áp dụng vào dữ liệu thật → Viết lại bằng lời của mình**. Làm 3-5 dự án trọn vẹn có giá trị hơn xem hàng chục khóa học.
