# Özge Çinko
###################################
# Veri Seti Hikayesi
###################################
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve
# İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.
###################################
# GÖREV 1 : Keşifçi Veri Analizi
###################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

df_ = pd.read_csv("datasets/Telco-Customer-Churn.csv")
df = df_.copy()

def analyze_data(df):
    """
    Prints detailed information about a pandas DataFrame.

    Parameters:
    df (pandas DataFrame): The DataFrame to analyze.
    """
    print("#" * 30)
    print("Data Shape")
    print("-" * 30)
    print(df.shape)
    print("#" * 30)
    print("Data Columns")
    print("-" * 30)
    print(df.columns)
    print("#" * 30)
    print("Data Types")
    print("-" * 30)
    print(df.dtypes)
    print("#" * 30)
    print("Data Missing Values")
    print("-" * 30)
    print(df.isnull().sum())
    print("#" * 30)
    print("Data Summary Statistics")
    print("-" * 30)
    print(df.describe().T)
    print("#" * 30)


analyze_data(df)


# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.

# Veri seti içindeki kategorik ve numerik değişkenleri bize ayrı ayrı getiren bir çalışma yapalım.
# cat_th=10 => eşsiz değer sayısı 10'dan küçükse kategorik değişken.
# car_th=30 => eşsiz değer sayısı 30'dan büyükse kardinal değişken.
# Fonksiyona docstring tanımlaması yapalım ki başkaları da kullanabilsin.
# """ + enter ile docstring yapılır.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeridir.
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeridir.

    Returns
    -------
    cat_cols: list
        kategorik değişken listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    -----
    cat_cols + num_cols + cat_but_car = toplam değişken sayısıdır.
    num_but_cat, cat_cols'un içerisindedir.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car



# Bool değerleri int'e çevirdik.
# Fonksiyon dışında bunu yapmak daha mantıklı oldu.
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)

# Boş string'ler varsa NaN ile değiştirilir.
df.replace(" ", np.nan, inplace=True) # TotalCharges 11 adet boş değer içeriyormuş.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)

# Hedef değişkeni sınıflandırma probleminde kullanabilmek için 1-0 olarak değiştirilir.
df["Churn"] = [1 if i == "Yes" else 0 for i in df["Churn"]]
df["Churn"] = df["Churn"].astype(int)

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# Kategorik değişkenlerin hepsini görüntüleyelim.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)


# Numerik değişkenlerin hepsini görüntüleyelim.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
target = "Churn"
# Tüm kategorik değişkenlerle hedef değişkeni incelemek için fonksiyon yazalım.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(dataframe.groupby(categorical_col)[target].mean(), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, target, col)


# Tüm numerik değişkenlerle hedef değişkeni incelemek için fonksiyon yazalım.
def target_summary_with_num(dataframe, target, numerical_col):
    result = dataframe.groupby(target)[numerical_col].mean().reset_index()
    result.columns = [target, numerical_col]
    print(result, end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, target, col)


# Adım 5: Aykırı gözlem var mı inceleyiniz.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 6: Eksik gözlem var mı inceleyiniz.

# Eksik gözlem var mı yok mu sorgusu
df.isnull().values.any() # Var

# Değişkenlerdeki eksik değer sayısı
df.isnull().sum() # TotalCharges 11 adet eksik değere sahip

# Değişkenlerdeki tam değer sayısı
df.notnull().sum()

# Veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum() # 11

# En az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# Tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

###################################
# GÖREV 2 : Feature Engineering
###################################
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# Aykırı gözlem yok.
# Eksik değerler TotalCharges içerisinde 11 adet var.
# Ortalamayla doldurdum.
df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
df.isnull().sum()

# Adım 2: Yeni değişkenler oluşturunuz.
df.corr().sort_values(by="Churn", ascending=False)
df["MonthlyChargesCategory"] = pd.qcut(df["MonthlyCharges"], 3, labels=["Low", "Medium", "High"])
df["tenureCategory"] = pd.cut(df["tenure"], bins=[0, 24, 48, np.inf], labels=["New", "Intermediate", "Long-term"])
df["TotalChargesCategory"] = pd.qcut(df["TotalCharges"], q=3,  labels=["Low", "Medium", "High"])


# Adım 3: Encoding işlemlerini gerçekleştiriniz.
le = LabelEncoder()

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()


###################################
# GÖREV 3 : Modelleme
###################################
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


# RANDOM FORESTS
###########################
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# GBM
################################################
gbm_model = GradientBoostingClassifier(random_state=17)
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# XGBOOST
###########################
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# CATBOOST
###########################
catboost_model = CatBoostClassifier(random_state=17, verbose=False)
cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.
# Yukarıdakilerden en iyi Random Forests ve GBM sonuç verdiği için bu ikisi üzerinde hiperparametre optimizasyonu gerçekleştirmeye karar verdim.
# RANDOM FORESTS
###########################
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15],
             "n_estimators": [100, 200, 400]}


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# GBM
################################################
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_ # En iyi parametreleri gösterir.
# En iyi parametreleri set ederiz ve modeli fit ederiz.
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
