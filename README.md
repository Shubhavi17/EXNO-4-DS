# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
 ```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
<img width="408" height="249" alt="image" src="https://github.com/user-attachments/assets/aae6980d-8c93-495e-87e8-e7383a9138a8" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="233" height="263" alt="image" src="https://github.com/user-attachments/assets/f937c125-c0c6-4eaa-a023-2d48d19b0d21" />
```
df.dropna()
```
<img width="533" height="521" alt="image" src="https://github.com/user-attachments/assets/6c167859-ed64-4864-aeab-7d7e3e447b90" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```

<img width="212" height="191" alt="image" src="https://github.com/user-attachments/assets/243d5548-9680-4dde-b11f-a7a2b19d89bb" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
<img width="567" height="268" alt="image" src="https://github.com/user-attachments/assets/da21be96-1c34-422f-b8f9-44dd7ab589ba" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="622" height="456" alt="image" src="https://github.com/user-attachments/assets/9ae2a1c1-1954-43f4-8f14-4fe25723574c" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="505" height="450" alt="image" src="https://github.com/user-attachments/assets/d2749060-f82e-41ce-b8b4-a007e2fe15f0" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="555" height="523" alt="image" src="https://github.com/user-attachments/assets/59bdcb8d-75d9-4525-a015-daba4549d0c3" />
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
<img width="570" height="260" alt="image" src="https://github.com/user-attachments/assets/c71c3e71-f888-4e4a-b383-a84df0409e14" />
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="563" height="445" alt="image" src="https://github.com/user-attachments/assets/d6bbc05e-c52f-4272-8481-9539af608923" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="294" height="598" alt="image" src="https://github.com/user-attachments/assets/3380d813-83ed-4cfc-8918-5389b25da0ee" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="1121" height="505" alt="image" src="https://github.com/user-attachments/assets/a62dbabf-c536-427b-9920-3a6483bcdc8f" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="1054" height="502" alt="image" src="https://github.com/user-attachments/assets/a288ba85-d92b-41f5-9c93-f39a123e1c6c" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="530" height="102" alt="image" src="https://github.com/user-attachments/assets/d9713254-4d72-436e-b15e-d2085b6675e6" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="605" height="448" alt="image" src="https://github.com/user-attachments/assets/3bf17302-3bdf-4947-9e4f-ddc0d2223800" />

```
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="1144" height="504" alt="image" src="https://github.com/user-attachments/assets/d1dba0d4-518a-4e3e-8db1-5a79611ea272" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="1050" height="505" alt="image" src="https://github.com/user-attachments/assets/2cbe6f88-fe8d-42e9-8e77-6bf42aeb62ba" />

```

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="791" height="98" alt="image" src="https://github.com/user-attachments/assets/f505fa86-623a-4305-b1bb-cfcf3045de99" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="709" height="97" alt="image" src="https://github.com/user-attachments/assets/c7cea308-6646-4f7e-b75e-f70e3760b8fb" />
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="707" height="38" alt="image" src="https://github.com/user-attachments/assets/bb2a5a5a-fc8c-407d-8a0f-6f0366d42fd4" />

```
!pip install skfeature-chappers
```
<img width="1424" height="358" alt="image" src="https://github.com/user-attachments/assets/a947dc0d-a43a-4443-b2f6-064042eb8bff" />

```
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
<img width="1092" height="514" alt="image" src="https://github.com/user-attachments/assets/345b020f-f15e-41ce-83cc-67ccb928f765" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_anova = 5
 selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
 X_anova = selector_anova.fit_transform(X, y)
 selected_features_anova = X.columns[selector_anova.get_support()]
 print("\nSelected features using ANOVA:")
 print(selected_features_anova)
```
<img width="992" height="87" alt="image" src="https://github.com/user-attachments/assets/11ae4ff3-fa34-4d0f-a8b2-7671b2598246" />

```
 # Wrapper Method
 import pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 df=pd.read_csv("/content/income(1) (1).csv")
 # List of categorical columns
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 # Convert the categorical columns to category dtype
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="1151" height="513" alt="image" src="https://github.com/user-attachments/assets/e78c9f1b-999a-4fa3-af9c-3f6eb1ff86b0" />

```
X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 logreg = LogisticRegression()
 n_features_to_select =6
 rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
 rfe.fit(X, y)
```
<img width="961" height="694" alt="image" src="https://github.com/user-attachments/assets/04017837-edf3-4432-a854-cd4b06a417a7" />
<img width="251" height="139" alt="image" src="https://github.com/user-attachments/assets/f9af6ea2-1144-41df-b0cd-037e778c1e47" />



# RESULT:
     The given data and perform Feature Scaling and Feature Selection process and save the data to a file is succussfully verified.
