import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def Check_Dataframe(dataframe):
   print("##########COMPLETE INFO#######################")
   print("Head\n", dataframe.head())
   print("\nInfo\n")
   dataframe.info()
   print("\nDescribe\n", dataframe.describe(include='all'))
   print("\nIsNull\n", dataframe[dataframe.isnull().any(axis = 1)])
   print("\nColumns\n", dataframe.columns)
   print("#################################")


# 1. Load data
df = pd.read_csv("profiles.csv")
Check_Dataframe(df)

# 2. Select features and target
target = "orientation"
categorical_features = ["diet", "drinks", "drugs", "education", "religion", "sex", "smokes", "status"]
numeric_features = ["height", "income"]
text_features = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Drop rows with missing target
df = df.dropna()

# Combine all essays into one text column
df["all_essays"] = df[text_features].fillna("").apply(lambda x: " ".join(x), axis=1)

X = df[categorical_features + numeric_features + ["all_essays"]]
y = df[target]

# 3. Preprocessing
# One-hot encode categorical
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Scale numeric
numeric_transformer = StandardScaler()

# TF-IDF for essays
text_transformer = TfidfVectorizer(max_features=5000, stop_words="english")

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numeric_transformer, numeric_features),
        ("txt", text_transformer, "all_essays")
    ]
)

# 4. Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Train
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
