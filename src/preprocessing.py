import pandas as pd
import numpy as np
from pathlib import Path


path = Path("../data/raw/Bank_Churn_Classification_Dataset.csv")
df = pd.read_csv(path)

df.drop(["Unnamed: 0", "CustomerID"], axis=1, inplace=True)
categorical = ["Gender", "Contract", "PaymentMethod"]

df_new = df.copy()
df_new = pd.get_dummies(df_new, columns=categorical, drop_first=True)

data_dir = Path("../data") / "processed"
data_dir.mkdir(exist_ok=True)

df_new.to_csv(data_dir/'data.csv', index=False)