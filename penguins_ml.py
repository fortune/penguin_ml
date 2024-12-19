import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

penguin_df = pd.read_csv("penguins.csv")

#   species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex  year
# 0  Adelie  Torgersen            39.1           18.7              181.0       3750.0    male  2007
# 1  Adelie  Torgersen            39.5           17.4              186.0       3800.0  female  2007
# 2  Adelie  Torgersen            40.3           18.0              195.0       3250.0  female  2007
# 3  Adelie  Torgersen             NaN            NaN                NaN          NaN     NaN  2007
# 4  Adelie  Torgersen            36.7           19.3              193.0       3450.0  female  2007
print(penguin_df.head())

# NULL 値を含む行を削除
penguin_df.dropna(inplace=True)

# 出力（ペンギンの種類）と特徴量に分割する。
output = penguin_df["species"]
features = penguin_df[
    [
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]
]

# 特徴量を one-hot-encoding して文字列を数値にし、カテゴライズする。
features = pd.get_dummies(features)


# Here are our output variables
# 0    Adelie
# 1    Adelie
# 2    Adelie
# 4    Adelie
# 5    Adelie
# Name: species, dtype: object
print("Here are our output variables")
print(output.head())

# Here are our feature variables
#    bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g  island_Biscoe  island_Dream  island_Torgersen  sex_female  sex_male
# 0            39.1           18.7              181.0       3750.0          False         False              True       False      True
# 1            39.5           17.4              186.0       3800.0          False         False              True        True     False
# 2            40.3           18.0              195.0       3250.0          False         False              True        True     False
# 4            36.7           19.3              193.0       3450.0          False         False              True        True     False
# 5            39.3           20.6              190.0       3650.0          False         False              True       False      True
print("Here are our feature variables")
print(features.head())

# output 中の値を 0, 1, 2, ... に変換する。
# uniques は 0, 1, 2, ... に変換された元の値（ペンギンの種類名）が入っている。
output, uniques = pd.factorize(output)


x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)

rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_pred, y_test)
print("Our accuracy score for this model is {}".format(score))

# モデルと種類名が入ったファイルを Pickle 形式で保存
rf_pickle = open("random_forest_penguin.pickle", "wb")
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open("output_penguin.pickle", "wb")
pickle.dump(uniques, output_pickle)
output_pickle.close()

# モデル解釈に役立つ特徴量重要度のグラフを作成し、保存しておく。
# これも Streamlit アプリで読み出し、表示する。
fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title("Which features are the most important for species prediction?")
plt.xlabel("Importance")
plt.ylabel("Feature")

# tight_layout() を実行することで、グラフのフォーマットをより良くし、ラベルが途切れてしまうのを
# 防ぐことができる。
plt.tight_layout()
fig.savefig("feature_importance.png")
