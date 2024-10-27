import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.title("Penguin Classifier")
st.write(
    "This app uses 6 inputs to predict the species of penguin using"
    "a model built on the Palmer Penguins dataset. Use the form below"
    " to get started!"
)

# penguins_ml.py で作成したモデルファイルとマッピングファイルをロードする。
rf_pickle = open("random_forest_penguin.pickle", "rb")
map_pickle = open("output_penguin.pickle", "rb")

rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)

# 確認用のデバッグ出力
st.write(rfc)
st.write(unique_penguin_mapping)

rf_pickle.close()
map_pickle.close()


# ユーザに特徴量を入力あるいは選択してもらう
island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
sex = st.selectbox("Sex", options=["Female", "Male"])
bill_length = st.number_input("Bill Length (mm)", min_value=0)
bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
body_mass = st.number_input("Body Mass (g)", min_value=0)

# 確認用のデバッグ出力
user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]
st.write(f"""the user inputs are {user_inputs}""".format())

# 入力値をモデルが想定している形式に変換する
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

# 入力された特徴量に対するペンギンの種類をモデルで予測する
new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male,
        ]
    ]
)
prediction_species = unique_penguin_mapping[new_prediction][0]

st.subheader("Predicting Your Penguin's Species:")
st.write(f"We predict your penguin is of the {prediction_species} species")

# 保存済みの特徴量重要度のグラフのファイルを表示する。
st.write(
    """We used a machine learning (Random Forest)
    model to predict the species, the features
    used in this prediction are ranked by
    relative importance below."""
)
st.image("feature_importance.png")


# ペンギンの種類ごとに特徴量の分布を表示するためのセクションを開始する。
st.write(
    """Below are the histograms for each
    continuous variable separated by penguin
    species. The vertical line represents
    your the inputted value."""
)

# 特徴量の分布を表示するために元のデータセットを読み出さないといけない。
penguin_df = pd.read_csv("penguins.csv")
print(penguin_df.head())

# NULL 値を含む行を削除
penguin_df.dropna(inplace=True)

# bill_length_mm 特徴量の分布をペンギンの種類ごとに表示する。
# ユーザが推論のために入力した bill_length 値も分布図上にラインとして表示する。
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_length_mm"], hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Bill Length by Species")
st.pyplot(ax)

# bill_depth_mm について同じことをおこなう。
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"], hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)

# flipper_length_mm について同じことをおこなう。
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["flipper_length_mm"], hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)
