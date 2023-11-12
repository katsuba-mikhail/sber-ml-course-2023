import pandas as pd
import streamlit as st
import os

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from IPython.display import Image
from PIL import Image


# загрузка и фильтрация данных
script_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_directory, "train.csv")
data = pd.read_csv(data_path)
data = data[['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
               'BsmtFinSF1', '2ndFlrSF', 'LotArea', 'GarageArea', 'SalePrice']]

# обучение модели
X = data.drop(columns='SalePrice')
y = data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = CatBoostRegressor(iterations=500, depth=10, learning_rate=0.05, loss_function='RMSE', random_seed=42)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=100)

# словарь соответствий фичей
feature_translation = {
    'OverallQual': 'Общее качество дома',
    'GrLivArea': 'Жилая площадь над землей',
    'TotalBsmtSF': 'Общая площадь подвала',
    'GarageCars': 'Вместимость гаража',
    'BsmtFinSF1': 'Площадь готового подвала',
    '2ndFlrSF': 'Площадь второго этажа',
    'LotArea': 'Площадь участка',
    'GarageArea': 'Площадь гаража'
}

# заголовок
st.markdown("""
    <h1 style='color: #0000ff; text-align: center; padding: 20px; font-size: 36px; background-color: #ADD8E6;'>Прогнозирование цены на дом</h1>
""", unsafe_allow_html=True)

# фон
background_image_path = os.path.join(script_directory, "123.jpg")
background_image = Image.open(background_image_path)
st.image(background_image, caption='', use_column_width=True)

# подзаголовок
st.markdown("<h3 style='text-align: center; color: white;'>Ввод значений признаков</h3>", unsafe_allow_html=True)

# ввод значений признаков
feature_values = {}
for feature in X.columns:
    if X[feature].dtype in ['int64', 'float64']:
        translated_feature = feature_translation.get(feature, feature)
        value = st.number_input(f'{translated_feature}', value=0, key=f'{feature}_input')
        feature_values[feature] = value

# прогноз цены
prediction = 0

if feature_values:
    input_data = pd.DataFrame([feature_values])
    prediction = model.predict(input_data)
    st.success(f'Предсказанная цена: ${prediction[0]:,.2f}')
elif not feature_values:
    st.warning('Введите значения  признака')
