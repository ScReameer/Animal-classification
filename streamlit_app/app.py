import streamlit as st
import pandas as pd
import numpy as np
import os
# less tensorflow verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_hub as hub
import keras
from PIL import Image, UnidentifiedImageError
import requests
import wget
# display device
if tf.config.list_physical_devices('GPU'):
    print('tensorflow using GPU')
else:
    print('tensorflow using CPU')
# One-hot inverse transform for labels 
INVERSE_TRANSFORM_DICT = {
    0: 'butterfly',
    1: 'cat',
    2: 'chicken',
    3: 'cow',
    4: 'dog',
    5: 'elephant',
    6: 'horse',
    7: 'sheep',
    8: 'spider',
    9: 'squirrel'
}
# Download/load CNN function
@st.cache_resource
def get_model() -> keras.Model:
    """Returns pretrained EfficientNet V2

    Returns:
        keras.Model: model
    """
    folder_name='../model/'
    model_filename = folder_name + 'effnetv2.h5'
    # Download model if not already downloaded
    if not os.path.exists(folder_name): 
        os.mkdir(folder_name)
        url_model = r'https://drive.google.com/uc?id=16vWem3RdeF6ZTt0G7OPI7aVx-2pmIX9q&export=download&confirm=yes'
        wget.download(url_model, out=model_filename)
    # Load and return pretrained model
    model = keras.models.load_model(model_filename, custom_objects=dict(KerasLayer=hub.KerasLayer))
    return model

# Load CNN
model = get_model()
# Function to get label prediction and probability
def get_prediction(model, img_tensor) -> tuple:
    # Transform prediction one-hot matrix to label via argmax
    def inverse_transform(labels:tf.Tensor) -> pd.Series:
        prediction = pd.Series(np.argmax(labels, axis=1)).map(INVERSE_TRANSFORM_DICT)
        return prediction
    # Get one-hot matrix of probabilities
    proba_prediction = model.predict(tf.expand_dims(img_tensor, axis=0), verbose=False)
    # Return most likely label and max probability
    label_prediction = inverse_transform(proba_prediction)[0]
    max_proba_label = proba_prediction.max()
    return label_prediction, max_proba_label
# Title
st.title('Классификация животных по изображению')
st.markdown(
    """Сейчас модель корректно распознает только следующих животных: бабочка, кошка, курица, корова, собака, слон, лошадь, овца, паук, белка"""
)
# URL input
url = st.text_input('Ссылка на изображение', '', key='url')
def clear_btn():
    st.session_state.url = ''
# Clear button
st.button('Очистить поле', key='clear', on_click=clear_btn)
# Submit button
if st.button('Предсказать класс', key='prediction') and url:
    with st.spinner():
        try:
            img = Image.open(requests.get(url, stream=True).raw)
            img_tensor = tf.convert_to_tensor(img)
            label_prediction, max_proba_label = get_prediction(model, img_tensor)
            if max_proba_label > 0.7:
                st.markdown(f'**Предсказанный класс: {label_prediction}, вероятность: {max_proba_label:.3f}**',)
                st.image(img)
            else:
                animals = list(INVERSE_TRANSFORM_DICT.values())
                st.markdown(f'Не удалось распознать животное. Есть возможность распознать только следующий список животных: {animals}')
        except requests.exceptions.MissingSchema:
            st.write('Неверный формат URL')
        except UnidentifiedImageError:
            st.write('Неверный формат изображения')
        except requests.exceptions.InvalidSchema:
            st.write('Неверный формат изображения')
        except requests.exceptions.ConnectTimeout:
            st.write('Не получилось установить связь с сервером')
