from keras.preprocessing import image
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.models import load_model
from fastapi import UploadFile
import io
import numpy as np

efficientNet = EfficientNetB0(include_top=False, weights="imagenet")
classify_model = load_model('./Dish_Recognition_12.h5')

LABELS = ['Banh chung','Banh mi', 'Banh tet', 'Banh trang', 'Banh xeo', 'Bun',
           'Com tam', 'Goi cuon', 'Pho', 'Bun dau mam tom', 'Nem chua', 'Chao long']

def process_image (upload_file: UploadFile):
    content = upload_file.file.read()
    img = image.load_img(io.BytesIO(content), target_size=(224, 224))
    img = image.img_to_array(img)
    feature = np.expand_dims(img, axis=0)
    feature = preprocess_input(feature)
    return feature

def predict_label(upload_file: UploadFile):
    feature = process_image(upload_file)
    feature = efficientNet.predict(feature)
    output = classify_model.predict(feature)

    percent = output.max() * 100
    label_predict = LABELS[np.argmax(output)]

    return percent, label_predict, output.tolist()[0]