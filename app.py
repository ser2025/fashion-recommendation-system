import streamlit as st
import numpy as np
import pickle as pkl
import zipfile
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors

VERI='Sibeller/fashion-data' # Hugging Face veri deposu (dataset)

@st.cache_resource
def yukle():
    zipfile.ZipFile(hf_hub_download(VERI,'images.zip',repo_type='dataset')).extractall('.') # fotoğrafları images klasörüne açar
    filenames=pkl.load(open(hf_hub_download(VERI,'filenames.pkl',repo_type='dataset'),'rb'))
    Image_features=pkl.load(open(hf_hub_download(VERI,'Images_features.pkl',repo_type='dataset'),'rb'))
    neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='cosine')
    neighbors.fit(Image_features)
    model=VGG16(weights='imagenet',include_top=False,input_shape=(64,64,3))
    model.trainable=False
    model=tf.keras.models.Sequential([model,GlobalMaxPool2D()])
    return filenames,Image_features,neighbors,model

filenames,Image_features,neighbors,model=yukle()

def extract_features_from_images(resim, model):
    img=resim.resize((64,64))
    img_array=image.img_to_array(img)
    img_expand_dim=np.expand_dims(img_array,axis=0)
    img_preprocess=preprocess_input(img_expand_dim)
    result=model.predict(img_preprocess,verbose=0).flatten()
    return result/np.linalg.norm(result)

def benzerleri_goster(ozellik):
    distance,indices=neighbors.kneighbors([ozellik])
    st.subheader('Benzer ürünler')
    st.image(np.array(filenames)[indices[0][1:6]].tolist(),width=120)

st.title('Moda Ürünü Öneri Sistemi (VGG16)')
sekme1,sekme2=st.tabs(['Listeden ürün seç','Fotoğraf yükle'])

with sekme1:
    sira=st.slider('Ürün numarası',0,len(filenames)-1,100)
    st.image(filenames[sira],width=150,caption='Seçilen ürün')
    benzerleri_goster(Image_features[sira])

with sekme2:
    dosya=st.file_uploader('Ürün fotoğrafı',type=['jpg','jpeg','png']) or filenames[2000]
    resim=Image.open(dosya).convert('RGB')
    st.image(resim,width=150,caption='Yüklenen ürün')
    benzerleri_goster(extract_features_from_images(resim,model))
