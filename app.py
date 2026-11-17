import os
import streamlit as st
import numpy as np
import pickle as pkl
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors

VERI='Sibeller/fashion-data' # Hugging Face veri deposu (dataset)

@st.cache_resource
def yukle():
    klasor=snapshot_download(VERI,repo_type='dataset',allow_patterns=['images/*']) # images klasöründeki fotoğrafları indirir
    filenames=pkl.load(open(hf_hub_download(VERI,'filenames.pkl',repo_type='dataset'),'rb'))
    filenames=[os.path.join(klasor,'images',os.path.basename(str(f).replace('\\','/'))) for f in filenames] # dosya yollarını indirilen klasöre göre düzeltir
    Image_features=pkl.load(open(hf_hub_download(VERI,'Images_features.pkl',repo_type='dataset'),'rb'))
    var=[i for i,f in enumerate(filenames) if os.path.exists(f)] # sadece Hugging Face'te bulunan fotoğrafları kullanır
    filenames=[filenames[i] for i in var]
    Image_features=np.array(Image_features)[var]
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
