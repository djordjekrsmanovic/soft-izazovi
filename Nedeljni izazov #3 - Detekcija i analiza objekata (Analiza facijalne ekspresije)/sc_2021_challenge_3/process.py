# import libraries here
import math
from math import sqrt, atan

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def form_hog(img):
    nbins = 9  # broj binova
    cell_size = (30, 30)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog;

def getLabelMark(label):
    #neutral=0
    #contempt=1
    #anger=2
    #disgust=3
    #happiness=4
    #surprise=5
    #sadness=6
    if label=='neutral':
        return 0
    elif label=='contempt':
        return 1
    elif label=='anger':
        return 2
    elif label=='disgust':
        return 3
    elif label=='happiness':
        return 4
    elif label=='surprise':
        return 5
    else:
        return 6


def formArray(distance_list):
    array=[];
    for element in distance_list:
        array.append(np.array(element));
    return array;

def extract_features(images,labels,face_points):

    x=[];
    y=[];
    for img,label,person_points in zip(images,labels,face_points):
        #img=load_image(path);
        hog=form_hog(img);
        hog_value=hog.compute(img);
        distance_list=calculate_distances(person_points)
        array=np.array(distance_list);
        array=array.reshape(11,1);
        hog_value=np.concatenate((hog_value,array),axis=0);
        x.append(hog_value);
        y.append(getLabelMark(label));


    x=np.array(x);
    x=reshape_data(x);
    print(x.shape);
    y=np.array(y)
    print(y.shape);
    return x,y

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))



def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje i listu labela za svaku fotografiju iz prethodne liste

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno istreniran. 
    Ako serijalizujete model, serijalizujte ga odmah pored main.py, bez kreiranja dodatnih foldera.
    Napomena: Platforma ne vrsi download serijalizovanih modela i bilo kakvih foldera i sve ce se na njoj ponovo trenirati (vodite racuna o vremenu). 
    Serijalizaciju mozete raditi samo da ubrzate razvoj lokalno, da se model ne trenira svaki put.

    Vreme izvrsavanja celog resenja je ograniceno na maksimalno 1h.

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """

    # TODO - Istrenirati model ako vec nije istreniran
    face_images=[];
    face_points=[]
    detector = dlib.get_frontal_face_detector()
    i=0;
    for image_path in train_image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects=detector(gray,1);

        faces,points=get_faces(rects,gray);
        face_images.append(faces);
        face_points.append(points);
        print('slika',i);
        i+=1;
    model = None
    print('Treniranje modela zapoceto');
    x_train,y_train=extract_features(face_images,train_image_labels,face_points); # features from hog
    model = SVC(kernel='linear', probability=True)
    model.fit(x_train, y_train)
    print('Treniranje modela zavrseno');
    return model

def get_euclidean(x1,y1,x2,y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

# Za ovo je koristen naucni radi koji gleda na koji nacin razdaljina izmedju odredjenih tacaka lica utice na emocije
# medjutim cini mi se da ne donosi neka poboljsanja
# mozda je problem sto nisam dovoljno razdaljina koristio
# featuri koje dobijemo na ovaj nacin se konkateniraju na feature vektor koji se dobije od hog deskriptora
def calculate_distances(face_points):
    #tacke ociju
    print(face_points[5]);
    t38=face_points[37];
    t42=face_points[41];
    t45=face_points[44];
    t47=face_points[46];
    t37=face_points[36];
    t40=face_points[39];
    t43=face_points[42];
    t46=face_points[45];
    t39=face_points[38];
    t44=face_points[43];

    #tacke nosa
    t32=face_points[31];
    t36=face_points[35];

    #tacke usta
    t49=face_points[48];
    t55=face_points[54];
    t52=face_points[51];
    t58=face_points[57];
    t63=face_points[62];
    t67=face_points[66];

    #tacke obrva lijeva
    t18=face_points[17];
    t19=face_points[18];
    t20=face_points[19];
    t21=face_points[20];
    t22=face_points[21];

    #tacke obrva desna
    t23 = face_points[22];
    t24 = face_points[23];
    t25 = face_points[24];
    t26 = face_points[25];
    t27 = face_points[26];

    #tacke za sirinu glave
    t1=face_points[0];
    t17=face_points[16];

    #tacke za duzinu
    t9=face_points[8]; # druga je vec definisana kod obrva

    #distance
    d1=get_euclidean(t38[0],t38[1],t42[0],t42[1])/get_euclidean(t9[0],t9[1],t23[0],t23[1]);
    d2 = get_euclidean(t45[0], t45[1], t47[0], t47[1]) / get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d3= get_euclidean(t38[0], t38[1], t20[0], t20[1]) / get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d4=get_euclidean(t45[0], t45[1], t25[0], t25[1]) / get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d5=get_euclidean(t52[0], t52[1], t58[0], t58[1]) / get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d6 = get_euclidean(t63[0], t63[1], t67[0], t67[1]) / get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d7=get_euclidean(t49[0], t49[1], t55[0], t55[1]) / get_euclidean(t1[0], t1[1], t17[0], t17[1]);
    d8=get_euclidean(t25[0],t25[1],t23[0],t23[1])/get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d9=get_euclidean(t63[0], t63[1], t67[0], t67[1]) / get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d10=get_euclidean(t22[0], t22[1], t39[0], t39[1]) / get_euclidean(t9[0], t9[1], t23[0], t23[1]);
    d11=get_euclidean(t23[0],t23[1],t43[0],t43[1])/get_euclidean(t9[0], t9[1], t23[0], t23[1]);

    distance_list=[];
    distance_list.append(d1);
    distance_list.append(d2);
    distance_list.append(d3);
    distance_list.append(d4);
    distance_list.append(d5);
    distance_list.append(d6);
    distance_list.append(d7);
    distance_list.append(d8);
    distance_list.append(d9);
    distance_list.append(d10);
    distance_list.append(d11);

    return distance_list;

def rotate_image(img,deg):
    rows,cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),deg,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst;

def get_faces(rects,image):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face=image[y:y+h+1,x:x+w+1]
        print('lice obradjeno');
        width = 250
        height = 300
        dim = (width, height)

        # # resize image
        face = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)
        shape = predictor(face, rect)
        shape = face_utils.shape_to_np(shape)
        t45 = shape[44]; #tacka na desnom oku
        t39 = shape[38]; #tacka na lijevom oku

        degree = math.degrees(atan((t39[1] - t45[1]) / (t39[0] - t45[0])));
        face=rotate_image(face,degree);
        print(degree)

        return face,shape # jedno lice se nalazi na slikama

def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """
    facial_expression = ""
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)
    img=load_image(image_path)
    detector = dlib.get_frontal_face_detector()
    rects=detector(img,1);
    face,points=get_faces(rects,img);
    hog=form_hog(face)
    x=hog.compute(face)
    distance_list = calculate_distances(points);
    array = np.array(distance_list);
    array = array.reshape(11, 1);
    x = np.array(x)
    x = np.concatenate((x, array), axis=0);
    nx,ny=x.shape;
    x=x.reshape((1, nx*ny))
    value=trained_model.predict(x);
    facial_expression=get_string_from_numeric(value);
    return facial_expression


def get_string_from_numeric(value):
    if value==0:
        return 'neutral'
    elif value==1:
        return 'contempt'
    elif value==2:
        return 'anger'
    elif value==3:
        return 'disgust'
    elif value==4:
        return 'happiness'
    elif value==5:
        return 'surprise'
    elif value==6:
        return 'sadness'
    else:
        return '';