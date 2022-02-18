# import libraries here
from __future__ import print_function
#import potrebnih biblioteka
import cv2
import collections
from fuzzywuzzy import fuzz

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
import math
import numpy as np
import matplotlib.pylab as plt
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno

#Funkcionalnost implementirana u V1
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_OTSU)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

#Funkcionalnost implementirana u OCR basic
def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
    
def rotate_image(img,deg):
    rows,cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),deg,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst;



def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    rect_list = []

    avg, cnt, sum = get_average(contours)
    #print('average_area:',avg);
    #print('max_area:',max_area);
    angle=[]
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        area=cv2.contourArea(contour);
        if area<avg/10:
            continue
        
        rect = cv2.minAreaRect(contour)
        rect_list.append(rect[0]) #kordinate centra kontura
        k=rect[0]
        
        #print(k)
        region = image_bin[y:y+h+1,x:x+w+1];
        sum+=area
        cnt+=1
        regions_array.append([region, (x,y,w,h),area,rect[2]]) #TODO resizing
        #cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    
    
    #print(rect_list)
    deg = get_rotation_deg(rect_list)
    for region in regions_array:
        region[0]=rotate_image(region[0],-region[3]);
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    connected_regions=[];
    for i in range(0,len(regions_array)):
        region,cordinates,area,angle=regions_array[i];
        #print(area);
        #print(area);
        
        if i==len(regions_array)-1: #poslednji region ne sadrzi sledeci region
            print('usao sam u uslov za poslednji')
            if area>=avg/2: #ne treba da ga stavljas u povezane regione ako je kvakica
                #print('dodao sam poslednji')
                connected_regions.append([resize_region(region),cordinates]);
                cv2.rectangle(image_orig,(cordinates[0],cordinates[1]),(cordinates[0]+cordinates[2],cordinates[1]+cordinates[3]),(0,255,0),2)
            break;
        next_region,next_cordinates,next_area,angle=regions_array[i+1];
        x1,y1,w1,h1=cordinates;
        x2,y2,w2,h2=next_cordinates;
        #print('kordinate:',cordinates,next_cordinates);
        
        if  x1<x2 and x1<x2<x1+w1 and y2<y1:
            region_chosen=image_bin[y2:y2+h1+h2+1,x1:x1+w1+1];
            cordinates_ch=(x1,y2,w1,h1+h2)
        else:
            if area<avg/2 and avg>1000:
                continue
            if area<avg/4 and avg<1000:
                continue
            region_chosen=image_bin[y1:y1+h1+1,x1:x1+w1+1];
            cordinates_ch=(x1,y1,w1,h1);
        
        connected_regions.append([resize_region(region_chosen),cordinates_ch]);
        cv2.rectangle(image_orig,(cordinates_ch[0],cordinates_ch[1]),(cordinates_ch[0]+cordinates_ch[2],cordinates_ch[1]+cordinates_ch[3]),(0,255,0),2)
        
        
    
    connected_regions = sorted(connected_regions, key=lambda item: item[1][0])
        
    sorted_regions = [region[0] for region in connected_regions]
    
    sorted_regions=[rotate_image(region,-deg) for region in sorted_regions];
    
    sorted_rectangles = [region[1] for region in connected_regions]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances

#metoda uradjena po uzoru na dokumentaciju https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
def get_rotation_deg(rect_list):
    x_values = [];
    y_values = [];
    for center in rect_list:
        x_values.append(center[0]);
        y_values.append(center[1]);
    regr = linear_model.LinearRegression()
    x_values = np.array(x_values);
    y_values = np.array(y_values);
    x_values = np.reshape(x_values, (-1, 1))
    y_values = np.reshape(y_values, (-1, 1));
    plt.scatter(x_values, y_values);
    # Train the model using the training sets
    regr.fit(x_values, y_values)
    # print('intercept',regr.intercept_);
    # print('slope',regr.coef_);
    plt.plot(x_values, regr.predict(x_values), color='k')
    deg = math.atan(-regr.coef_) * (180.0 / math.pi)
    return deg


def get_average(contours):
    avg = 0;
    cnt = 0;
    sum = 0;
    max_area = -1
    for contour in contours:
        area = cv2.contourArea(contour);
        if area > max_area:
            max_area = area;
        sum += area
        cnt += 1
    avg = sum / cnt
    return avg, cnt, sum


def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(500, input_dim=784, activation='sigmoid'))
    ann.add(Dense(128, input_dim=500, activation='sigmoid'))
    ann.add(Dense(60, activation='softmax'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.5, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose=1, shuffle=True)

    return ann


def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialization_folder/neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialization_folder/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann;
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        print('Greska prilikom ucitavanja modela');
        return None

def load_test(img_path):
    image_color = load_image(img_path) #images/a0.png
    img_gr=image_gray(image_color);
    img = image_bin(image_gray(image_color))
    img_binary=image_bin(img_gr);
    inverted_image = cv2.bitwise_not(img_binary)
    inverted_image=erode(inverted_image);
    inverted_image=dilate(inverted_image);
    selected_regions, letters, region_distances = select_roi(image_color.copy(), inverted_image)
    display_image(selected_regions)
    #print ('Broj prepoznatih regiona:', len(letters))
    return letters;

def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran
    alphabet = ['a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž',
                'A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'Ž']
    small_letters_path=train_image_paths[0];
    capital_letters_path=train_image_paths[1];
    small_letters = load_test(small_letters_path);
    capital_letters = load_test(capital_letters_path);
    small_letters.extend(capital_letters);

    inputs = prepare_for_ann(small_letters)
    outputs = convert_output(alphabet)
    model = None
    # probaj da ucitas prethodno istreniran model
    model = load_trained_ann()
    #print(model);

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if model == None:
        print("Traniranje modela zapoceto.")
        model = create_ann()
        model = train_ann(model, inputs, outputs)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(model)

    return model


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """



    alphabet = [
                'A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'Ž',
                'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q',
                'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž',
    ]

    image_color = load_image(image_path)  # images/alphabet0.png
    img_gr = image_gray(image_color);
    ret, image_bin = cv2.threshold(img_gr, 127, 255, cv2.THRESH_OTSU)
    if ret>180:
        binary_image, image_color = load_basic_image(image_path);
    #elif 117<ret<128:
        #binary_image, image_color=load_with_letters(image_path);
    else:
        binary_image, image_color = load_harder_image(image_path);

    selected_regions, letters, distances = select_roi(image_color.copy(), binary_image)
    #display_image(selected_regions)
    letters = [remove_gray_pixels(letter) for letter in letters];
    #print('Broj prepoznatih regiona:', len(letters))

    # Podešavanje centara grupa K-means algoritmom
    distances = np.array(distances).reshape(len(distances), 1)
    # Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente

    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    extracted_text = ""
    if (len(distances)>1):
        k_means.fit(distances)
        inputs = prepare_for_ann(letters)
        results = trained_model.predict(np.array(inputs, np.float32))
        extracted_text = display_result(results, alphabet, k_means)

    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string
    extracted_text=post_processing(extracted_text,vocabulary)

    return extracted_text

def load_basic_image(img_path):
    image_color = load_image(img_path)
    img_gr = image_gray(image_color);
    img = image_bin(image_gray(image_color))
    img_binary = image_bin(img_gr);
    inverted_image = cv2.bitwise_not(img_binary)
    inverted_image = erode(inverted_image);
    inverted_image = dilate(inverted_image);
    # width = int(2000)
    # height = int(300)
    # dim = (width, height)
    # # resize image
    # inverted_image = cv2.resize(inverted_image, dim, interpolation=cv2.INTER_AREA)
    # image_color = cv2.resize(image_color, dim, interpolation=cv2.INTER_AREA)
    return inverted_image,image_color

def load_harder_image(img_path):
    image_color = load_image(img_path)
    low_color = (190)
    high_color = (256)
    img_gr = image_gray(image_color);
    bright_mask = cv2.inRange(img_gr, low_color, high_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bright_mask = cv2.erode(bright_mask, kernel)
    bright_mask = cv2.dilate(bright_mask, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    return bright_mask,image_color

def load_with_letters(img_path):
    image_color = load_image(img_path)
    hsv = cv2.cvtColor(image_color, cv2.COLOR_RGB2HSV)
    image_ada_bin = cv2.adaptiveThreshold(hsv[:, :, 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    bright_mask = image_ada_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bright_mask = cv2.erode(bright_mask, kernel)
    # bright_mask = cv2.dilate(bright_mask, kernel)
    # bright_mask = cv2.erode(bright_mask, kernel2)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    # bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    # bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    # bright_mask = cv2.dilate(bright_mask, kernel)
    bright_mask = cv2.dilate(bright_mask, kernel)
    return bright_mask,image_color

def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result


def post_processing(result,vocabulary):
    processed_result = '';
    #print(type(result));
    words = result.split(' ');
    for word in words:
        if word=='i' or word=='l':
            best_word='I';
        else:
            best_word = find_best_word(word, vocabulary);
        processed_result += best_word;
        processed_result += ' ';
    return processed_result;


def find_best_word(word, vocabulary):
    best_word = [];
    best_word.append(word);
    best_word.append(0);
    for voc in vocabulary:
        ratio = fuzz.ratio(voc, word);
        if ratio > best_word[1]:
            best_word[1] = ratio;
            best_word[0] = voc;
    return best_word[0];

def remove_gray_pixels(image):
    rows = len(image)
    columns = len(image[0])
    for i in range(rows):
        for j in range(columns):
            if image[i][j]>100:
                image[i][j]=255;
            else:
                image[i][j]=0
    return image