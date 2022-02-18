import numpy as np
import cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt


def count_cars(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj prebrojanih automobila. Koristiti ovu putanju koja vec dolazi
    kroz argument procedure i ne hardkodirati nove putanje u kodu.

    Ova procedura se poziva automatski iz main procedure i taj deo koda nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih automobila
    """
    car_count = 0
    # TODO - Prebrojati auta i vratiti njihov broj kao povratnu vrednost ove procedure
    img_rgb,size_flag=load_image(image_path);
    
    if size_flag==True:
        car_count=process_small(image_path)
    else:
        car_count=process_big(image_path)
    
    return car_count

def load_image(image_path):
    matplotlib.rcParams['figure.figsize'] = 16,12

    img_car = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_height = img_car.shape[0]
    image_width = img_car.shape[1]
    process_small=False
    if image_height<350 or image_width<250:
        process_small=True;
    width = int(500)
    height = int(250)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img_car, dim, interpolation = cv2.INTER_AREA)
    return resized,process_small

def process_bright(image_path):
    img_rgb,flag=load_image(image_path)
    image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.erode(image, kernel, iterations=2)
    # plt.imshow(image)
    # plt.imshow(image, 'gray')
    print(np.mean(image))
    low_color = (180)
    high_color = (256)
    bright_mask = cv2.inRange(image, low_color, high_color)
    bright_mask = cv2.erode(bright_mask, kernel)
    return bright_mask

def process_dark(image_path):
    img_rgb,flag = load_image(image_path);
    image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY);
    plt.imshow(image, 'gray');
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    image = cv2.erode(image, kernel, iterations=3);
    plt.imshow(image, 'gray');
    low_color = (30);
    high_color = (80);
    dark_mask = cv2.inRange(image, low_color, high_color);
    # plt.imshow(dark_mask, 'gray');
    return dark_mask

def process_big(image_path):
    bright_image = process_bright(image_path);
    dark_image=process_dark(image_path);
    first_image=bright_image+dark_image;
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    first_image = cv2.erode(first_image, kernel, iterations=2)
    first_image = cv2.dilate(first_image, kernel2, iterations=2)

    image, contours, hierarchies = cv2.findContours(first_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_sb = []


    k = hierarchies[0][3]

    i = 0;

    for contour in contours:
        k = hierarchies[0][i]
        if cv2.contourArea(contour) < 600:
            i += 1;
            continue;
        if k[3] < 0:
            contours_sb.append(contour);
        i += 1;


    print('Number of cars', len(contours_sb))
    # image_sb_contours = img_rgb.copy();
    # cv2.drawContours(image_sb_contours, contours_sb, -1, (255, 0, 0), 1)
    # plt.imshow(image_sb_contours);
    car_count=len(contours_sb)
    return car_count

def process_small(image_path):
    car_count=0;
    img,flag=load_image(image_path);
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.erode(image, kernel, iterations=2)
    image_ada_bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -9)
    image_ada_bin = cv2.dilate(image_ada_bin, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    image_ada_bin = cv2.erode(image_ada_bin, kernel2)
    plt.figure()
    plt.imshow(image_ada_bin, 'gray')
    image, contours, hierarchies = cv2.findContours(image_ada_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_sb = []

    k = hierarchies[0][3]
    i = 0;

    for contour in contours:
        k = hierarchies[0][i]
        if cv2.contourArea(contour) < 130:
            i += 1;
            continue;
        if k[3] < 0:
            contours_sb.append(contour);
        i += 1;

    print('Number of cars', len(contours_sb))
    image_sb_contours = img.copy();
    cv2.drawContours(image_sb_contours, contours_sb, -1, (255, 0, 0), 1)
    plt.imshow(image_sb_contours);
    car_count = len(contours_sb)
    return car_count
