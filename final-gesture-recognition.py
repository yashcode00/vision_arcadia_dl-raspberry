import time
from sklearn.model_selection import train_test_split
import keras
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pyautogui as pag
import webbrowser as web
import screen_brightness_control as sbc
np.random.seed(5)
# tf.set_random_seed(2)

CATEGORIES = ['A', 'B', 'C', 'F', 'G', 'L', 'M', 'O', 'Q', 'V', 'Y', 'nothing']

model = tf.keras.models.load_model('ASLGray.model')

# prepare image to prediction

def prepare(filepath):
    image = cv2.imdecode(np.fromfile(
        filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = image.reshape(-1, 64, 64, 1)
    image = image.astype('float32')/255.0
    return image

# use this function to predict images


#def predict(my_model, filepath):
#    prediction = model.predict([prepare(filepath)])
#    category = np.argmax(prediction[0])
#    return CATEGORIES[category]


import scipy
#use this function to predict images
def predict(my_model, filepath):
    prediction = model.predict([prepare(filepath)])
    probs = scipy.special.softmax(prediction[0])
    confidence=str(round((1-max(probs))*100,2))+"%" 
    category = np.argmax(prediction[0])
    print("-> Recognised hand sign: ",CATEGORIES[category]," (",confidence,")")
    return CATEGORIES[category]


# for file in os.listdir('test/'):
#   category = predict(model,'test/'+file)
#   print("The image class is: " + str(category))
#   display(Image('test/'+file))

def runfunc(prediction):
    
    if prediction == "B":
        curr = sbc.get_brightness()+10
        print("Your brightness increased to: ",str(curr)+" %")
        sbc.set_brightness(curr, display = 0, force=False)
    elif prediction=='C':
        
        curr = sbc.get_brightness()-10
        print("Your brightness is reduced to: ",str(curr)+" %")
        sbc.set_brightness(curr, display = 0, force=False)
    elif prediction=='O':
        web.open("https://www.google.com")
    elif prediction == "L":
        pag.press("volumeup")
        pag.press("volumeup")
    elif prediction == "Q":
        pag.press("volumedown")
        pag.press("volumedown")
        pag.press("volumedown")
        pag.press("volumedown")
    elif prediction == "M":
        pag.press("volumemute")
    # elif prediction=='F':
    #     pyautogui.hotkey('alt', 'shift', 'esc')



def collectGestureImages():
    '''
    Take a folder name as input from user and create it if not exisits
    Open camera capture each frame and save it in that folder
    '''

    folderName = input("Enter the folder name to save the images: ")
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    time.sleep(1)
    count = 0
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if ret:
            cv2.imshow('frame', frame)
            #print(type(frame))
            if count % 50 == 0:
                cv2.imwrite(folderName+"/frame%d.jpg" % img_counter, frame)
                category = predict(model, folderName +
                                   "/frame%d.jpg" % img_counter)
                # count+=1
                #print("Recognised hand sign: ",category)
                if category == "V":
                    if not os.path.isdir('pics/'):
                        os.mkdir('pics')
                    cv2.imwrite("pics/pic%d.jpg"%img_counter, frame)
                elif category == "Y":
                    path = "screenshots/"
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    pag.screenshot("screenshots/screenshot%d.jpg"%img_counter)
                else:   
                    runfunc(category)
                #3.print("Current File %d \r" % img_counter, end='')
                os.remove(folderName+"/frame%d.jpg"%img_counter)
                img_counter += 1
                count = 0
            count += 1
        else:
            break
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # time.sleep(1)

    cam.release()
    cv2.destroyAllWindows()
    print("Created folder: " + folderName)


collectGestureImages()
