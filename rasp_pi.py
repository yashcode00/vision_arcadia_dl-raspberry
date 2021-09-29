import time,scipy
import cv2
import os
import tensorflow as tf
import numpy as np

dim_x,dim_y=1000,1000

CATEGORIES = ['A', 'B', 'C', 'F', 'G', 'L', 'M', 'O', 'Q', 'V', 'Y', 'nothing']

model = tf.keras.models.load_model('ASLGray_model.h5')

def prepare(filepath):
    image = cv2.imdecode(np.fromfile(
        filepath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = image.reshape(-1, 64, 64, 1)
    image = image.astype('float32')/255.0
    return image

def predict(my_model, filepath):
    prediction = model.predict([prepare(filepath)])
    probs = scipy.special.softmax(prediction[0])
    confidence=str(round((1-max(probs))*100,2))+"%" 
    category = np.argmax(prediction[0])
    print("-> Recognised hand sign: ",CATEGORIES[category]," (",confidence,")")
    return CATEGORIES[category],confidence


def runfunc(prediction):
    
    if prediction=='F':
        #brightness control
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(20,GPIO.OUT,initial=GPIO.HIGH)
        P = GPIO.PWM(20,100)
        P.start(0)
        #increase in brightness
        while 1 :
        
         for i in range(100):
          P.start(i)
          time.sleep(0.5)
        #decrease in brightness  
        while 1 :
     
         for i in range(100):
          P.start(100-i)
          time.sleep(0.5)
    
    elif prediction == "A" or prediction == "M":
        #Blinking an LED
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)
        while 1 :
            GPIO.output(18, GPIO.HIGH) # Turn on
            time.sleep(1) # Sleep for 1 second
            GPIO.output(18, GPIO.LOW) # Turn on
            time.sleep(1) # Sleep for 1 second



dict={'Blinking LED':'na', 'B':"B detected", 'C':"C detected", 'F':"Fade in,Fade out", 'G':"G detected", 'L':"L detected", 'M':"Blinking LED", 'O':"O detected", 'Q':"Q detected", 'V':"V detected", 'Y':"Y  detected", 'nothing':"No motion detected"}


from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO

camera= PiCamera ()
camera. resolution=(dim_x,dim_y)
camera.framerate=4

def collectGestureImages():
    count = 0
    img_counter = 0
    with picamera.PiCamera() as camera:
        while(True):
            rawCapture =PiRGBArray ( camera , size = (dim_x,dim_y))
            camera.capture(rawCapture, format="bgr")
            frame1 = rawCapture.array
            time.sleep(.5)
            camera.capture(rawCapture, format="bgr")
            frame2 = rawCapture.array

            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                if cv2.contourArea(contour) < 9000:
                    continue
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imwrite("frame%d.jpg" % img_counter, frame1)
                category ,confidence= predict(model, "frame%d.jpg" % img_counter)
                out_text="I guess it is "+category+" "+" ("+str(confidence)+")"
                action="Action taken: "+str(dict[category])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame2,out_text,(x,y-40), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame2,action,(x,y-10), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("feed", frame2)


            if cv2.waitKey(40) == 27:
                break

                category=0
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
                os.remove(folderName+"/frame%d.jpg"%img_counter)
                img_counter += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

collectGestureImages()