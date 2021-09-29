# vision_arcadia_dl-raspberry
This is a projection aims to detect, classify and interpret hand signs and use them to perform certain tasks using raspberry pi 4. It's model is made using Deep Learning! 

The model uses 5 layer CNN,trained using the American Sign Language(ASL) data on kaggle(https://www.kaggle.com/grassknoted/asl-alphabet) for 12 classes(letters) using 3000 images for each class.

The original model has been saved as "ASLGray_model.h5" file.

For use with raspberry pi 4,it has been converted to "ML_model.tflite".However "ASLGray_model.h5" can also be used.



## About LED Blinking:

Process : 

For blinking an LED we initially make it's voltage high then giving a sleep of about 1 sec and then dropping at voltage to off it and again giving a sleep of 1 sec. This process is repeated till the condition for particular LED is satisfied.

Info : 

There are in total 7 LED which will blink on the basis of that prediction category. It checkes which class is predicted and then runs blinking process for that particular LED.


## About LED Fade in Fade out:

The logic basically uses two while loops in order  to first increase the brightness of LED light 
and then decrease it. 

Initially the bulb is set on using GPIO.setup and GPIO.PWM functions with 0% brightness.

PWM - Pulse width modulation is a method of altering the average power delivered by an electrical signal
thus changing the brightness.

The brightness first increases gradually(sleep of 0.5sec) from 0% to 100% and decreases gradually(sleep of 0.5sec) from 100% to 0% using GPIO.PWM
function which is altered in the two while loops.

Instructions :-
 Make sure that you have Raspberry pi operaring system installed and run the code in it.

1) Connect the LED bulb on the pin number 20 of GPIO board of your raspberry pi hardware.

2) If the catogery predicted is 'F' then the brightness will change as said above.



## Important ##
On linux Operating systems, make sure to remove the whole code and add the following code to 
"~/.local/lib/python3.x/site-packages/pyautogui/_pyautogui_x11.py"
replace x by your version of python, for eg if it is python3.8 then replace x by 8

Copy Code from:
"""
https://pastebin.com/0HcxJHDm