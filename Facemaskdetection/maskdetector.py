import os
import cv2
from keras.preprocessing import image
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16 , preprocess_input
from keras import Sequential
from keras.layers import Dense
from keras.models import save_model, load_model


categories = ['with_mask', 'without_mask']
data=[]
for categ in categories:
    path = os.path.join('DataSet',categ)
    label = categories.index(categ)
    for file in os.listdir(path):
        img_path=os.path.join(path,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))

        data.append([img,label])

fraction = 0.25 #change this for better performance on better machine i.e data size to train on.
sampled_indices = np.random.choice(len(data), int(fraction * len(data)), replace=False)
sampled_data = [data[i] for i in sampled_indices]

random.shuffle(sampled_data)

X = np.array([item[0] for item in sampled_data])
Y = np.array([item[1] for item in sampled_data])
X = X / 255
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print("Number of images in X_train:", X_train.shape[0])
print("Number of images in X_test:", X_test.shape[0])

if os.path.exists(model_filename):
    # Load the existing model
    loaded_model = load_model("detectore_model")
else :
    vgg=VGG16()
    model = Sequential()
    for layer in vgg.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable=False
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))
    model.save("detector_model")
    loaded_model=load_model("detector_model")

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()    
    if frame is not None:
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        prediction = loaded_model.predict(frame.reshape(1,224,224,3))
        
        def draw_label(img,text,pos,bg_color):
            text_size= cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
            endx = pos[0]+text_size[0][0]+2
            endy = pos[1]+text_size[0][1]-2
            cv2.rectangle(img,pos,(endx,endy),bg_color,cv2.FILLED)
            cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
        
        if prediction == 0 :
            draw_label(frame,"Masked",(30,30),(0,255,0))
        else:
            draw_label(frame,"Unmasked",(30,30),(255,0,0))
            
        frame = preprocess_input(frame.astype(np.float32))
        cv2.imshow("Window", frame)
    else:
        print("Error: Invalid frame")

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
