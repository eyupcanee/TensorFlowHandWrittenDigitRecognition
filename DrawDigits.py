import tensorflow as tf
import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

img = np.zeros((512,512,3),np.uint8)

drawing = False
ix,iy = -1,-1

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),3,(255,255,255),30)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow("image",600,600)
cv2.setMouseCallback("image",draw_circle)

while(1):
    cv2.imshow("image",img)
    if cv2.waitKey(20) & 0xFF == 13:
        break

img = cv2.resize(img,(28,28))
img = [[[i[0]*255*255+i[1]*255+i[2]]for i in part] for part in img]
img = np.array(img)
print(img.shape)
img = np.divide(img,np.max(img))
print(img.max())
img = np.around(img,decimals=2)
resim = np.multiply(img,255)
resim = resim.reshape(28,28)

img = np.expand_dims(img,axis=0)
print(img.shape)

model = tf.keras.models.load_model("./model/hwrModel.h5")
model.load_weights("./model/hwrModelWeights.h5")
res = []

try:
    res = model.predict(x = img)
except Exception as e:
    print(e)

labels = [0,1,2,3,4,5,6,7,8,9]
res = np.around(res,decimals=2)
print(res)
#print(np.argmax(res))
print("This Number Might Be : "+ str(labels[np.argmax(res)]))

cv2.destroyAllWindows()
