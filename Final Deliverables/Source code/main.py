import cv2
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as np
video = cv2.VideoCapture(0)
model = load_model(
    'E:/ibm project/with flask/IBM-Project-39752-1660497070-main/IBM-Project-39752-1660497070-main/Application Building/Build a Flask Application/realtime.h5')
index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
y = None

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    copy = frame.copy()
    copy = copy[150:150 + 200, 50:50 + 200]
    # Prediction Start
    cv2.imwrite('image.jpg', copy)
    copy_img = image.load_img('image.jpg', target_size=(64,64),color_mode='rgb')
    x = image.img_to_array(copy_img)
    x = np.math.argmax(x, axis=0)
    pred = np.math.argmax(model.predict(x), axis=1)
    y = pred[0]
    cv2.putText(frame, 'The Predicted Alphabet is: ' + str(index[y]), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                3)
    # ret, jpg = cv2.imencode('.jpg', frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
