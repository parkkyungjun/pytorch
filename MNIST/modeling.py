import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import load_model
import time

oldx = oldy = -1  # 좌표 기본값 설정
k = 0
# 내가 만든 마우스 함수를 실행시켜줘

model = load_model('model.h5')


def on_mouse(event, x, y, flags, param):

    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이타, 안쓰더라도 넣어줘야함

    global oldx, oldy,k  # 밖에 있는 oldx, oldy 불러옴
    if k == 1:
        cv2.rectangle(img, (0, 0), (280, 280), (255, 255, 255), -1)
        cv2.imshow('image', img)
        k = 0

    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽이 눌러지면 실행
        oldx, oldy = x, y  # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
     #   print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))  # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 뗏을때 발생
     #   print('EVENT_LBUTTONUP: %d, %d' % (x, y))  # 좌표 출력
        cv2.imwrite('test3.jpg',img)
        img_color2 = cv2.imread('test3.jpg')
        img_color = cv2.imread('test3.jpg', cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img_digit = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        img_digit = cv2.resize(img_digit, (28, 28), interpolation=cv2.INTER_AREA)

        img_digit = img_digit / 255.0

        img_input = img_digit.reshape(1, 28, 28, 1)
        predictions = model.predict(img_input)
        if np.max(predictions) > 0.8:
            k = 1
            number = np.argmax(predictions)
            print(number,np.max(predictions))
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = 0.5
            cv2.putText(img_color2, 'The number you wrote is ' + str(number), (22,250), font, fontScale, (0, 255, 0), 1)
            cv2.imshow('image', img_color2)
            time.sleep(5)
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스가 움직일 때 발생
        if flags & cv2.EVENT_FLAG_LBUTTON:  # ==를 쓰면 다른 키도 입력되었을 때 작동안하므로 &(and) 사용
            # cv2.circle(img, (x, y), 5, (0, 255, 0), -1) # 단점이 빠르게 움직이면 끊김
            # circle은 끊기므로 line 이용
            # 마우스 클릭한 좌표에서 시작해서 마우스 좌표까지 그림
            cv2.line(img, (oldx, oldy), (x, y), (0, 0, 0), 10, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y  # 그림을 그리고 또 좌표 저장

model.summary()
# 흰색 컬러 영상 생성
img = np.ones((280, 280, 3), dtype=np.uint8) * 255

# 윈도우 창
cv2.namedWindow('image')

# 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
# 마우스 이벤트가 발생하면 on_mouse 함수 실행
cv2.setMouseCallback('image', on_mouse, img)

# 영상 출력
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()