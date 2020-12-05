import cv2

capture = cv2.VideoCapture(0)
i = 0
while (capture.isOpened()):
    ret, frame = capture.read()
    if ret == False:
        break
    #if i % 50 == 0:
    cv2.imwrite('img' + str(i) + '.jpg', frame)
    i += 1

capture.release()
cv2.destroyAllWindows()