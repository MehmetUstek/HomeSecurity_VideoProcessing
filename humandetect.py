# import the necessary packages
import numpy as np
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time

#PIXEL aralıkları
TEKNE_PIXEL_X1 = 20
TEKNE_PIXEL_X2 = 250
TEKNE_PIXEL_Y1 = 37
# TEKNE_PIXEL_Y2 = 320
TEKNE_PIXEL_Y2 = 400
RESIZE_X = 320
RESIZE_Y = 240
MAIL = "mailname"
PASS = "password"

FRAME_INCLUDED_AREA_X1= 400
FRAME_INCLUDED_AREA_X2= 1200
FRAME_INCLUDED_AREA_Y1= 700
FRAME_INCLUDED_AREA_Y2= 1800

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def sendMail():
    msg = MIMEMultipart()  # create a message

    # add in the actual person name to the message template
    message = "Mail test"

    # setup the parameters of the message
    msg['From'] = MAIL
    msg['To'] = "mailto"
    msg['Subject'] = "This is TEST"

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    # send the message via the server set up earlier.
    s.send_message(msg)

    del msg
#mail
s = smtplib.SMTP(host='smtp.gmail.com', port=587)
s.starttls()
s.login(MAIL, PASS)



cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture('rtsp://usr:pass@ip')
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# frame1 = cv2.resize(frame1, (RESIZE_X, RESIZE_Y))
# frame2 = cv2.resize(frame2, (RESIZE_X, RESIZE_Y))
frame1= frame1[FRAME_INCLUDED_AREA_X1:FRAME_INCLUDED_AREA_X2, FRAME_INCLUDED_AREA_Y1:FRAME_INCLUDED_AREA_Y2]
frame1 = cv2.resize(frame1, (RESIZE_X, RESIZE_Y))
frame2= frame2[FRAME_INCLUDED_AREA_X1:FRAME_INCLUDED_AREA_X2, FRAME_INCLUDED_AREA_Y1:FRAME_INCLUDED_AREA_Y2]
frame2 = cv2.resize(frame2, (RESIZE_X, RESIZE_Y))
# the output will be written to output.avi


counter = 0
temp_now = 0
while (True):
    ret, frame1 = cap.read()
    # frame1 = frame1[FRAME_INCLUDED_AREA_X1:FRAME_INCLUDED_AREA_X2, FRAME_INCLUDED_AREA_Y1:FRAME_INCLUDED_AREA_Y2]
    # frame1 = cv2.resize(frame1, (RESIZE_X, RESIZE_Y))
    frame1 = frame1[FRAME_INCLUDED_AREA_X1:FRAME_INCLUDED_AREA_X2, FRAME_INCLUDED_AREA_Y1:FRAME_INCLUDED_AREA_Y2]
    frame1 = cv2.resize(frame1, (RESIZE_X, RESIZE_Y))
    # Capture frame-by-frame
    # ret, frame = cap.read()
    cv2.rectangle(frame1, (TEKNE_PIXEL_X1, TEKNE_PIXEL_Y1), (TEKNE_PIXEL_X2, TEKNE_PIXEL_Y2),
                  (255, 0, 0), 2)
    diff = cv2.absdiff(frame1, frame2)

    # resizing for faster detection
    # frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    # gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    # for contour in contours:
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #
    #     if cv2.contourArea(contour) < 50000:
    #         continue
    #     cv2.rectangle(frame1,(x,y),(x+w, y+h), (0,255,0), 2)
    #     cv2.putText(frame1, "Status: {}".format('Movement'), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    # boxes, weights = hog.detectMultiScale(frame1, winStride=(4,4),scale=1.03,useMeanshiftGrouping=True)
    #
    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    #
    # for (xA, yA, xB, yB) in boxes:
    #     # display the detected boxes in the colour picture
    #     cv2.rectangle(frame1, (xA, yA), (xB, yB),
    #                   (0, 255, 0), 2)
    #     if TEKNE_PIXEL_X1 < xA and TEKNE_PIXEL_X2 > xB and TEKNE_PIXEL_Y2 > yB and TEKNE_PIXEL_Y1 < yA:
            # print("detected person")
            # sendMail()
            # cv2.rectangle(frame1, (xA, yA), (xB, yB),
            #               (0, 0, 255), 5)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x1, y1, w1, h1) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 10000:
            continue
        counter += 1

        if counter > 100:
            cv2.rectangle(frame1,(x1,y1),(x1+w1, y1+h1), (0,255,0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print("detected person")
            sendMail()
            cv2.rectangle(frame1, (x1, y1), (x1+w1, y1+h1),
                          (0, 0, 255), 5)
            img_name = "opencv_frame_{}.png".format(counter)
            cv2.imwrite(img_name, frame1)
            # cv2.rectangle(frame1, (xA, yA), (xB, yB), (0, 255, 0), 2)
            # cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # counter += 1
            # temp_now = time.perf_counter()
            # if counter > 5:
            #     print("detected person")
            #     sendMail()
            #     cv2.rectangle(frame1, (xA, yA), (xB, yB),
            #                   (0, 0, 255), 5)
            #     img_name = "opencv_frame_{}.png".format(counter)
            #     cv2.imwrite(img_name, frame1)
                # else:
                #     if time.perf_counter() - temp_now > 5:
                #         print(counter)
                #         counter = 0
    if time.perf_counter() - temp_now > 10:
        print(counter)
        temp_now = time.perf_counter()
        counter = 0


    # Display the resulting frame
    cv2.imshow('frame', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    # frame2= frame2[FRAME_INCLUDED_AREA_X1:FRAME_INCLUDED_AREA_X2, FRAME_INCLUDED_AREA_Y1:FRAME_INCLUDED_AREA_Y2]


    frame2 = frame2[FRAME_INCLUDED_AREA_X1:FRAME_INCLUDED_AREA_X2, FRAME_INCLUDED_AREA_Y1:FRAME_INCLUDED_AREA_Y2]
    frame2 = cv2.resize(frame2, (RESIZE_X, RESIZE_Y))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)

