import cv2
import time
from win10toast import ToastNotifier

toast = ToastNotifier()


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

face_not_detected_start_time = None
eyes_not_detected_start_time = None
notification_threshold = 5 #seconds

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image,1.1,5, minSize=(40,40))
    eyes_detected = False
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x,y), (x+w, y+h),(0,255,0),4)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = vid[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1,5, minSize=(20,20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0),2)
            eyes_detected = True
    return faces, eyes_detected

while True:
    result, video_frame = video_capture.read()
    if result is False:
        break
    faces, eyes_detected = detect_bounding_box(video_frame)
    current_time = time.time()

    if len(faces) == 0:
        if face_not_detected_start_time is None:
            face_not_detected_start_time = current_time
        elif current_time - face_not_detected_start_time > notification_threshold:
            toast.show_toast(
                "WORK!",
                "GO BACK TO WORK DUDE",
                duration = 20,
                icon_path = "icon.ico",
                threaded = True,
            )
    else:
        face_not_detected_start_time = None

    if not eyes_detected:
        if eyes_not_detected_start_time is None:
            eyes_not_detected_start_time = current_time
        elif current_time - eyes_not_detected_start_time > notification_threshold:
            toast.show_toast(
                "WORK!",
                "GO BACK TO WORK DUDE",
                duration = 20,
                icon_path = "icon.ico",
                threaded = True,
            )
    else:
        eyes_not_detected_start_time = None


    cv2.imshow('go work', video_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()                                                             