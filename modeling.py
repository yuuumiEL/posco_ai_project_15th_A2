import cv2
import numpy as np
import pyaudio
import wave
from imutils import face_utils
import dlib
import argparse
import os.path as osp

from GazeTracking.gaze_tracking import GazeTracking
from HeadposeDetection import headpose


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"
cap = cv2.VideoCapture(0)  ###### 웹캠을 키는 명령어
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
frame_aud = []
###### 저장파트

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

while (True):
    fra = cap.read()[1]  # Read 결과와 frame
    cv2.imshow('frame_color', fra)  # 컬러 화면 출력
    out.write(fra)
    data = stream.read(CHUNK)
    frame_aud.append(data)

    if cv2.waitKey(1) == 27:
        break
stream.stop_stream()
stream.close()
p.terminate()

cap.release()
cv2.destroyAllWindows()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frame_aud))
wf.close()

## 64
p = "HeadposeDetection/models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

## gaze tracking
gaze = GazeTracking()

# cv2.namedWindow('your_face')
# camera = cv2.VideoCapture('output.avi') ####### 저장된 동영상을 불러옴


####headpose

def main(args):
    filename = args["input_file"]

    if filename is None:
        isVideo = False
        cap = cv2.VideoCapture('output.avi')
        cap.set(3, args['wh'][0])
        cap.set(4, args['wh'][1])
    else:
        isVideo = True
        cap = cv2.VideoCapture('output.avi')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name, ext = osp.splitext(filename)
        out = cv2.VideoWriter(args["output_file"], fourcc, fps, (width, height))

    # Initialize head pose detection
    hpd = headpose.HeadposeDetection(args["landmark_type"], args["landmark_predictor"])
    k = open("angles.csv", 'w')

    if cap.isOpened():  ########## 캡쳐 객체 초기화 확인
        while True:
            ret, frame = cap.read()

            #### 진행확인
            print('Frame count:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)), end='\r')
            if np.shape(frame) != ():
                if ret:
                    if isVideo:
                        frame, angles = hpd.process_image(frame)
                        if frame is None:
                            break
                        else:
                            out.write(frame)
                    else:
                        frame = cv2.flip(frame, 1)
                        frame, angles = hpd.process_image(frame)

                    ## 64
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray, 0)
                    for (i, rect) in enumerate(rects):
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # loop over the (x, y)-coordinates for the facial landmarks
                        # and draw them on the image
                        for (x, y) in shape:
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    ###gaze tracking
                    gaze.refresh(frame)
                    frame = gaze.annotated_frame()
                    text = ""

                    if gaze.is_blinking():
                        text = "Blinking"
                    elif gaze.is_right():
                        text = "Looking right"
                    elif gaze.is_left():
                        text = "Looking left"
                    elif gaze.is_center():
                        text = "Looking center"

                    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

                    left_pupil = gaze.pupil_left_coords()
                    right_pupil = gaze.pupil_right_coords()
                    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (147, 58, 31), 1)
                    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                (147, 58, 31), 1)

                    if left_pupil == None or right_pupil == None:
                        pupil_right_x = None
                        pupil_right_y = None
                        pupil_left_x = None
                        pupil_left_y = None
                    else:
                        pupil_right_x = right_pupil[0]
                        pupil_right_y = right_pupil[1]
                        pupil_left_x = left_pupil[0]
                        pupil_left_y = left_pupil[1]
                    k.write('{0}\t{1}\t{2}\t{3}\t{4} \n'.format(angles, pupil_right_x, pupil_right_y, pupil_left_x,
                                                                pupil_left_y))
                    cv2.imshow("Demo", frame)
                    cv2.waitKey(100)  #####프레임을 지연시킴 100ms 이게 제일 중요
                else:
                    break
            else:
                break
    k.close()
    print("end")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', metavar='FILE', dest='input_file', default=None,
                        help='Input video. If not given, web camera will be used.')
    parser.add_argument('-o', metavar='FILE', dest='output_file', default=None, help='Output video.')
    parser.add_argument('-wh', metavar='N', dest='wh', default=[720, 480], nargs=2, help='Frame size.')
    parser.add_argument('-lt', metavar='N', dest='landmark_type', type=int, default=1, help='Landmark type.')
    parser.add_argument('-lp', metavar='FILE', dest='landmark_predictor',
                        default='./model/shape_predictor_68_face_landmarks.dat', help="Landmark predictor data file.")
    args = vars(parser.parse_args())
    main(args)
