#!/usr/bin/env python3

import cv2
from openvino.inference_engine import IEPlugin
from detectors import FaceDetector, EmotionDetector, GenderDetector
import logging as log
import sys
import socketio

MODEL_PATH = '/app/models/'
VERSION = '16/'
FACE_MODEL = MODEL_PATH + VERSION + 'face-detection-retail-0004.xml'
EMOTION_MODEL = MODEL_PATH + VERSION + 'emotions-recognition-retail-0003.xml'
GENDER_MODEL = MODEL_PATH + VERSION + 'age-gender-recognition-retail-0013.xml'

INPUT_DEVICE = 'cam'
ML_DEVICE = 'MYRIAD'

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    plugin = IEPlugin(device=ML_DEVICE)
    face_detector = FaceDetector(FACE_MODEL, log)
    face_detector.init(plugin)

    emotion_detector = EmotionDetector(EMOTION_MODEL, log)
    emotion_detector.init(plugin)

    gender_detector = GenderDetector(GENDER_MODEL, log)
    gender_detector.init(plugin)

    cap = cv2.VideoCapture(0) # dla pi camery inaczej


    sio = socketio.Client()
    sio.connect('http://localhost:7777')

    while cap.isOpened():
        ret, next_frame = cap.read()

        if not ret:
            break

        initial_w = cap.get(3)
        initial_h = cap.get(4)

        in_frame = face_detector.preprocess_frame(next_frame)
        face_detector.start(request_id = 0, frame = in_frame)

        if face_detector.request(request_id = 0):
            response = face_detector.response(request_id = 0)
            faces = []
            face_id = 0
            for detected_face in response[0][0]:
                if face_detector.can_read(detected_face[2], 0.9):
                    face_id += 1
                    face = face_detector.extract_face(detected_face, next_frame, initial_w, initial_h)

                    in_emotion = emotion_detector.preprocess_frame(face)
                    emotion_detector.start(0, in_emotion)
                    emotion = None
                    if(emotion_detector.request(0)):
                        response = emotion_detector.response(0)
                        emotion = emotion_detector.extract_emotion(response[0])

                    in_gender = gender_detector.preprocess_frame(face)
                    gender_detector.start(0, in_gender)
                    gender = None
                    if(gender_detector.request(0)):
                        response = gender_detector.response(0)
                        gender = gender_detector.extract_gender(response[0])
                    
                    if emotion and gender:
                        faces.append((face_id, gender, emotion))

            if len(faces) > 0:
                sio.emit('ai', faces)
                # log.info(faces)

    cv2.destroyAllWindows()
    del plugin
                


if __name__ == '__main__':
    sys.exit(main() or 0)