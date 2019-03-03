import sys
import os
import cv2
from openvino.inference_engine import IENetwork

class Detector:
    model_path = ''
    weight_path = ''

    net = None
    net_inputs = None
    net_outputs = None
    
    engine = None
    logger = None
    
    params = {}

    def __init__(self, model_path, logger):
        self.logger = logger
        self.model_path = model_path
        self.weight_path = self.weights(self.model_path)

    def init(self, plugin):
        # utwórz sieć na podstawie modelu
        net = IENetwork(model=self.model_path, weights=self.weight_path)
        # pobierz informacje o wejściach i wyjściach z sieci
        self.net_inputs = self.inputs(net.inputs)
        self.net_outputs = self.outputs(net.outputs)
        self.engine = plugin.load(network=net, num_requests=1)
        # przygotuj informacje do przetworznia klatek
        n, c, h, w = net.inputs[self.net_inputs].shape
        self.params = { 'n': n, 'c': c, 'h': h, 'w': w }
        del net


    def preprocess_frame(self, frame):
        in_frame = cv2.resize(frame, (self.params['w'], self.params['h']))
        in_frame = in_frame.transpose((2, 0, 1)) 
        in_frame = in_frame.reshape((self.params['n'], self.params['c'], self.params['h'], self.params['w']))
        return in_frame


    def start(self, request_id, frame):
        self.engine.start_async(request_id = request_id, inputs = {self.net_inputs: frame})


    def request(self, request_id):
        return self.engine.requests[request_id].wait(-1) == 0


    def response(self, request_id):
        return self.engine.requests[request_id].outputs[self.net_outputs]


    def can_read(self, response, threshold): 
        return response > threshold


    def weights(self, model_path):
        return os.path.splitext(model_path)[0] + ".bin"


    def inputs(self, inputs):
        return next(iter(inputs))


    def outputs(self, outputs):
        return next(iter(outputs))


class FaceDetector(Detector):
    
    def extract_face(self, face_object, frame, initial_width, initial_height):
        # pobierz obraz twarzy
        x_min, y_min = int(face_object[3] * initial_width), int(face_object[4] * initial_height)
        x_max, y_max = int(face_object[5] * initial_width), int(face_object[6] * initial_height)
        
        if y_min < 0:
            y_min = 0
        if x_min < 0:
            x_min = 0

        return frame[y_min:y_max, x_min:x_max].copy()


class EmotionDetector(Detector):
    def extract_emotion(self, response):
        emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']
        emotions = response[0][0][0], response[1][0][0], response[2][0][0], response[3][0][0], response[4][0][0]
        emotion_index = emotions.index(max(emotions))
        return emotion_labels[emotion_index]


class GenderDetector(Detector):
    def extract_gender(self, response): 
        female = response[0][0][0]
        male = response[1][0][0]
        
        if female > male:
            return 'female'
        return 'male'

    def outputs(self, outputs):
        iterator = iter(outputs)
        first = next(iterator)
        second = next(iterator)
        return second

