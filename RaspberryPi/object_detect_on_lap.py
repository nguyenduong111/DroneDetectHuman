import json
import redis
import base64
import cv2
import numpy as np
import glob
import os
import time
from threading import Thread
import importlib.util

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter


# -------------------------------------------------------------------------
Q = False
LOCK = False
LFRAME = cv2.imread(r'/home/pi/tflite1/test1.jpg')

# setup ip pi -------> redis host
RD = redis.Redis(host='192.168.1.36', port=6379, db=0)

# Set up variables for running user's model yolo
PATH_TO_MODEL_YOLO=r'/home/pi/Desktop/model_yolo/detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS_YOLO=r'/home/pi/Desktop/model_yolo/labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold_YOLO=0.5

# Set up variables for running user's model mobilenet
PATH_TO_MODEL_MOBILENET=r'/home/pi/Desktop/model_mobilenet/detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS_MOBILENET=r'/home/pi/Desktop/model_mobilenet/labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold_MOBILENET=0.35  

#-----func------------------------------------------------------------------

def makeFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        return True
    return False

#format img to byte
def imageCV2Byte(img, format = '.jpg'):
    retval, buffer = cv2.imencode(format, img)
    if not retval:
        print(f"[ERROR] convert image to byte {format}")
    return base64.b64encode(np.array(buffer).tobytes()).decode('utf-8')

def getMinName(files):
    return min(files)

#-----YOLO------------------------------------------------------------------
    
### Define function for inferencing with TFLite model and displaying results
def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def YOLOdetect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = classFilter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    # print(len(scores))


    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

def tflite_detect_images_yolo(modelpath, image, lblpath, min_conf=0.45):

# Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    isHuman = False

    image = cv2.resize(image, (640, 480))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH = image.shape[0]
    imW = image.shape[1]
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # print(input_data)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    boxes, classes, scores = YOLOdetect(output_data) #boxes(x,y,x,y), classes(int), scores(float) [25200]

    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)

    #------NMS----------------------------------------------------------------------------------------------
    def nms(boxes_1, classes, scores, threshold):
        # Tìm chỉ mục của các mục có điểm số cao nhất
        sorted_indices = np.argsort(scores)[::-1]

        # Khởi tạo danh sách các hộp đã chọn
        selected_boxes = []
        boxes = []

        for i in range(len(scores)):
            ymin = int(max(1,(boxes_1[1][i] * imH)))
            xmin = int(max(1,(boxes_1[0][i] * imW)))
            ymax = int(min(imH,(boxes_1[3][i] * imH)))
            xmax = int(min(imW,(boxes_1[2][i] * imW)))
            boxes.append((xmin, ymin, xmax, ymax))

        boxes = np.array(boxes)
        # Lặp qua tất cả các hộp và áp dụng NMS
        while len(sorted_indices) > 0:
            # Lấy chỉ mục của hộp có điểm số cao nhất
            best_index = sorted_indices[0]
            # print("len ", len(boxes))
            best_box = boxes[best_index]

            # Thêm hộp có điểm số cao nhất vào danh sách các hộp đã chọn
            selected_boxes.append((best_box, classes[best_index], scores[best_index]))

            # Tính toán IOU giữa hộp có điểm số cao nhất và các hộp khác
            ious = calculate_iou(best_box, boxes[sorted_indices[1:]])

            # Lấy chỉ mục của các hộp mà iou < ngưỡng threshold
            selected_indices = np.where(ious < threshold)[0]

            # Giữ lại các hộp mà iou >= ngưỡng threshold
            sorted_indices = sorted_indices[selected_indices + 1]

        return selected_boxes
    def calculate_iou(box1, boxes2):
        # Tính toán tọa độ các góc của hộp 1
        x1_1, y1_1, x2_1, y2_1 = box1

        # Tính toán tọa độ các góc của các hộp còn lại
        x1_2, y1_2, x2_2, y2_2 = np.split(boxes2, 4, axis=1)
        # x1_2, y1_2, x2_2, y2_2 = boxes2
        # Tính toán tọa độ các góc của hộp giao nhau
        x1_inter = np.maximum(x1_1, x1_2)
        y1_inter = np.maximum(y1_1, y1_2)
        x2_inter = np.minimum(x2_1, x2_2)
        y2_inter = np.minimum(y2_1, y2_2)
        
        # Tính toán diện tích của hộp giao nhau
        intersection = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
        
        # Tính toán diện tích của hộp 1 và các hộp còn lại
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Tính toán IoU
        iou = intersection / (area1 + area2 - intersection)
        
        return iou
    selected_boxes = nms(boxes, classes, scores, threshold = 0.1)
    selected_boxes = np.array(selected_boxes, dtype="object")
    detail = selected_boxes[selected_boxes[:, 2] > 0.35]
    box = detail[:, 0]
    
    detections = []

    for i in range(len(box)):
        isHuman = True
        xmin,ymin,xmax,ymax = box[i]
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

    # print(detections)
    # cv2.imshow("a", image)
    # cv2.waitKey(0)

    return image, isHuman

# Run inferencing function!
# tflite_detect_images_yolo(PATH_TO_MODEL_YOLO, img, PATH_TO_LABELS_YOLO, min_conf_threshold_YOLO)


#------Mobilenet----------------------------------------------------------------------------------------------------
def tflite_detect_images_mobilenet(modelpath, image, lblpath, min_conf=0.35):

    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    isHuman = False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    # print("---------", scores)
    

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            print("----- ", xmin, ymin, xmax, ymax)

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            if(int(classes[i]) == 0): 
                isHuman = True

            detections.append([xmin, ymin, xmax, ymax])

    # print("---------", detections)
    # print(detections)
    # cv2.imshow("a", image)
    # cv2.waitKey(0)
      
    return image, isHuman

# Run inferencing function!
# tflite_detect_images_mobilenet(PATH_TO_MODEL_MOBILENET, img, PATH_TO_LABELS_MOBILENET, min_conf_threshold_MOBILENET)


#-------Camera-----------------------------------------------------------------------------------

# Đường dẫn đến file video
#video_path = r'D:\AI\data_collect\data_kiem_thu\data_test\c2_3m.mp4'

#cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

makeFolder('data_img_on_lap')
makeFolder('buffer_on_lap')

thr1_is_ready = False
def getImgThread():
    global Q, LOCK, LFRAME, thr1_is_ready

    desired_fps = 0.5
    cap.set(cv2.CAP_PROP_FPS, desired_fps)


    # Kiểm tra xem VideoCapture đã được mở thành công hay không
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()

    while not Q:
       
        # Kiểm tra phím ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

        r, img = cap.read()
            
        if not r:
            break
        
        #cv2.imshow("Video", img)
        
        k = cv2.waitKey(1)
        if k == ord("q"):
            Q = True
        LOCK = True
        #cv2.imwrite(os.path.join('data_img_on_lap', str(time.time())+'.jpg'), img)
        LFRAME = img
        LOCK = False

        # Dừng lại trong khoảng thời gian tương ứng với FPS mong muốn
        #cv2.waitKey(int(1000 / desired_fps))

        thr1_is_ready = True

    # Giải phóng đối tượng VideoCapture và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()


#---------Object detection---------------------------------
def objectDetect():
    global LFRAME, Q, LOCK
    while not Q:
        #imageDetect, isHuman = tflite_detect_images_mobilenet(PATH_TO_MODEL_MOBILENET, LFRAME, PATH_TO_LABELS_MOBILENET, min_conf_threshold_MOBILENET)
        imageDetect, isHuman = tflite_detect_images_yolo(PATH_TO_MODEL_YOLO, LFRAME, PATH_TO_LABELS_YOLO, min_conf_threshold_YOLO)
        print('isHuman: ', isHuman)
        
        if(isHuman):
            LOCK = True
            cv2.imwrite(os.path.join('buffer_on_lap', str(time.time())+'.jpg'), imageDetect)
            LOCK = False
            
        time.sleep(2)

#---------Send image from pi to server---------------------
def sendImgThread():
    global Q
    img_files = glob.glob(os.path.join("buffer_on_lap", "*.jpg"))
    pre_name = ''
    
    while not Q:
        k = cv2.waitKey(1)
        if k == ord("q"):
            Q = True
        if len(img_files) > 0:
            img_path = getMinName(img_files)
            try:
               
                time.sleep(0.5)
                img = cv2.imread(img_path)
                img_byte = imageCV2Byte(img)
                cv2.imshow('a', img)
                cv2.waitKey(1)

                data = {
                'image': img_byte,
                'name' : os.path.basename(img_path)
                }
   # Chuyển đổi dữ liệu thành chuỗi JSON
                json_data = json.dumps(data)

 # Gửi chuỗi JSON tới Redis
                RD.set('pi', json_data)

                resp = RD.get('client')
                resp = json.loads(resp)

                if pre_name != resp['name']:   
                    os.remove(img_path)
                    img_files.remove(img_path)
                    pre_name = resp['name']
                print('------[Succ] ', img_path)
            except:
                #print(f'[Err] {img_path}')
                os.remove(img_path)
                img_files.remove(img_path)
                continue
            
        else:
            img_files = glob.glob(os.path.join("buffer_on_lap", "*.jpg"))


#-----------Main-----------------------------------
            
thr1 = Thread(target=getImgThread)
thr1.start()
while not thr1_is_ready:
    continue

thr2 = Thread(target=objectDetect)
thr2.start()

thr3 = Thread(target=sendImgThread)
thr3.start()