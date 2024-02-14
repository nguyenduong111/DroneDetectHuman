import redis
import json
import numpy as np
import cv2
import base64
import time
import os
# import ntplib
def byte2imageCV(string):
    byte = base64.b64decode(string)
    byte_image = np.frombuffer(byte, np.uint8)
    image = cv2.imdecode(byte_image, flags=cv2.IMREAD_COLOR)
    if image is None:
        print("[ERROR] convert byte to image")
    return image

def makeFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        return True
    return False

def getTime():
    return time.time()

# Init connection
RD = redis.Redis(host='192.168.1.36', port=6379, db=0)
makeFolder('buffer_on_lap')
pre_t = getTime()

# First time receiving data
result = RD.get('pi')
result_data = json.loads(result)

real_fps = 0
pre_time_s = time.time()
cnt = 0
pre_name = ''
while True:
    # get data form pi
    result = RD.get('pi')
    result_data = json.loads(result)
    

    if pre_name != result_data['name']:
        pre_name = result_data['name']
    else:
        continue
    # send response
    data_resp = {
        'name' : result_data['name']
    }
    json_data = json.dumps(data_resp)
    RD.set('client', json_data)

    # convert to image
    img_byte = result_data['image']
    name_file = result_data['name']
    img = byte2imageCV(img_byte)
    curr_t = getTime()
    cv2.imshow("a", img)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join('buffer_on_lap', name_file), img)
    
    # ***** show information *****
    pre_time_c = float(result_data['name'][:-4]) # check delay
    
    # check real fps
    curr_time_s = time.time()
    if curr_time_s - pre_time_s >= 1:
        pre_time_s = curr_time_s
        cnt = 0
    


    print(f' ***************************************** \n'+
            f'[Delay]: {time.time() - float(pre_time_c)}\n' +
            f'[Real FPS]: {real_fps}\n'
    )

    with open("delay.txt", "a") as f:
        f.write("\n" + str(time.time() - float(pre_time_c)))

    # ****************************

    # show image
    # cv2.imshow("Video from pi", img)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break
    cnt+=1