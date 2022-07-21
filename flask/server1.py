import base64
from flask_cors import CORS
from flask import Flask,request
import cv2
import numpy as np
app=Flask(__name__,static_folder='static1')
CORS(app,resources=r'/*')
@app.route('/trans',methods=['POST'])
def transImg():
    file1=request.json['img1']
    file2=request.json['img2']
    try:
        data1=str.split(file1,',')[1]
        img_data1=base64.urlsafe_b64decode(data1+'='*(4-len(data1)%4))
        img_data1=np.frombuffer(img_data1,np.uint8)
        img_data1=cv2.imdecode(img_data1,cv2.IMREAD_COLOR)
        data2=str.split(file2,',')[1]
        img_data2=base64.urlsafe_b64decode(data2+'='*(4-len(data2)%4))
        img_data2=np.frombuffer(img_data2,np.uint8)
        img_data2=cv2.imdecode(img_data2,cv2.IMREAD_COLOR)
        
    #加
        plus = cv2.add(img_data1, img_data2)
        cv2.imwrite('./static1/加.jpg', plus)
    #减
        minus = cv2.subtract(img_data1, img_data2)
        cv2.imwrite('./static1/减.jpg', minus)
    #乘
        multiply = cv2.multiply(img_data1, img_data2)
        cv2.imwrite('./static1/乘.jpg', multiply)
    #除
        divide = cv2.divide(img_data1, img_data2)
        cv2.imwrite('./static1/除.jpg', divide)
    #与
        yu = img_data1 & img_data2
        cv2.imwrite('./static1/与.jpg', yu)
    #或
        huo = img_data1 | img_data2
        cv2.imwrite('./static1/或.jpg', huo)
    #非
        fei = ~img_data1
        cv2.imwrite('./static1/非.jpg', fei)
        
        return 'ok'
    except:
        return 'err'
app.run()