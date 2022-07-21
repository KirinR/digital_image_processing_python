import base64
from flask_cors import CORS
from flask import Flask,request
import cv2
import numpy as np
app=Flask(__name__,static_folder='static2')
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

    #模板匹配
        tpl = img_data1  # 模板
        target = img_data2  # 目标
 
        th, tw = tpl.shape[:2]  # 模板的高、宽
        result = cv2.matchTemplate(target, tpl, cv2.TM_SQDIFF_NORMED)  # 计算每个位置匹配程度
 
        # 平方差越小越匹配
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        tl = min_loc
 
        # 以红色方框标出匹配区域
        br = (tl[0]+tw, tl[1]+th)
        cv2.rectangle(target, tl, br, (0, 0, 255), 2)
        cv2.imwrite('./static2/模板匹配.jpg', target)
 
        return 'ok'
    except:
        return 'err'
app.run()