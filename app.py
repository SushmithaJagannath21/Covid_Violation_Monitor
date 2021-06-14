import streamlit as st 
import tempfile
import cv2
import tensorflow as tf
import numpy as np
from glob import glob
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import glob
import torch
import os
import time
import math
import pickle
from imutils.video import VideoStream
import face_recognition
import imutils
import pandas as pd
import base64
import gc
import plotly.graph_objects as go

gc.enable()
from retinaface.pre_trained_models import get_model as get_detector
#model = tf.keras.models.load_model('my_model1.h5')
#detector = get_detector("resnet50_2020-07-20", max_size=800)

#data = pickle.loads(open("encodings.pickle", "rb").read())
#detector = pickle.loads(open("detector.pickle", "rb").read())

        
        
caught=[]
def face_recognise(detector, data, frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])


    boxes = face_recognition.face_locations(rgb,model=detector)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []


    for encoding in encodings:
        name1=0
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        print(matches)
        name = "Unknown"


        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}


            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                print(counts)

            name = max(counts, key=counts.get)
            name1 = max(counts.values()) 
            print(name1)
        if(name1>5):
            names.append(name)
        else:
            names.append("Unknown")

    
    for ((top, right, bottom, left), name) in zip(boxes, names):

        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)


        #cv2.rectangle(frame, (left, top), (right, bottom),
         #   (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
        caught.append(name)
    return frame

class_map = {
    0: 'With mask',
    1: 'Without mask'
    }

color_map_video = {
    0: (0,255,0),
    1:(0,0,255)
    }

color_map_image = {
    0: [0,1,0],
    1:[1,0,0]
    }
image_h, image_w = 128, 128
classes = sorted(['with_mask','without_mask'])




def visualize_detections_video(image,model, boxes, mask, nomask,maskper,nomaskper):
  count_mask=0
  count_nomask=0
  image = np.array(image, dtype=np.uint8)
  
  for box in boxes:
    x, y, w1, h1 = box
    w=w1-x
    h=h1-y    
    face_image = image[y:y+h,x:x+w] 
    
    #To handle those cases where the  height and width of the generated cropped face become 0
    if face_image.shape[0] and face_image.shape[1]:
      face_image = tf.image.resize(face_image, [image_w, image_h])
      face_image = face_image/127.5-1
      
      _cls = model.predict(np.expand_dims(face_image,axis=0))
      _cls = np.argmax(_cls,axis=1)
      if _cls[0] == 0:
        count_mask=count_mask+1
      elif _cls[0] == 1:
        count_nomask= count_nomask+1    
      text = '{}'.format(class_map[_cls[0]])    
      start = tuple(np.array((x,y)).astype('int'))
      end = tuple(np.array((x+w,y+h)).astype('int'))
      mask.text("TOTAL WITH MASK:"+"                                        "+(str(count_mask)))
      nomask.text("TOTAL WITHOUT MASK:"+"                                     "+(str(count_nomask)))
      maskperc=(count_mask/(count_mask+count_nomask))*100
      nomaskperc=(count_nomask/(count_mask+count_nomask))*100  
      maskper.text("MASKED % :"+"                                              "+(str(maskperc)))
      nomaskper.text("WITHOUT MASK % :"+"                                        "+(str(nomaskperc)))  
      cv2.rectangle(image,start,end,color_map_video[_cls[0]],2)
      cv2.putText(image, class_map[_cls[0]], start , cv2.FONT_HERSHEY_SIMPLEX, 0.70, color_map_video[_cls[0]], 2)
  return image, maskperc, nomaskperc

def social_dist(net, img,textp1,textp2,textp3,high, safe):
    low_risk_p = 0       
    high_risk_p = 0      
    safe_p = 0
    #net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = ["person"]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    np.random.uniform(0, 255, size=(len(classes),3))
    riskperc=0
    safeperc=0
    #cap = cv2.VideoCapture('pedestrians.mp4')

    def E_dist(p1, p2):
        x_dist=p1[0]-p2[0]
        y_dist=p1[1]-p2[1]
        return math.sqrt((x_dist * x_dist)+  (y_dist * y_dist))

    def isclose(p1, p2):
        c_d = E_dist(p1, p2)
        
        calib = (p1[1] + p2[1]) / 2
        if 0 < int(c_d) < 120:
            return 1

        elif 120 < int(c_d) < 220:
            return 2

        else:
            return 0

    height,width=(None,None)
    q=0       


   # while(cap.isOpened()):
       # ret, img = cap.read()  
      #  print(ret)

       # if not ret:
        #    break

    if width is None or height is None: 
        height,width=img.shape[:2]
        q=width

    img =img[0:height, 0:q]
    height,width=img.shape[:2]

    blob = cv2.dnn.blobFromImage(img,0.00392, (416, 416), (0,0,0), True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end=time.time()
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    font = cv2.FONT_HERSHEY_SIMPLEX    
    if len(indexes)>0:        
        status=list()        
        idf = indexes.flatten()        
        close_pair = list()        
        s_close_pair = list()        
        center = list()        
        dist = list()        
        for i in idf:            
            (x, y) = (boxes[i][0], boxes[i][1])            
            (w, h) = (boxes[i][2], boxes[i][3])            
            center.append([int(x + w / 2), int(y + h / 2)])            
            status.append(0)            

        for i in range(len(center)):            
            for j in range(len(center)):               
                g=isclose(center[i], center[j])                
                if g ==1:                    
                    close_pair.append([center[i],center[j]])                    
                    status[i] = 1                    
                    status[j] = 1                    

                elif g == 2:                    
                    s_close_pair.append([center[i], center[j]])                    
                    if status[i] != 1:                        
                        status[i] = 2                        

                    if status[j] != 1:                        
                        status[j] = 2

        total_p = len(center)        
        low_risk_p = low_risk_p+status.count(2)        
        high_risk_p = high_risk_p+status.count(1)        
        safe_p = safe_p + status.count(0)
        safeperc= (safe_p/total_p)*100
        riskperc=((low_risk_p+high_risk_p)/total_p)*100
        high.text("SOCIAL DISTANCING VIOLATOR PERCENTAGE:"+"                  "+(str(riskperc)))
        safe.text("SOCIAL DISTANCING MAINTAINED PERCENTAGE:"+"                "+(str(safeperc)))
        kk = 0        

        for i in idf:            
            
            (x, y) = (boxes[i][0], boxes[i][1])            
            (w, h) = (boxes[i][2], boxes[i][3])        

            if status[kk] == 1:                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1
    
    textp1.text("LOW RISK:"+"                                               "+(str(low_risk_p)))
    textp2.text("HIGH RISK:"+"                                              "+(str(high_risk_p)))        
    textp3.text("SAFE:"+"                                                   "+(str(safe_p)))
        
    return img, riskperc, safeperc

def get_table_download_link1(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="summary.csv">Download summary file</a>'
    return href

def get_table_download_link2(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="summary.csv">Download No-mask Identities</a>'
    return href

@st.cache(allow_output_mutation=True, max_entries=2, ttl=60)
def download_wget():
    path1 = './my_model.h5'
    path2 = './yolov3.weights'
    if not os.path.exists(path1):    
            model_url = 'wget -O my_model1.h5 https://www.dropbox.com/s/7meuxh8a6iul0e0/my_model1.h5'

            with st.spinner('done!\nmodel was not found, downloading them...'):
               os.system(model_url)
    else:
        print("Model is here.")

    if not os.path.exists(path2):    
            yolo_url = 'wget -O yolov3.weights https://www.dropbox.com/s/oeu6m85ahsw22ci/yolov3.weights'
            with st.spinner('Downloading yolo weights'):
               os.system(yolo_url)
    else:
        print("yolo is here.")
         
        
def main():
    """COVID-19 Violator App"""
    original_title = '<p style=" font-type:bold; color:#faca2b; font-size: 36px;">COVID-19 VIOLATION MONITOR</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    download_wget()
    #model_path='my_model1.h5'
    #model = torch.load(model_path)
    #weights_path='./yolov3.weights'
    #yolov3_weights=torch.load(weights_path)
    main_bg = "static/back1.jpg"
    main_bg_ext = "jpg"
    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    .sidebar .sidebar-content {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

    activities = ["Upload","About"]
    choice = st.sidebar.selectbox("MENU",activities)
    model = tf.keras.models.load_model('my_model1.h5')
    # For checkpoint saved elsewhere
    #checkpoint = 'https://www.dropbox.com/s/oeu6m85ahsw22ci/yolov3.weights?dl=0'
    #model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
    detector = get_detector("resnet50_2020-07-20", max_size=800)
    data = pickle.loads(open("encodings.pickle", "rb").read())
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    if choice == 'Upload':
        sub_title = '<p style=" font-type:bold; font-size: 20px;">Here you can upload a video file(.mp4) that you wish to check for violations.</p>'
        st.markdown(sub_title, unsafe_allow_html=True)
        f = st.file_uploader("Upload file")
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            
            frame_rate = 20
            prev = 0
            start=0
            image_placeholder = st.empty()
            people= st.empty()
            textp1= st.empty()
            textp2= st.empty()
            textp3= st.empty()
            namesp= st.empty()
            mask=st.empty()
            nomask=st.empty()
            maskper=st.empty()
            nomaskper=st.empty()
            high=st.empty()
            safe=st.empty()
            cap = cv2.VideoCapture(tfile.name)
            img_array = []
            img_array1 = []
            img_arr2=[]
            maskarr=[]
            nomaskarr=[]
            higharr=[]
            safearr=[]
            namearr=[]
            j=0
            while True:
                now = time.time()
                time_elapsed= now-start
                #print(time_elapsed)
                _,image = cap.read()
                if not _:
                    break
                w, h = image.shape[0], image.shape[1]
                j += 1 

                if time_elapsed > 1./frame_rate:
                    result = detector.predict_jsons(image)
                    #print(j)
                    img1, finhigh, finsafe = social_dist(net, image,textp1,textp2,textp3,high, safe)
                    img_array1.append(img1)
                    higharr.append(finhigh)
                    safearr.append(finsafe)
                    img2=face_recognise(detector, data, image)
                    img_arr2.append(img2)
                    #image_placeholder1.image(img1,channels="BGR") 
                    boxes = []
                    for i in range(len(result)):
                        boxes.append(result[i]['bbox'])
                    boxes = np.array(boxes)
                    people.text("NUMBER OF PEOPLE:"+"                                       "+(str(len(boxes))))
                  
                    #print(len(boxes))
                    #print(boxes)
                    if boxes.size == 0:
                        img_array.append(image)      
                    else:
                        img, finmask, finnomask=visualize_detections_video(image, model, boxes, mask, nomask,maskper,nomaskper)
                        print(finmask)
                        maskarr.append(finmask)
                        nomaskarr.append(finnomask)
                        image_placeholder.image(img, channels="BGR")
                        img_array.append(img)
                    start = time.time()
                    #print(start)
            print(namearr)
            mask_avg=sum(maskarr)
            mask_avg= (mask_avg/len(maskarr))
            #st.text("AVERAGE MASK PERCENTAGE:"+(str(int(mask_avg))))
            nomask_avg=sum(nomaskarr)
            nomask_avg= (nomask_avg/len(nomaskarr))
            #st.text("AVERAGE WITHOUT-MASK PERCENTAGE:"+(str(int(nomask_avg))))
            safe_avg=sum(safearr)
            safe_avg= (safe_avg/len(safearr))
            #st.text("AVERAGE SOCIAL DISTANCE MAINTAINED PERCENTAGE:"+(str(int(safe_avg))))
            high_avg=sum(higharr)
            high_avg= (high_avg/len(higharr))
            #st.text("AVERAGE SOCIAL DISTANCE VIOLATION PERCENTAGE:"+(str(100-int(safe_avg))))
            out = cv2.VideoWriter('demo_.avi', 0, 24, (h,w))
            namelist=list(dict.fromkeys(caught))
            s=" ; "
            s=s.join(namelist)
            #st.text("FACES RECOGNIZED WITHOUT MASK:" + s)
            data1=pd.DataFrame(namelist, columns=["RECOGNIZED PEOPLE"])
            my_dict={'ANALYSIS':["AVERAGE MASK PERCENTAGE:","AVERAGE WITHOUT-MASK PERCENTAGE:","AVERAGE SOCIAL DISTANCE MAINTAINED PERCENTAGE:","AVERAGE SOCIAL DISTANCE VIOLATION PERCENTAGE:"],'SUMMARY %':[(int(mask_avg)),(int(nomask_avg)),(int(safe_avg)),(100-int(safe_avg))]}
            df=pd.DataFrame(my_dict)
            sub_title = '<p style=" font-type:bold; color:#faca2b; font-size: 25px;">STATISTICAL REPORT</p>'
            st.markdown(sub_title, unsafe_allow_html=True)
            fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns),fill_color='darkslategray',align='left'),cells=dict(values=[df['ANALYSIS'],df['SUMMARY %']],fill_color='teal',align='left',height=30))])
            fig.update_layout(
    autosize=False,
    width=400,
    height=400,
    margin=dict(l=2,r=2,b=2,t=2),
    font=dict(family="Courier New, monospace",size=16)
)

            fig.update_yaxes(automargin=True)
            #fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),paper_bgcolor='seagreen',font=dict(family="Courier New, monospace",size=16,))      
            fig1 = go.Figure(data=[go.Table(header=dict(values=list(data1.columns),fill_color='darkslategray',align='left'),cells=dict(values=[data1['RECOGNIZED PEOPLE']],fill_color='teal',align='left',height=30))])
            fig1.update_layout(
    autosize=False,
    width=300,
    height=300,
    margin=dict(l=2,r=2,b=2,t=2),
    paper_bgcolor='teal',
    font=dict(family="Courier New, monospace",size=16)
)
            st.write(fig)
            st.write(fig1)
            st.markdown(get_table_download_link1(df), unsafe_allow_html=True)
            st.markdown(get_table_download_link2(data1), unsafe_allow_html=True)
            for i in range(len(img_array)):
                for _ in range(10): 
                    out.write(img_array[i])
            for i in range(len(img_array1)):
                for _ in range(10): 
                    out.write(img_array1[i])
            for i in range(len(img_arr2)):
                for _ in range(10): 
                    out.write(img_arr2[i])
            out.release()
            cap.release()
            del model, detector, data, net, img_array, img_array1, img_arr2, data1, df, maskarr, nomaskarr, higharr, safearr, namearr, cap 
            gc.collect()
                               
    elif choice == 'About':
        st.subheader("About")
        st.markdown("Built with Streamlit by Rahul Raman R, Sushmitha J, Sanjana V")
        st.text("The Covid-19 Violator Monitor helps monitor people within a designated\narea,it helps monitor people who are wearing or not wearing a mask and\n also monitorthe social distancing followed by people and the system is\nalso capable ofrecognizing the individuals who are not wearing a mask.\nThis allows the authority/administrators to take appropriate action on\nthose who do not follow the safety guidelines.\n\nThis system can be implemented in educational institutions and workspaces\nto prevent and reduce the spread of covid-19 among students and employees.\nThis would significantly benefit these organizations to start it functions\nin the new norm and this would also provide a sense of safety for\nthose who are part of these organizations.")
        st.success("")
    #del model, detector, data, net, img_array, img_array1, img_arr2, data1, df, maskarr, nomaskarr, higharr, safearr, namearr, cap 
    #gc.collect()
    
    
if __name__ == '__main__':
        main()
        
