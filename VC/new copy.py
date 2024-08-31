# Object Detecion 
import cv2
from ultralytics import YOLO

#basics
import pandas as pd

path = "VC/traffic_vid.mp4"

# Load pre-trained model
model = YOLO('yolov8n.pt')

#geting names from classes
dict_classes = model.model.names

### Configurations
# Scaling percentage of original frame
scale_percent = 150

# Auxiliary functions
def resize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

#-------------------------------------------------------
# Reading video with cv2
video = cv2.VideoCapture(path)

# Objects to detect Yolo
class_IDS = [2, 3, 5, 7]
 
# Auxiliary variables
centers_old = {}
centers_new = {}
# obj_id = 0 
vehicles_counter_in = dict.fromkeys(class_IDS, 0)
vehicles_counter_out = dict.fromkeys(class_IDS, 0)
# end = []
# frames_list = []
cy_linha = int(200 * scale_percent/100 )
cx_sentido = int(300 * scale_percent/100) 
offset = int(8 * scale_percent/100 )
counter_in = 0
counter_out = 0

# Original informations of video
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video.get(cv2.CAP_PROP_FPS)
print('[INFO] - Original Dim: ', (width, height))

# Scaling Video for better performance 
if scale_percent != 100:
    print('[INFO] - Scaling change may cause errors in pixels lines ')
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    print('[INFO] - Dim Scaled: ', (width, height))

#-------------------------------------------------------
# Executing Recognition
vid = cv2.VideoCapture(path)

while True:
    # reading frame from video
    ret, frame = video.read()
    
    if not ret:
        break
    
    #Applying resizing of read frame
    frame  = resize_frame(frame, scale_percent)

    # Getting predictions
    pred = model.predict(frame, conf = 0.7, classes = class_IDS, device = 0, verbose = False)
    
    # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
    boxes   = pred[0].boxes.xyxy.cpu().numpy()
    conf    = pred[0].boxes.conf.cpu().numpy()
    classes = pred[0].boxes.cls.cpu().numpy() 

    # Extract the data attribute from the Boxes object
    # boxes_data = pred[0].cpu().numpy().boxes.data
    
    # Storing the above information in a dataframe
    # positions_frame = pd.DataFrame(boxes_data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    positions_frame = pd.DataFrame(pred[0].cpu().numpy().boxes.data, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    
    #Translating the numeric class labels to text
    labels = [dict_classes[i] for i in classes]
    
    # Drawing transition line for in\out vehicles counting 
    cv2.line(frame, (0, cy_linha), (int(700 * scale_percent/100 ), cy_linha), (0,255,0), 1)
    
    # For each vehicles, draw the bounding-box and counting each one the pass thought the transition line (in\out)
    for ix, row in enumerate(positions_frame.iterrows()):
        # Getting the coordinates of each vehicle (row)
        xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')
        
        # Calculating the center of the bounding-box
        center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
        
        # drawing center and bounding-box of vehicle in the given frame 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1) # box
        cv2.circle(frame, (center_x,center_y), 1,(255,0,0),-1) # center of box
        
        #Drawing above the bounding-box the name of class recognized.
        # cv2.putText(img=frame, text=labels[ix]+' - '+str(np.round(conf[ix],2)),
        cv2.putText(img=frame, text=labels[ix],
                    org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
        
        # Checking if the center of recognized vehicle is in the area given by the transition line + offset and transition line - offset 
        if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
            if  (center_x >= 0) and (center_x <=cx_sentido):
                counter_in +=1
                vehicles_counter_in[category] += 1
            else:
                counter_out += 1
                vehicles_counter_out[category] += 1
    
    #updating the counting type of vehicle 
    counter_in_plt = [f'{dict_classes[k]}: {i}' for k, i in vehicles_counter_in.items()]
    counter_out_plt = [f'{dict_classes[k]}: {i}' for k, i in vehicles_counter_out.items()]
    
    #drawing the number of vehicles in\out
    cv2.putText(img=frame, text='N. vehicles In', 
                org= (30,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=1, color=(255, 255, 0),thickness=1)
    
    cv2.putText(img=frame, text='N. vehicles Out', 
                org= (int(390 * scale_percent/100 ),30), 
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=1)
    
    #drawing the counting of type of vehicles in the corners of frame 
    xt = 40
    for txt in range(len(counter_in_plt)):
        xt +=30
        cv2.putText(img=frame, text=counter_in_plt[txt], 
                    org= (30,xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=1, color=(255, 255, 0),thickness=1)
        
        cv2.putText(img=frame, text=counter_out_plt[txt],
                    org= (int(390 * scale_percent/100 ),xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 0),thickness=1)
    
    #drawing the number of vehicles in\out
    cv2.putText(img=frame, text=f'In:{counter_in}', 
                org= (int(1820 * scale_percent/100 ),cy_linha+60),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)
    
    cv2.putText(img=frame, text=f'Out:{counter_out}', 
                org= (int(1820 * scale_percent/100 ),cy_linha-40),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cv2.destroyAllWindows()
vid.release()
