import cv2
import numpy as np
import os
import time

red_low1=np.array([0,160,120])
red_up1=np.array([3,255,255])
red_low2=np.array([177,160,120])
red_up2=np.array([180,255,255])

yellow_low=np.array([20,100,100])
yellow_up=np.array([45,255,255])

green_low=np.array([50,100,100])
green_up=np.array([90,255,255])

cap=cv2.VideoCapture("traffic_light1.mp4")

if not cap.isOpened():
    print("Could not open the video source.")
    exit()

#video writer
fourcc=cv2.VideoWriter_fourcc(*'mp4v') #define codec for video writer
fps=int(cap.get(cv2.CAP_PROP_FPS)) #retrive frames per second
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #retrives width of frame
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #retrives height of frame
out=cv2.VideoWriter('output_traffic_light1.mp4',fourcc,fps,(width,height)) #stores the output frames with same dimensions

#accuracy tracking
detec_log={'Red':{'TP':0,'FP':0}, 'Yellow':{'TP':0,'FP':0},'Green':{'TP':0,'FP':0}} #initialises true positives and false positives for each colour
frame_count=0 #count the number of frames processed
start_time=time.time() #records the start time tp calculate total processing time later

#function for processing frame
def detect_light(mask,colour_label,frame):
    global detec_log,frame_count
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #finds the outer lining of the contour(detected traffic light circle)
    print(f"{colour_label} contours found: {len(contours)}")
    detected=False
    for count in contours: #loops through every contour detected
        area=cv2.contourArea(count) #calculates the area of the detected contour
        if area<50 or area>10000: #size constraint
            detec_log[colour_label]['FP']+=1 
            continue
        perimeter=cv2.arcLength(count,True)
        if perimeter==0:
            detec_log[colour_label]['FP']+=1
            continue
        circularity=4*np.pi*(area/(perimeter**2)) #calculates roundness of the contour
        if circularity <0.5:
            detec_log[colour_label]['FP']+=1
            continue
        x,y,w,h=cv2.boundingRect(count) #getting coordinates for rectangular bounding box

        roi_red    = cv2.countNonZero(mask_red[y:y+h, x:x+w])
        roi_yellow = cv2.countNonZero(mask_yellow[y:y+h, x:x+w])
        roi_green  = cv2.countNonZero(mask_green[y:y+h, x:x+w])

        if roi_red > roi_yellow and roi_red > roi_green:
            label = "Red"
        elif roi_yellow > roi_green:
            label = "Yellow"
        else:
            label = "Green"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2) #annotation specifications
        detec_log[label]['TP'] += 1
        detected=True
    if detected:
        detec_log[colour_label]['TP']+=1 #increments the number of detected contour for that colour(assuming only true positives)
    return frame #returns the annotaed frame


while True:
    ret,frame=cap.read() #return is true if frame is read successfully
    if not ret:
        break

    frame_count+=1
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #converting frame from default BGR to HSV colour space

    #binary masks
    mask_red1=cv2.inRange(hsv,red_low1,red_up1) #creating binary masks for red light
    mask_red2=cv2.inRange(hsv,red_low2,red_up2)
    mask_red=cv2.bitwise_or(mask_red1,mask_red2) #using bitwise or since red has two ranges

    #add noise reduction
    mask_red = cv2.erode(mask_red, np.ones((5, 5), np.uint8), iterations=1)
    mask_red = cv2.dilate(mask_red, np.ones((5, 5), np.uint8), iterations=1)

    mask_yellow=cv2.inRange(hsv,yellow_low,yellow_up) #creating binary masks for yellow light
    mask_yellow = cv2.erode(mask_yellow, np.ones((5, 5), np.uint8), iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, np.ones((3, 3), np.uint8), iterations=1)
    mask_yellow = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_red))
    mask_green=cv2.inRange(hsv,green_low,green_up)#creating binary masks for green light
    mask_green = cv2.erode(mask_green, np.ones((5, 5), np.uint8), iterations=1)
    mask_green = cv2.dilate(mask_green, np.ones((5, 5), np.uint8), iterations=1)
    mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_yellow))

    #pixel counts
    red_pix = cv2.countNonZero(mask_red)
    yellow_pix = cv2.countNonZero(mask_yellow)
    green_pix = cv2.countNonZero(mask_green)
    print(f"Frame {frame_count}: Red pixels: {red_pix}, Yellow pixels: {yellow_pix}, Green pixels: {green_pix}")
    #detect lights and annotate
    frame=detect_light(mask_red,"Red",frame)
    frame=detect_light(mask_yellow,"Yellow",frame)
    frame=detect_light(mask_green,"Green",frame)

    #pixel based detection
    state = "None"
    if red_pix > 200: 
        state = "Red"
    elif green_pix > 100:
        state = "Green"
    elif yellow_pix > 50:
        state = "Yellow"



    cv2.putText(frame, f"Traffic Light: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    #display and save frame
    cv2.imshow("Mask Red", mask_red) 
    cv2.imshow("Mask Yellow", mask_yellow)
    cv2.imshow("Mask Green",mask_green)
    cv2.imshow("Traffic Light Detection ",frame)
    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #accuracy report
    end_time=time.time()
    total_time=end_time-start_time
    with open('accuracy_report.txt','w') as f:
        f.write(f"Traffic Light Detection Accuracy Report\n")
        f.write(f"Total Frames Proccessed: {frame_count}\n")
        f.write(f"Total Processing Time: {total_time:.2f}\n")
        f.write(f"FPS: {frame_count/ total_time:.2f}\n")
        for colour in detec_log:
            tp=detec_log[colour]['TP']
            fp=detec_log[colour]['FP']
            total=tp+fp
            precision=tp/(tp+fp) if (total)>0 else 0
            recall=tp/frame_count if frame_count>0 else 0
            f.write(f"{colour}- True Positives: {tp}, False Positives: {fp}, Precison: {precision:.2f}, Recall: {recall:.2f}\n")
    print("Processing complete. Output video saved as 'output_traffic_light.mp4', screenshots saved, and accuracy report saved as 'accuracy_report.txt'.")

cap.release()
cv2.destroyAllWindows()
