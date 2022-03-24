import mediapipe as mp
import time
import numpy as np
import cv2

# variables 
frame_counter =0

# constants 
FONTS =cv2.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# Iris indices
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)

    

def textWithBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3, bgOpacity=0.5):
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    overlay = img.copy() # coping the image
    cv2.rectangle(overlay, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    new_img = cv2.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0) # overlaying the rectangle on the image.
    cv2.putText(new_img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text
    img = new_img

    return img


def fillPoly(img, points, color, opacity):
    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = img.copy()  # coping the image
    cv2.fillPoly(overlay,[list_to_np_array], color )
    new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    # print(points_list)
    img = new_img
    cv2.polylines(img, [list_to_np_array], True, color,1, cv2.LINE_AA)
    return img

map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv2.VideoCapture(0)
# landmark detection function 
def landmarksDetection(img, results, count=0, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[count].landmark]
    if draw :
        [cv2.circle(img, p, 2, GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5, max_num_faces=5,refine_landmarks=True) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        #  resizing frame
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        face_count = 0
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, face_count, False)
            face_count = (face_count+1) % len(results.multi_face_landmarks)
            frame =fillPoly(frame, [mesh_coords[p] for p in FACE_OVAL], WHITE, opacity=0.4)
            frame =fillPoly(frame, [mesh_coords[p] for p in LEFT_EYE], GREEN, opacity=0.4)
            frame =fillPoly(frame, [mesh_coords[p] for p in RIGHT_EYE], GREEN, opacity=0.4)
            frame =fillPoly(frame, [mesh_coords[p] for p in LEFT_EYEBROW], ORANGE, opacity=0.4)
            frame =fillPoly(frame, [mesh_coords[p] for p in RIGHT_EYEBROW], ORANGE, opacity=0.4)
            frame =fillPoly(frame, [mesh_coords[p] for p in LIPS], BLACK, opacity=0.3)
            frame =fillPoly(frame, [mesh_coords[p] for p in LEFT_IRIS], BLUE, opacity=0.3)
            frame =fillPoly(frame, [mesh_coords[p] for p in RIGHT_IRIS], BLUE, opacity=0.3)

            [cv2.circle(frame,mesh_coords[p], 1, GREEN , -1, cv2.LINE_AA) for p in LIPS]
            [cv2.circle(frame,mesh_coords[p], 1, BLACK ,- 1, cv2.LINE_AA) for p in RIGHT_EYE]
            [cv2.circle(frame,mesh_coords[p], 1, BLACK , -1, cv2.LINE_AA) for p in LEFT_EYE]

            [cv2.circle(frame,mesh_coords[p], 1, BLACK , -1, cv2.LINE_AA) for p in RIGHT_EYEBROW]
            [cv2.circle(frame,mesh_coords[p], 1, BLACK , -1, cv2.LINE_AA) for p in LEFT_EYEBROW]
            [cv2.circle(frame,mesh_coords[p], 1, RED , -1, cv2.LINE_AA) for p in FACE_OVAL]

            [cv2.circle(frame,mesh_coords[p], 1, MAGENTA , -1, cv2.LINE_AA) for p in LEFT_IRIS]
            [cv2.circle(frame,mesh_coords[p], 1, MAGENTA , -1, cv2.LINE_AA) for p in RIGHT_IRIS]

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(np.array([mesh_coords[p] for p in LEFT_IRIS]))
            center = (int(l_cx),int(l_cy))
            radius = int(l_radius)
            cv2.circle(frame, center, radius, (0,255,0), -1, cv2.LINE_AA)





        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame = textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (20, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key==ord('q') or key ==ord('Q'):
            break
    cv2.destroyAllWindows()
    camera.release()