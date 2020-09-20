import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
import math
from bright_spot_detection import find_bright_spots

class FaceDetection(object):

    #As in process.py an object is created so by default this constructor gets initialised

    def __init__(self):

        #Returns array of face co ordinates
        self.detector = dlib.get_frontal_face_detector()

        #Load shape_predictor_68_face_landmarks.dat
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        #Alignment of face with landmarks
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=256)

    def face_detect(self, frame):
        #frame = imutils.resize(frame, width=400)
        #Make four arrays of heigth and width as 10 
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        #ROI3 = np.zeros((10, 10, 3), np.uint8)
        status = False
        
        if frame is None:
            return 
        
        #grayscale conversion of the frame captured
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply detector to detect frontal face coordinates in the grayscale image
        # O is for image resolution , we can do upscaling to n layers
        rects = self.detector(gray, 0)
        
        # loop over the face detections
        #for (i, rect) in enumerate(rects): 
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            
        #assumpion: only 1 face is detected
        if len(rects)>0:
            print("Face mean: " + str(self.calculate_face_mean_hsv(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))))
            status = True
            # print(find_bright_spots(frame))
            # shape = self.predictor(gray, rects[0])
            # shape = face_utils.shape_to_np(shape)

                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
            #The face that is detected basically a box is shown around it
            #4-tuple coordinates will be created 
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if y<0:
                print("a")
                return frame, face_frame, ROI1, ROI2, status, mask
            #if i==0:
            #New face frame i.e the roi is deteted by changing x an y co ordinates of face by additing width and height
            face_frame = frame[y:y+h,x:x+w]
                # show the face number
                #cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                
            # for (x, y) in shape:
                # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1) #draw facial landmarks
            #shape-he NumPy array that contains the 68 facial landmark coordinates that map to various facial parts.
            if(face_frame.shape[:2][1] != 0):
                face_frame = imutils.resize(face_frame,width=256)
                
            face_frame = self.fa.align(frame,gray,rects[0]) # align face
            
            #after resizing and alignment again grayscale and rectangle formation
            grayf = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            rectsf = self.detector(grayf, 0)
            
            if len(rectsf) >0:
                #predictor giving us the 68 (x, y)-coordinates that map to the specific facial features in the image.
                shape = self.predictor(grayf, rectsf[0])
                #converts dlib shape object to numpy array
                shape = face_utils.shape_to_np(shape)
                
                for (a, b) in shape:
                    #This the very first red circle spot on the face 
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1) #draw facial landmarks
                
                cv2.rectangle(face_frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
                        (shape[12][0],shape[33][1]), (0,255,0), 0)
                cv2.rectangle(face_frame, (shape[4][0], shape[29][1]), 
                        (shape[48][0],shape[33][1]), (0,255,0), 0)
                
                ROI1 = face_frame[shape[29][1]:shape[33][1], #right cheek
                        shape[54][0]:shape[12][0]]
                        
                ROI2 =  face_frame[shape[29][1]:shape[33][1], #left cheek
                        shape[4][0]:shape[48][0]]    

                # ROI3 = face_frame[shape[29][1]:shape[33][1], #nose
                        # shape[31][0]:shape[35][0]]
                
                #get the shape of face for color amplification 
                rshape = np.zeros_like(shape) 
                rshape = self.face_remap(shape)    

                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))
                #Fills a convex polygon(image,pts,no_of_poly_vertices).
                cv2.fillConvexPoly(mask, rshape[0:27], 1) 
                # mask = np.zeros((face_frame.shape[0], face_frame.shape[1],3),np.uint8)
                # cv2.fillConvexPoly(mask, shape, 1)
                
            #cv2.imshow("face align", face_frame)
            
            # cv2.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
                    # (shape[12][0],shape[54][1]), (0,255,0), 0)
            # cv2.rectangle(frame, (shape[4][0], shape[29][1]), 
                    # (shape[48][0],shape[48][1]), (0,255,0), 0)
            
            # ROI1 = frame[shape[29][1]:shape[54][1], #right cheek
                    # shape[54][0]:shape[12][0]]
                    
            # ROI2 =  frame[shape[29][1]:shape[54][1], #left cheek
                    # shape[4][0]:shape[48][0]]   
                
        else:
            high_brightness_range = range(40, 70)
            low_brightness_threshold = 80

            v = self.calculate_face_mean_hsv(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            print("No Face mean: " + str(v))

            if math.floor(v) in high_brightness_range:
                # high brightness 
                cv2.putText(frame, "High brightness detected, Kindly reduce your lighting",
                       (100,100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            elif v > low_brightness_threshold:
                # low brightness
                cv2.putText(frame, "Low brightness detected, Kindly improve your lighting",
                       (100,100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            else:
                # check for bright spots in approximate center of the frame
                bright_spots = find_bright_spots(frame)
                
                if bright_spots <= 1:
                    cv2.putText(frame, "Face lighting low",
                       (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)

            # else:
            #     cv2.putText(frame, "No face detected",
            #            (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)

            status = False
        return frame, face_frame, ROI1, ROI2, status, mask    
    
    #Basically changes of the points out of 68 from shape
    # some points in the facial landmarks need to be re-ordered
    def face_remap(self,shape):
        remapped_image = shape.copy()
        # left eye brow
        remapped_image[17] = shape[26]
        remapped_image[18] = shape[25]
        remapped_image[19] = shape[24]
        remapped_image[20] = shape[23]
        remapped_image[21] = shape[22]
        # right eye brow
        remapped_image[22] = shape[21]
        remapped_image[23] = shape[20]
        remapped_image[24] = shape[19]
        remapped_image[25] = shape[18]
        remapped_image[26] = shape[17]
        # neatening 
        remapped_image[27] = shape[0]
        # Output convex hull. It is either an integer vector of indices or vector of points.
        remapped_image = cv2.convexHull(shape)
        return remapped_image       
        
    def calculate_face_mean_hsv(self, frame):
        # gets the HSV value
        return np.mean(frame[:, 2])
