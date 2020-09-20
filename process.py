import cv2
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
#from sklearn.decomposition import FastICA

class Process(object):
    def __init__(self):
        #As in GUI.py an object is created so by default this constructor gets initialised
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        #self.red = np.zeros((256,256,3),np.uint8)
        
    def extractColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        #A 3D matrix where The first two coordinates are the coordinates of the pixel. The third index is the color component;
        #Take mean of all the green pixels in the image
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g           
        
    def run(self):
        
        #Call face_detect function from FaceDetection class 
        frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        #after grayscale and roi detection we get frame out
        self.frame_out = frame
        self.frame_ROI = face_frame
        
        #extracted green colors
        g1 = self.extractColor(ROI1)
        g2 = self.extractColor(ROI2)
        #g3 = self.extractColor(ROI3)
        
        L = len(self.data_buffer)
        
        #calculate average green value of 2 ROIs
        #r = (r1+r2)/2
        g = (g1+g2)/2
        #b = (b1+b2)/2
        
        #---------------
        #Ye nahi samjha kis liye
        if(abs(g-np.mean(self.data_buffer))>10 and L>99): #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = self.data_buffer[-1]

        #time append and append g value in the buffer for a frame
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        
        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
        #----------------

        #create an array of data_buffer
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 10 frames
        #This part will only be executed when 10 frames are captured because buffer_size is 100
        if L == self.buffer_size:

            #-----processing-----#
            
            self.fps = float(L) / (self.times[-1] - self.times[0])#calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(self.times[0], self.times[-1], L)#Returns num evenly spaced samples, calculated over the interval [start, stop] and L samples.

            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change

            #(x,xp=100,fp=100)
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1

            #input para is the window size and output is the window with max value normalized to one
            interpolated = np.hamming(L) * interpolated#make the signal become more periodic (advoid spectral leakage)
            #norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization

            #normalize the array
            norm = interpolated/np.linalg.norm(interpolated)

            #Compute the one-dimensional discrete Fourier Transform for input array.
            raw = np.fft.rfft(norm*30)#do real fft with the normalization multiplied by 10

            #calculate freqs by dividing fps by l*half of length of data_buffer
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs

            # idx_remove = np.where((freqs < 50) & (freqs > 180))
            # raw[idx_remove] = 0
            #get absolute value
            self.fft = np.abs(raw)**2#get amplitude spectrum

            #---------------------#

            #After processing three arrays are calculated frequency ,fps and raw(fourier transformed)
        
            #select freq within range 50 and 100
            #returns element of frequency
            idx = np.where((freqs > 50) & (freqs < 180))#the range of frequency that HR is supposed to be within 
            #create a pruned array of raw fft by selecting only thos elements from fft with index freq in range 
            pruned = self.fft[idx]
            #create a new array of frequencies
            pfreq = freqs[idx]
            
            #Assign 
            self.freqs = pfreq 
            self.fft = pruned
            
            #returns indices of the maximum value from pruned 
            idx2 = np.argmax(pruned)#max in the range can be HR
            
            #Assign bpm the value from freqs at index having max of pruned 
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
            
            #The detrend frame(data buffer) and fps is given to this
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 3)
            #ifft = np.fft.irfft(raw)

        #This will be executed after every frame
        self.samples = processed # multiply the signal with 5 for easier to see in the plot
        #TODO: find peaks to draw HR-like signal.
        

        #-------pending------
        if(mask.shape[0]!=10): 
            out = np.zeros_like(face_frame)
            mask = mask.astype(np.bool)
            out[mask] = face_frame[mask]
            if(processed[-1]>np.mean(processed)):
                out[mask,2] = 180 + processed[-1]*10
            face_frame[mask] = out[mask]
            
            
        #cv2.imshow("face", face_frame)
        #out = cv2.add(face_frame,out)
        # else:
            # cv2.imshow("face", face_frame)
    
    def reset(self):
        #Various parameters being initialized 
        #Firstly three frames are initialised
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        #Filter data along one-dimension with an IIR or FIR filter.b and 1d seq
        y = signal.lfilter(b, a, data)
        return y    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
