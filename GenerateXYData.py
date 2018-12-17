
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pickle
#import cv2
import os
import json_lines
import json
from PIL import Image
import cv2
import librosa
import librosa.display
import CreateSpectrogramLibrosa as specrosa
def ConvertDataToLabels(data):
    return data['genre']

def ConvertDataToFeats(data):
    labels =[]   
    labels.extend([data['danceability']])
    labels.extend([data['energy']])
    labels.extend([data['valence']])
    labels.extend([data['acousticness']])
    labels.extend([data['loudness']])    
    return labels

    
def ClassifyHighLow(strFeature, doubleRating):
    if(doubleRating<.5):
        return 'low_'+strFeature
    else:
        return 'high_'+strFeature

def ClassifyHighLow(strFeature, doubleRating):
    if(doubleRating<.5):
        return 'low_'+strFeature
    else:
        return 'high_'+strFeature


xDataList = []
genreData = []
featData=[]
drWavs= r'C:\Users\Michael Yu\Desktop\Music Recommender\wavs'
drImgs= r'C:\Users\Michael Yu\Desktop\Music Recommender\imgs'
dataJl = r'C:\Users\Michael Yu\Desktop\Music Recommender\totaldata.json'

#loop over files in wav folder
directory = os.fsencode(drWavs)
with open(dataJl) as json_file:  
    jData = json.load(json_file)
    
''' Make the imgpath valid,about 100 samples missing cuz of this'''
count =0
nslices=10
for j in jData:
    print(j['genre'])
    filepath = j['path']
    count=count+1
    labels = ConvertDataToLabels(j)
    feats = ConvertDataToFeats(j)
    
    y, sr = librosa.load(filepath, mono=True, duration=30) 
    songslices = []
    for i in range(nslices):
        songslices.append(y[i*65024:(i+1)*65024])
    for Slice in songslices:
        # Make Mel spectrogram
        S = librosa.feature.melspectrogram(Slice, sr=sr, n_mels=128)
        # Convert to log scale (dB)
        log_S = librosa.power_to_db(S, ref=np.max)
        xDataList.extend([log_S])
        genreData.extend([labels])
        featData.extend([feats])
   
    
            
xDataSmaller=[]
genreDataSmaller=[]
featDataSmaller=[]
pergenre = 750
RockCount = 0
PopCount=0
HipHopCount=0
FolkCount=0
InstrumentalCount=0
ElectronicCount=0
#['Rock','Pop', 'Hip-Hop','Folk','Instrumental','Electronic']
for n in range(0, len(xDataList)):
    if(genreData[n]=='Rock' and RockCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        RockCount = RockCount+1
    elif (genreData[n]=='Pop' and PopCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        PopCount = PopCount+1
    elif (genreData[n]=='Hip-Hop' and HipHopCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        HipHopCount = HipHopCount+1
    elif (genreData[n]=='Folk' and FolkCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        FolkCount = FolkCount+1
    elif (genreData[n]=='Instrumental' and InstrumentalCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        InstrumentalCount = InstrumentalCount+1
    elif (genreData[n]=='Electronic' and ElectronicCount<pergenre):
        xDataSmaller.extend([xDataList[n]])
        featDataSmaller.extend([featData[n]])
        genreDataSmaller.extend([genreData[n]])
        ElectronicCount = ElectronicCount+1
        
            

for xdata in xDataList:
    if (xdata.shape != xDataList[0].shape):
        print(xdata.shape)
            
with open('xDataMel.pkl', 'wb') as f:
    pickle.dump(np.asarray(xDataList), f)        

with open('genreData.pkl', 'wb') as f:
    pickle.dump(np.asarray(genreData), f)        
    
with open('featData.pkl', 'wb') as f:
    pickle.dump(np.asarray(featData), f)        

    
with open('xDataMelSmaller2.pkl', 'wb') as f:
    pickle.dump(np.asarray(xDataSmaller), f)        

with open('genreDataSmaller2.pkl', 'wb') as f:
    pickle.dump(np.asarray(genreDataSmaller), f)        
    
with open('featDataSmaller2.pkl', 'wb') as f:
    pickle.dump(np.asarray(featDataSmaller), f)        

    
