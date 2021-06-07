import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
import pickle
import winsound
import pywt.data
from scipy.fftpack import dct, idct
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from joblib import dump, load

jaffeDatabase = []
jaffeDatabase0 = []
jaffeDatabase1 = []
jaffeDatabase2 = []
jaffeDatabase3 = []
jaffeDatabase4 = []
jaffeDatabase5 = []
jaffeDatabase6 = []
jaffeTarget = []
directory = "F:\\University\\Y3\\S2\\Final Project\\Code\\JAFFE\\"
directory2 = "F:\\University\\Y3\\S2\\Final Project\\Code\\CK+\\"
folders = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
folders2 = ["AN", "DI", "FE", "HA", "SA", "SU"]
blockSizes = [32, 16, 8]

def CLAHE(original):
    claheFinal = cv2.resize(original, (50, 50))
    claheGreyScale = cv2.cvtColor(claheFinal, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit = 5)
    claheFinal = clahe.apply(claheGreyScale)

    return claheFinal

def HaarTransformation(claheFinal):
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(claheFinal, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2

    return LH

def DiscreteCosineTransformation(Input, switch):
    result = None
    def Image():
        dctImg = dct(Input)
        return dctImg

    def List():
        dctList = [[],[],[]]
        i = 0
        for element in Input:
            for img in element:
                dctList[i].append(dct(img))
            i = i + 1
        return dctList
        
    if switch == 0:
        result = Image()
    else:
        result = List()

    return result

def ImagePyramid(claheFinal):
    lowRes1 = cv2.pyrDown(claheFinal)
    lowRes2 = cv2.pyrDown(lowRes1)

    return lowRes1, lowRes2

def ImageDivider(blockSizes, claheFinal, lowRes1, lowRes2):
    imgList = [claheFinal, lowRes1, lowRes2]
    grey_levels = 256
    originalWindows = []
    lowResWindows = []
    lowestResWindows = []
    windowsList = [originalWindows, lowResWindows, lowestResWindows]
    x = 0

    for img in imgList:
        windowsize_r = blockSizes[x]
        windowsize_c = blockSizes[x]
            
        for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
            for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
                windowsList[x].append(img[r:r+windowsize_r,c:c+windowsize_c])
        x = x + 1

    return originalWindows, lowResWindows, lowestResWindows

def Entropy(originalWindows, lowResWindows, lowestResWindows):
    originalInformative = []
    lowResInformative = []
    lowestResInformative = []
    informativeLists = [originalInformative, lowResInformative, lowestResInformative]
    inputList = [originalWindows, lowResWindows, lowestResWindows]
    x = 0

    while x < 3:
        for img in inputList[x]:
            entropyImg = entropy(img, disk(6))
            entropyMean = np.mean(entropyImg)
            if 3 <= entropyMean <= 8:
                informativeLists[x].append(img)
        x = x + 1

    return originalInformative, lowResInformative, lowestResInformative

def TrainSVM(X, y):
    print("A")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
      
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
    print("A")
    dump(svm_model_linear, 'SVM JAFFE.joblib')

def SVM(x, y):
    svm_model_linear = load('SVM JAFFE.joblib')
    svm_predictions = svm_model_linear.predict(x)

    f = open("A.txt", "w")
    f.write(str(y))
    f.write(str(svm_predictions))
    f.close()
    
    accuracy = svm_model_linear.score(x, y)
    print(accuracy)
      
    cm = confusion_matrix(y, svm_predictions)
    print(cm)
    winsound.Beep(300, 150)
    
def Preprocess():
    i = 0
    for folder in folders2:
        x = 0
        for filename in os.scandir(directory2 + folder):
            original = cv2.imread(str(filename.path))
            claheFinal = CLAHE(original)
            lowRes1, lowRes2 = ImagePyramid(claheFinal)
            originalWindows, lowResWindows, lowestResWindows = ImageDivider(blockSizes, claheFinal, lowRes1, lowRes2)
            originalInformative, lowResInformative, lowestResInformative = Entropy(originalWindows, lowResWindows, lowestResWindows)
            dctOI = DiscreteCosineTransformation(originalInformative, 0)
            dctLRI = DiscreteCosineTransformation(lowResInformative, 0)
            dctLWRI = DiscreteCosineTransformation(lowestResInformative, 0)
            htInput = HaarTransformation(claheFinal)
            dctOutput = DiscreteCosineTransformation(htInput, 0)

            finalVector1 = []
            for lists in dctOutput:
                for element in lists:
                    finalVector1.append(element)

            finalVector2 = []
            dctOITMP = []
            dctLRITMP = []
            dctLWRITMP = []
            for lists in dctOI:
                for element in lists:
                    for ele in element:
                        dctOITMP.append(ele)

            for lists in dctLRI:
                for element in lists:
                    for ele in element:
                        dctLRITMP.append(ele)

            for lists in dctLWRI:
                for element in lists:
                    for ele in element:
                        dctLWRITMP.append(ele)

            while len(dctOITMP) < 50176:
                dctOITMP.append(0)

            finalVector2 = dctOITMP + dctLRITMP + dctLWRITMP
            finalVector = finalVector1 + finalVector2
            if i == 0:
                jaffeDatabase0.append(finalVector)
            if i == 1:
                jaffeDatabase1.append(finalVector)
            if i == 2:
                jaffeDatabase2.append(finalVector)
            if i == 3:
                jaffeDatabase3.append(finalVector)
            if i == 4:
                jaffeDatabase4.append(finalVector)
            if i == 5:
                jaffeDatabase5.append(finalVector)
            if i == 6:
                jaffeDatabase6.append(finalVector)
            jaffeTarget.append(i)
            print(x)
            x = x + 1
        print(i)
        i = i + 1

def writeToFileDB(filename, fTW0, fTW1, fTW2, fTW3, fTW4, fTW5, fTW6):
    array = []
    array.append(fTW0)
    array.append(fTW1)
    array.append(fTW2)
    array.append(fTW3)
    array.append(fTW4)
    array.append(fTW5)
    array.append(fTW6)
    i = 0
    for element in array:
        with open(filename + str(i) + ".txt", 'wb') as fp:
            pickle.dump(element, fp)
        i = i + 1

def writeToFileT(filename, fileToWrite):
    with open(filename, 'wb') as fp:
        pickle.dump(fileToWrite, fp)

def readFromFile(filename):
    with open (filename, 'rb') as fp:
        array = pickle.load(fp)
    return array

def loadDatabase(filename):
    i = 0
    tDB = []
    tDB2 = []
    while i < 7:
        tDB.append(readFromFile(filename + str(i) + ".txt"))
        i = i + 1
    tDB2 = tDB[0] + tDB[1] + tDB[2] + tDB[3] + tDB[4] + tDB[5] + tDB[6]
    return tDB2

#Preprocess()
#writeToFileDB("SVM - CK+ Database", jaffeDatabase0, jaffeDatabase1, jaffeDatabase2, jaffeDatabase3, jaffeDatabase4, jaffeDatabase5, jaffeDatabase6)
#writeToFileT("SVM - CK+ Targets.txt", jaffeTarget)
jaffeDatabase = loadDatabase("SVM - JAFFE Database" )
jaffeTarget = readFromFile("SVM - JAFFE Targets.txt")
TrainSVM(jaffeDatabase, jaffeTarget)
SVM(jaffeDatabase, jaffeTarget)
