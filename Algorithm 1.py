import os
import dlib
import cv2
import pickle
import statistics
import winsound
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils
from sklearn import datasets
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import tensorflow as tf

db1 = "JAFFE"
db2 = "CK+"
db3 = "NE"
db4 = "NE+"
wF = False
lF = True
database1= []
database2 = []
database3 = []
database4 = []
target1 = []
target2 = []
neutralFaceGen = []
topEmotions1 = []
topEmotions2 = []

directory1 = "F:\\University\\Y3\\S2\\Final Project\\Code\\JAFFE\\"
directory2 = "F:\\University\\Y3\\S2\\Final Project\\Code\\CK+\\"
directory3 = "F:\\University\\Y3\\S2\\Final Project\\Code\\JAFFE NE\\"
directory4 = "F:\\University\\Y3\\S2\\Final Project\\Code\\tmp\\db\\"
directory5 = "F:\\University\\Y3\\S2\\Final Project\\Code\\tmp\\ne\\"
folders = ["AN", "DI", "FE", "HA", "SA", "SU"]

database = []
target = []

def dlibCropFace(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dets = face_detector(grey, 1)
    
    for d in dets:
        cv2.rectangle(grey, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
        shape = predictor(grey, d)
        for i in range(shape.num_parts):
            p = shape.part(i)
            #cv2.circle(grey, (p.x, p.y), 2, 255, 1)
            #cv2.putText(grey, str(i), (p.x + 4, p.y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))

    crop_img = grey[d.top():d.bottom(), d.left():d.right()]
    dim = (128, 128)
    resized = cv2.resize(crop_img, dim)
    return resized

def bilateralFilter(img):
    bilateral = cv2.bilateralFilter(img, 85, 100, 100)
    return bilateral

def lbp(bilateralBlurred):
    radius = 1
    nPoints = 26
    lbpImge=local_binary_pattern(bilateralBlurred,nPoints,radius,method='default')
    return lbpImge

def apperanceFeatureCNN(database, target):
    loading = False
    training = True
    testOutput = True

    trainImages = []
    testImages = []
    
    (trainImages, testImages, trainLabels, testLabels) = train_test_split(database, target, test_size=0.2)

    #print(len(trainImages))
    #print(len(testImages))
    
    i = 0
    for img in trainImages:
        trainImages[i] = img / 255.0
        i = i + 1
    i = 0
    for img in testImages:
        testImages[i] = img / 255.0
        i = i + 1

    trainI = np.array(trainImages).reshape(len(trainImages), 128, 128, 1)
    testI = np.array(testImages).reshape(len(testImages), 128, 128, 1)
    trainL = np.array(trainLabels)
    testL = np.array(testLabels)

    if loading == False:
        model = models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=(5, 5), input_shape=(128, 128, 1)))
        model.add(layers.MaxPooling2D((2,2), input_shape=(64, 64, 1)))
        model.add(layers.Conv2D(128, kernel_size=(5, 5), input_shape=(64, 64, 1)))
        model.add(layers.MaxPooling2D((2,2), input_shape=(32, 32, 1)))
        model.add(layers.Conv2D(256, kernel_size=(3, 3), input_shape=(32, 32, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2), input_shape=(16, 16, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024))
        model.add(layers.Dense(500))
        model.add(layers.Dense(6))
        #model.summary()

        model.compile(optimizer="Adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    else:
        model = keras.models.load_model('ApperanceFeatureCNN')
    
    if training == True:
        history = model.fit(trainI, trainL, epochs=17, validation_data=(testI, testL), verbose = 0)
        model.save('ApperanceFeatureCNN')

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        #plt.show()

    if testOutput == True:
        results = []
        inp = np.array(database).reshape(len(database), 128, 128, 1)
        result = model.predict(inp)
        i = 0
        x = 0
        for element in result:
            highest = 0
            resultIndex = 0
            secondResultIndex = 0
            index = 0
            for idx in element:
                if idx > highest:
                    highest = idx
                    resultIndex = index
                index = index + 1
            highest = 0
            index = 0
            for idx in element:
                if index != resultIndex:
                    if idx > highest:
                        highest = idx
                        secondResultIndex = index
                index = index + 1                
            #print("---------")
            if resultIndex == target[i]:
                #print("Correct")
                x = x + 1
            #else:
                #print("Incorrect")
            results.append([resultIndex, secondResultIndex])
            i = i + 1
        #print("------------")
        print(x * 100 / i)
        confMatrix = []
        for element in results:
            confMatrix.append(element[0])
        confMatrix = tf.math.confusion_matrix(target, confMatrix)
        print(confMatrix)
        topEmotions1.append(results)
        winsound.Beep(300, 150)

def neutralAutoencoder(database, target):
    trainImages = []
    testImages = []
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    load = True
    train = False
    test = True

    trainImages = database;
    testImages = target;
    i = 0
    for img in trainImages:
        trainImages[i] = img.astype('float32') / 255.
        i = i + 1
    i = 0
    for img in testImages:
        testImages[i] = img.astype('float32') / 255.
        i = i + 1
    i = 0

    trainI = np.array(trainImages).reshape(len(trainImages), 128, 128, 1)
    testI = np.array(testImages).reshape(len(testImages), 128, 128, 1)
    inputShape = (len(trainI), 128, 128, 1)
    #print(tf.shape(trainI))

    encode = models.Sequential()
    decode = models.Sequential()
    iShape = keras.Input(shape=(128, 128, 1))
    model = models.Sequential()
    
    if load == False:
        # encoding
        model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(128,128,1), padding='same'))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096))
        model.add(layers.Dense(4096))
        model.add(layers.Dense(256))

        # decoding
        model.add(layers.Reshape((16, 16, 1)))
        model.add(layers.UpSampling2D(size=(2,2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.UpSampling2D(size=(2,2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.UpSampling2D(size=(2,2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
        model.compile(loss='MeanSquaredError', optimizer=opt)
        #model.summary()

    else:
        model = keras.models.load_model('Autoencoder')
        #model.load_weights('autoencoder-loss-0.0017.hdf5')
        #model.summary()
        #print("A")

    if train == True:
        checkpoint = tf.keras.callbacks.ModelCheckpoint("autoencoder-loss-{loss:.4f}.hdf5", monitor='loss', verbose=0, save_best_only=True, mode='min') 
        history = model.fit(trainI, testI, epochs=10, batch_size=20, verbose=1, callbacks=[checkpoint])
        print("A")
        model.save('Autoencoder')
        plt.plot(history.history['loss'], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 0.1])
        plt.show()
        print("A")

    if test == True:
        neutralFaceGen.append(model.predict(trainI, len(trainI)))
        winsound.Beep(300, 150)

def preprocessing(db, directory, folders):
    i = 0
    for folder in folders:
        x = 0
        for filename in os.scandir(directory + folder):
            original = cv2.imread(str(filename.path))
            croppedImage = dlibCropFace(original)
            bilateralBlurred = bilateralFilter(croppedImage)
            lbpImg = lbp(bilateralBlurred)
            if db == "JAFFE":
                database1.append(lbpImg)
                target1.append(i)
            if db == "CK+":
                database2.append(lbpImg)
                target2.append(i)
            if db == "NE":
                database3.append(croppedImage)
            if db == "NE+":
                database4.append(croppedImage)
            print(x)
            x = x + 1
        print(i)
        i = i + 1

def saveAndLoadAutoencoderOutput():
    i = 0
    while i < len(database):
        cv2.imwrite(directory4 + str(i) + ".png", database[i] * 255)
        i = i + 1
    database.clear()
    i = 0
    while i < len(neutralFaceGen[0]):
        cv2.imwrite(directory5 + str(i) + ".png", neutralFaceGen[0][i] * 255)
        i = i + 1
    neutralFaceGen.clear()

    db = []
    nfg = []
    for filename in os.scandir(directory4):
        db.append(cv2.imread(str(filename.path)))
    for filename in os.scandir(directory5):
        nfg.append(cv2.imread(str(filename.path)))
    return db, nfg

def faceLandmark(db, nfg):
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    dbImgs = []
    nfgImgs = []

    result1 = []
    result2 = []
    fResult = []

    for img in db:
        dbImgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    for img in nfg:
        nfgImgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    for img in dbImgs:
        dets = face_detector(img, 1)
        coords = np.zeros((68, 2), dtype="int")
        
        for d in dets:
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
            shape = predictor(img, d)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        result1.append(coords)

    for img in nfgImgs:
        dets = face_detector(img, 1)
        coords = np.zeros((68, 2), dtype="int")
        
        for d in dets:
            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
            shape = predictor(img, d)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        result2.append(coords)

    i = 0
    while i < len(result1):
        tmpEyebrows = []
        tmpEyes = []
        tmpLips = []
        x = 0
        while x < len(result1[i]):
            if x == 18 or x ==20 or x == 22 or x == 23 or x == 25 or x == 27:
                tmpEyebrows.append(abs(result1[i][x] - result2[i][x]))
            if x == 37 or x == 39 or x == 40 or x == 43 or x == 45 or x == 46:
                tmpEyes.append(abs(result1[i][x] - result2[i][x]))
            if x == 49 or x == 52 or x == 55 or x == 58:
                tmpLips.append(abs(result1[i][x] - result2[i][x]))
            x = x + 1
        i = i + 1
        tmpEb = []
        tmpEy = []
        tmpEyAvg = []
        tmpL = []
        tmpAbs = []
        tmpEb.append(tmpEyebrows[2][0])
        tmpEb.append(tmpEyebrows[2][1])
        tmpEb.append(tmpEyebrows[3][0])
        tmpEb.append(tmpEyebrows[3][1])
        tmpEb.append(tmpEyebrows[0][0])
        tmpEb.append(tmpEyebrows[0][1])
        tmpEb.append(tmpEyebrows[5][0])
        tmpEb.append(tmpEyebrows[5][1])
        tmpEb.append(tmpEyebrows[1][0])
        tmpEb.append(tmpEyebrows[1][1])
        tmpEb.append(tmpEyebrows[4][0])
        tmpEb.append(tmpEyebrows[4][1])
        tmpEy.append(tmpEyes[1][0])
        tmpEy.append(tmpEyes[1][1])
        tmpEy.append(tmpEyes[2][0])
        tmpEy.append(tmpEyes[2][1])
        tmpEy.append(tmpEyes[0][0])
        tmpEy.append(tmpEyes[0][1])
        tmpEy.append(tmpEyes[4][0])
        tmpEy.append(tmpEyes[4][1])
        tmpEy.append(tmpEyes[5][0])
        tmpEy.append(tmpEyes[5][1])
        tmpEy.append(tmpEyes[3][0])
        tmpEy.append(tmpEyes[3][1])
        tmpAbs = abs(abs(tmpEyes[0] - tmpEyes[2]) - tmpEyes[1]) - abs(tmpEyes[0] - tmpEyes[2])
        tmpEy.append(tmpAbs[0])
        tmpEy.append(tmpAbs[1])
        tmpAbs = abs(abs(tmpEyes[3] - tmpEyes[5]) - tmpEyes[4]) - abs(tmpEyes[3] - tmpEyes[5])
        tmpEy.append(tmpAbs[0])
        tmpEy.append(tmpAbs[1])
        tmpL.append(tmpLips[1][0])
        tmpL.append(tmpLips[1][1])
        tmpL.append(tmpLips[2][0])
        tmpL.append(tmpLips[2][1])
        tmpL.append(tmpLips[3][0])
        tmpL.append(tmpLips[3][1])
        tmpL.append(tmpLips[0][0])
        tmpL.append(tmpLips[0][1])
        fResult.append(tmpEb + tmpEy + tmpL)
    
    #print(result1[0])
    #print(result2[0])
    #print(fResult[0])
    #print(len(fResult[0]))
    return fResult

def geometricFeatureBasedCNN(db, targets):
    loading = False
    training = True
    testOutput = True

    trainVectors = []
    testVectors = []
    
    (trainVectors, testVectors, trainLabels, testLabels) = train_test_split(db, targets, test_size=0.2)

    #print(len(trainVectors))
    #print(len(testVectors))

    trainV = np.array(trainVectors).reshape(len(trainVectors), 36, 1, 1)
    testV = np.array(testVectors).reshape(len(testVectors), 36, 1, 1)
    trainL = np.array(trainLabels)
    testL = np.array(testLabels)

    if loading == False:
        model = models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=(3, 1), input_shape=(36, 1, 1)))
        model.add(layers.Conv2D(64, kernel_size=(3, 1)))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(128, kernel_size=(3, 1)))
        model.add(layers.Conv2D(128, kernel_size=(3, 1)))
        model.add(layers.Conv2D(128, kernel_size=(3, 1)))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(256, kernel_size=(3, 1)))
        model.add(layers.Conv2D(256, kernel_size=(3, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(500))
        model.add(layers.Dense(6))
        #model.summary()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    else:
        model = keras.models.load_model('GeometricFeatureBasedCNN')

    if training == True:
        history = model.fit(trainV, trainL, epochs=60, validation_data=(testV, testL), verbose = 0)
        model.save('GeometricFeatureBasedCNN')

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        #plt.show()
        
    if testOutput == True:
        results = []
        inp = np.array(db).reshape(len(db), 36, 1, 1)
        result = model.predict(inp)
        i = 0
        x = 0
        for element in result:
            highest = 0
            resultIndex = 0
            secondResultIndex = 0
            index = 0
            for idx in element:
                if idx > highest:
                    highest = idx
                    resultIndex = index
                index = index + 1
            highest = 0
            index = 0
            for idx in element:
                if index != resultIndex:
                    if idx > highest:
                        highest = idx
                        secondResultIndex = index
                index = index + 1                
            #print("---------")
            if resultIndex == targets[i]:
                #print("Correct")
                x = x + 1
            #else:
                #print("Incorrect")
            results.append([resultIndex, secondResultIndex])
            i = i + 1
        #print("------------")
        print(x * 100 / i)
        confMatrix = []
        for element in results:
            confMatrix.append(element[0])
        confMatrix = tf.math.confusion_matrix(targets, confMatrix)
        print(confMatrix)
        topEmotions2.append(results)
        winsound.Beep(300, 150)

def writeToFile(filename, fileToWrite):
    with open(filename, 'wb') as fp:
        pickle.dump(fileToWrite, fp)

def readFromFile(filename):
    with open (filename, 'rb') as fp:
        array = pickle.load(fp)
    return array

def calculateFinalResults(target):
    output = []
    i = 0
    while i < len(topEmotions1[0]):
        if topEmotions1[0][i][0] == topEmotions2[0][i][0]:
            output.append(topEmotions1[0][i][0])
        else:
            if topEmotions1[0][i][0] == topEmotions2[0][i][1]:
                output.append(topEmotions1[0][i][0])
            else:
                if topEmotions2[0][i][0] == topEmotions1[0][i][1]:
                    output.append(topEmotions2[0][i][0])
                else:
                    output.append(topEmotions1[0][i][0])
        i = i + 1

    i = 0
    x = 0
    for elements in output:
        #print("---------")
        if elements == target1[i]:
            #print("Correct")
            x = x + 1
        #else:
            #print("Incorrect")
        i = i + 1
    #print("------------")
    print(x * 100 / i)
    confMatrix = []
    for element in output:
        confMatrix.append(element)
    confMatrix = tf.math.confusion_matrix(target, confMatrix)
    print(confMatrix)

#preprocessing(db1, directory1, folders)
#preprocessing(db2, directory2, folders)
#preprocessing(db3, directory1, folders)
#preprocessing(db4, directory3, folders)
    
database = database1 + database2 + database3
target = target1 + target2

if wF == True:
    #writeToFile("ABCNN - JAFFE Database.txt", database)
    writeToFile("ABCNN - JAFFE Target.txt", target)
if lF == True:
    database = readFromFile("ABCNN - JAFFE Database.txt")
    #database4 = readFromFile("ABCNN - NE+ Database.txt")
    target1 = readFromFile("ABCNN - JAFFE Target.txt")
    target2 = readFromFile("ABCNN - JAFFE Target.txt")
    coordDiff = readFromFile("ABCNN - CD Database.txt")

#print(database[0])
#print(coordDiff[0])
    
apperanceFeatureCNN(database, target1)
#neutralAutoencoder(database, database4)
#db, nfg = saveAndLoadAutoencoderOutput()
#coordDiff = faceLandmark(db, nfg)
#writeToFile("ABCNN - CD Database.txt", coordDiff)
geometricFeatureBasedCNN(coordDiff, target2)
calculateFinalResults(target1)
