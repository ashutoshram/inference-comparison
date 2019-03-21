import os
import cv2
import numpy as np


def performAnnotations(path, prototxt, model, validImageShapes, defConfidence=0.2, showAnnotations=False):

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    for file_ in os.listdir(path):
        if file_.endswith('.png'):
            file_ = os.path.join(path, file_)
            outputFilename = os.path.splitext(file_)[0]
            outputFilename = outputFilename+'.txt'
            #read in image file and prepare file for detection
            #print('file = %s, outputFilename = %s' % (file_, outputFilename))

            image = cv2.imread(file_)

            if image is None:
                print('file was not loaded', file_)
                continue

            if not image.shape[:2] in validImageShapes:
                print('%s: invalid image size: %dx%d' % (file_, w, h)) 
                continue

            blob = cv2.dnn.blobFromImage(cv2.resize(image, (800, 200)), 0.007843, (800, 200), 127.5)
            boxes = runDetection(net, blob, defConfidence, image, showAnnotations)
            dumpToFile(outputFilename, boxes)
                   

def convertToYolo(box):
    (class_, startX, startY, endX, endY) = box
    relW = endX - startX 
    relH = endY - startY 
    relX = relW/2
    relY = relH/2
    return (class_, relX, relY, relW, relH)


def dumpToFile(outputFilename, boxes):
    with open(outputFilename, 'w') as output:
        print('dumping to %s' % outputFilename)
        for box in boxes:
            box = convertToYolo(box)
            line = ' '.join(map(str, box))
            output.write(line + '\n')


def runDetection(net, blob, defConfidence, image, showAnnotations):

    boxes = []

    CLASSES = ["background", "body", "head"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    #set up network for detection
    net.setInput(blob)
    #infer using network parameters
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < defConfidence: continue

        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7]
        (startX, startY, endX, endY) = box.astype("float")
        print('%d: class = %s, box = %s' % (i, CLASSES[idx], repr(box.astype('float'))))
        boxes.append([idx, startX, startY, endX, endY]) 

        if showAnnotations:
            h,w = image.shape[:2]
            box = box * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            #print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            # make sure we do not putText beyond image boundary
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    if showAnnotations:
        scaled = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('annotated', scaled)
        cv2.waitKey(30)
    return boxes

if __name__ == "__main__":

    prototxt_= 'ccg_headdet_800.cpu.prototxt.txt'
    model_= 'ccg_headdet_800.caffemodel'
    path_ = ''
    validImageShapes_ = ((1080, 3840), (1088, 3840))
    performAnnotations(path_, prototxt_, model_, validImageShapes_, showAnnotations=False)
