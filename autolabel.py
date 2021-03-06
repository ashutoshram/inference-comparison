import os
import cv2
import numpy as np


def performAnnotations(path, prototxt1, model1, prototxt2, model2, validImageShapes, defConfidence=0.2, showAnnotations=False):

    net1 = cv2.dnn.readNetFromCaffe(prototxt1, model1)
    net2 = cv2.dnn.readNetFromCaffe(prototxt2, model2)
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
        
            CLASSES1 = ["background","body","head"]
            CLASSES2 = ["background","head", "body","wb"]
            selected1 = []
            selected2 = []
            selected = 'head'
            
            blob1 = cv2.dnn.blobFromImage(cv2.resize(image, (800, 200)), 0.007843, (800, 200), 127.5)
            boxes1 = runDetection(net1, CLASSES1, blob1, defConfidence, image, showAnnotations)
            blob2 = cv2.dnn.blobFromImage(cv2.resize(image, (1400, 300)), 0.007843, (1400, 300), 127.5)
            boxes2 = runDetection(net2, CLASSES2, blob2, defConfidence, image, showAnnotations)
            for c in boxes1:
                if c.pop(0) == 2:
                    selected1.append(c)
            for c in boxes2:
                if c.pop(0) == 1:
                    selected2.append(c)
            
#            print('\nBoxes1 - ')
#            print(*selected1, sep = "\n")
#            print('\nBoxes2 - ')
#            print(*selected2, sep = "\n")
            m=calcIOU(selected1,selected2,image)
            drawBoxes(selected1,selected2,image,m, selected)

            #scaled = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            scaled = cv2.resize(image, (1280, 540))
            cv2.imshow("Annotation",scaled)
            if cv2.waitKey(-1) == ord('q'): break

#dumpToFile(outputFilename, boxes)

def drawBoxes(boxes1,boxes2,img,m,selected):
    
    h,w = img.shape[:2]
    for (boxA,iou) in zip(boxes1,m):
        boxA = boxA * np.array([w, h, w, h])
        x1,y1,x2,y2 = boxA.astype("int")
        label = "{}: {:.2f}%".format(selected, iou * 100)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
        y = y1 - 15 if y1 - 15 > 15 else y1 + 15
        cv2.putText(img, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    for boxB in boxes2:
        boxB = boxB * np.array([w, h, w, h])
        x1,y1,x2,y2 = boxB.astype("int")
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),5)


def calcIOU(boxes1, boxes2, image):
    idx = []
    m = []
    list(boxes1)
    list(boxes2)
    
    for boxA in boxes1:
        iou = []
        for boxB in boxes2:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
    
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou.append(interArea / float(boxAArea + boxBArea - interArea))
        idx.append(iou.index(max(iou)))
        m.append(max(iou))


#        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
#        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
#        y = startY - 15 if startY - 15 > 15 else startY + 15
#        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

#    print('\nIdx - ')
#    print(*idx, sep = ",")
#    print('\niou - ')
#    print(*m, sep = ",")
    # return the intersection over union value
    return m


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


def runDetection(net, CLASSES, blob, defConfidence, image, showAnnotations):

    boxes = []
    
    #CLASSES = ["background","head", "body","wb"]
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
#        print('%d: class = %s, box = %s' % (i, CLASSES[idx], repr(box.astype('float'))))
#        if(CLASSES[idx] == "head"):
#            boxes.append([startX, startY, endX, endY])
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
        cv2.waitKey(200)
    return boxes

if __name__ == "__main__":
    
    prototxt_1= 'ccg_headdet_800.cpu.prototxt.txt'
    model_1= 'ccg_headdet_800.caffemodel'
    prototxt_2= 'no_bnV2.prototxt.txt'
    model_2= 'no_bnV2.caffemodel'
    path_ = 'imgs'
    validImageShapes_ = ((1080, 3840), (1088, 3840))
    performAnnotations(path_, prototxt_1, model_1, prototxt_2, model_2, validImageShapes_, showAnnotations=False)
