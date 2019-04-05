import os
import cv2
import glob
import csv
import numpy as np

def performAnnotations(image_path, gt_path, prototxt1, model1, prototxt2, model2, validImageShapes, defConfidence=0.2, showAnnotations=False):

    net1 = cv2.dnn.readNetFromCaffe(prototxt1, model1)
    net2 = cv2.dnn.readNetFromCaffe(prototxt2, model2)

    files = sorted(glob.glob(os.path.join(gt_path, '*.txt')))
    classes_gt = open(files.pop(0), "r")
    CLASSES_gt = classes_gt.read().splitlines()
    
    print('\nConfusion matrix -- \n')
    sumFN1, sumFN2 = 0, 0
    sumFP1 = 0
    sumFP2 = 0
    head = ['Image#', 'Image', 'GroundTruth_ file','False Positive\n(Fan\'s Model)','False Positive\n(MobileNet SSD)','False Negative\n(Fan\'s Model)','False Negative\n(MobileNet SSD)']
    with open('confusion_matrix.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(head)
    csvFile.close()

    for img_num, (file_, filename) in enumerate(zip(sorted(os.listdir(image_path)), files)):
#        if file_.endswith('.png'):
        file_ = os.path.join(image_path, file_)
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
        
        #Fetching gt
        boxes_gt = []
        gt_file=open(filename,"r")
        gt = gt_file.read().splitlines()
        gt_file.close()
        
        for lines in gt:
            coordinates = lines.split(" ")
            boxes_gt.append([float(i) for i in coordinates])

        CLASSES1 = ["background","body","head"]
        CLASSES2 = ["background","head", "body","wb"]
        predicted1 = []
        predicted2 = []
        selected_gt = []

        blob1 = cv2.dnn.blobFromImage(cv2.resize(image, (800, 200)), 0.007843, (800, 200), 127.5)
        boxes1 = runDetection(net1, CLASSES1, blob1, defConfidence, image, showAnnotations)
        blob2 = cv2.dnn.blobFromImage(cv2.resize(image, (1400, 300)), 0.007843, (1400, 300), 127.5)
        boxes2 = runDetection(net2, CLASSES2, blob2, defConfidence, image, showAnnotations)
        for c in boxes1:
            if c.pop(0) == 2:
                predicted1.append(c)
        for c in boxes2:
            if c.pop(0) == 1:
                predicted2.append(c)
        for c in boxes_gt:
            if c.pop(0) == 0:
                converted = toCoordinates(c)
                selected_gt.append(converted)

        selected_gt1 = selected_gt.copy()
        selected_gt2 = selected_gt.copy()

        #print('\nIntel model IOU!\n')
        fn1, fp1 = findingParameters(selected_gt1,predicted1,image)
        #print('\nSSD model IOU!\n')
        fn2, fp2 = findingParameters(selected_gt2,predicted2,image)

        #confusion matrix
        FN1 = len(fn1)
        FP1 = len(fp1)
        FN2 = len(fn2)
        FP2 = len(fp2)
        sumFN1 += FN1
        sumFN2 += FN2
        sumFP1 += FP1
        sumFP2 += FP2
        print(img_num, file_[5:],filename[17:],FP1,FP2,FN1,FN2)
        row = [img_num, file_[5:],filename[17:],FP1,FP2,FN1,FN2]

        with open('confusion_matrix.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        
        drawBoxes(fn1,image,(0,0,255),5)  #False Negative for model1 - Red
        drawBoxes(fp1,image,(255,0,0),5)  #False Positive for model1 - Blue
        drawBoxes(fn2,image,(0,255,0),3)  #False Negative for model2 - Green
        drawBoxes(fp2,image,(255,0,255),3)  #False Positive for model2 - Pink

        #scaled = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        labelfn1 = "False Negative for Fan\'s Model - Red"
        labelfn2 = "False Negative for MobileNet SSD - Green"
        labelfp1 = "False Positive for Fan\'s Model - Blue"
        labelfp2 = "False Positive for MobileNet SSD - Pink"
        scaled = cv2.resize(image, (1280, 540))
        cv2.putText(scaled, labelfn1, (930, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(scaled, labelfn2, (930, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(scaled, labelfp1, (930, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(scaled, labelfp2, (930, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imshow("Annotation",scaled)
        if cv2.waitKey(-1) == ord('q'): break

    result = ['Average','','',sumFP1/(img_num+1),sumFP2/(img_num+1),sumFN1/(img_num+1),sumFN2/(img_num+1)]
    with open('confusion_matrix.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(result)
    csvFile.close()

#dumpToFile(outputFilename, boxes)

def findingParameters(selected,predicted,image):
    sel_pred = []
    match=calcIOU(selected,predicted,image)
    for i,it in enumerate(predicted):
        for item in match:
            if item[1] == i:
                sel_pred.append(i)
                break

    sel = []
    for item in match:
        if item[1] != -1:
            sel.append(item[0])

    for i in sorted(sel_pred, reverse=True):
        del predicted[i]
    for i in sorted(sel, reverse=True):
        del selected[i]

    return selected, predicted

def toCoordinates(c):
    xmin = c[0] - (c[2]/2)
    xmax = c[0] + (c[2]/2)
    ymin = c[1] - (c[3]/2)
    ymax = c[1] + (c[3]/2)
            
    return [xmin, ymin, xmax, ymax]


def drawBoxes(boxes,img,color,width):
    h,w = img.shape[:2]
    for boxA in boxes:
        boxA = boxA * np.array([w, h, w, h])
        x1,y1,x2,y2 = boxA.astype("int")
        #        label = "{}: {:.2f}%".format(selected, iou * 100)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,width)
#        y = y1 - 15 if y1 - 15 > 15 else y1 + 15
#        cv2.putText(img, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

def calcIOU(boxes1, boxes2, image):
    res = []
    list(boxes1)
    list(boxes2)

    for idx, boxA in enumerate(boxes1):
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
        check = 0
        if(max(iou)>0.5):
            if len(res)>0:
                for item in res:
                    if item[1] == iou.index(max(iou)) and item[2] <= max(iou):
                        item[1] = -1
                        item[2] = -1
                        check = -1
                        res.append(list((idx,iou.index(max(iou)),max(iou))))
                        break
                    if item[1] == iou.index(max(iou)) and item[2] > max(iou):
                        check = -1
                        res.append(list((idx,-1,-1)))
                        break
                if(check!=-1):
                    res.append(list((idx,iou.index(max(iou)),max(iou))))
            else:
                res.append(list((idx,iou.index(max(iou)),max(iou))))
        else:
            res.append(list((idx,-1,-1)))

#    print('\nres - ')
#    print(*res, sep = " , ")
    # return the intersection over union value
    return res


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
    
    prototxt_1= '../ccg_headdet_800.cpu.prototxt.txt'
    model_1= '../ccg_headdet_800.caffemodel'
    prototxt_2= '../no_bnV2.prototxt.txt'
    model_2= '../no_bnV2.caffemodel'
    image_path_ = 'imgs'
    gt_path_ = 'imgs_annotations/'
    validImageShapes_ = ((1080, 3840), (1088, 3840))
    performAnnotations(image_path_, gt_path_, prototxt_1, model_1, prototxt_2, model_2, validImageShapes_, showAnnotations=False)
