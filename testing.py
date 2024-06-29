import motionInfuenceGenerator as mig
import createMegaBlocks as cmb
import numpy as np
import cv2
import os


def square(a):
    return a ** 2


def diff(l):
    return l[0] - l[1]


def showUnusualActivities(unusual, vid, noOfRows, noOfCols, n):
    unusualFrames = sorted(unusual.keys())

    print(unusualFrames)
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("Error: Unable to open video file:", vid)
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to read frame from video.")
        cap.release()
        return

    rows, cols = frame.shape[0], frame.shape[1]
    rowLength = rows / (noOfRows / n)
    colLength = cols / (noOfCols / n)
    print("Block Size ", (rowLength, colLength))
    count = 0

    screen_res = 980, 520
    scale_width = screen_res[0] / 320
    scale_height = screen_res[1] / 240
    scale = min(scale_width, scale_height)
    window_width = int(320 * scale)
    window_height = int(240 * scale)

    cv2.namedWindow('Unusual Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unusual Frame', window_width, window_height)

    while True:
        print(count)
        ret, uFrame = cap.read()
        if not ret:
            break

        if count in unusualFrames:
            for blockNum in unusual[count]:
                print(blockNum)
                x1 = int(blockNum[1] * rowLength)
                y1 = int(blockNum[0] * colLength)
                x2 = int((blockNum[1] + 1) * rowLength)
                y2 = int((blockNum[0] + 1) * colLength)
                cv2.rectangle(uFrame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            print("Unusual frame number ", str(count))
        cv2.imshow('Unusual Frame', uFrame)
        cv2.waitKey(0)

        count += 1


def constructMinDistMatrix(megaBlockMotInfVal, codewords, noOfRows, noOfCols, vid):
    threshold = 5.83682407063e-05
    n = 2
    minDistMatrix = np.zeros((len(megaBlockMotInfVal[0][0]), (noOfRows // n), (noOfCols // n)))

    for index, val in np.ndenumerate(megaBlockMotInfVal[..., 0]):
        eucledianDist = []
        for codeword in codewords[index[0]][index[1]]:
            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]), list(codeword)]
            eucDist = (sum(map(square, map(diff, zip(*temp))))) ** 0.5
            eucledianDist.append(eucDist)

        minDistMatrix[index[2]][index[0]][index[1]] = min(eucledianDist)

    unusual = {}
    for i in range(len(minDistMatrix)):
        if np.amax(minDistMatrix[i]) > threshold:
            unusual[i] = []
            for index, val in np.ndenumerate(minDistMatrix[i]):
                if val > threshold:
                    unusual[i].append((index[0], index[1]))

    print(unusual)
    showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)


def test_video(vid):
    print("Test video ", vid)
    if not os.path.exists(vid):
        print("Error: Video file does not exist:", vid)
        return

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("Error: Unable to open video file:", vid)
        return

    MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)
    if MotionInfOfFrames is None:
        print("Error: Unable to generate motion influence map.")
        cap.release()
        return

    megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save(
        "C:/Users/91800/Downloads/testMajor/testMajor/humanActivity/Dataset/videos/scene1/codewords_set1_p2_train_20-20_k5.npy",
        megaBlockMotInfVal)
    codewords = np.load(
        "C:/Users/91800/Downloads/testMajor/testMajor/humanActivity/Dataset/videos/scene1/codewords_set2_p1_train_20-20_k5.npy")
    print("codewords", codewords)
    if codewords is None:
        print("Error: Unable to load codewords.")
        cap.release()
        return

    constructMinDistMatrix(megaBlockMotInfVal, codewords, rows, cols, vid)
    cap.release()


if __name__ == '__main__':
    filePath = "C:/Users/91800/Downloads/testMajor/testMajor/humanActivity/Dataset/videos/scene2/2_test4.avi"
    testSet = [filePath]
    for video in testSet:
        test_video(video)
    print("Done")
