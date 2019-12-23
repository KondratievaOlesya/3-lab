import numpy as np
import cv2
import json
from math import exp, pi, sqrt, cos, sin

COLS = 1
ROWS = 0


def getL(M, N):
    simpleNumbers = [2, 3, 5, 7, 11, 13,
                     17, 19, 23, 29, 31,
                     37, 41, 43, 47, 53,
                     59, 61, 67, 71, 73,
                     79, 83, 89, 97, 101,
                     103, 107, 109, 113,
                     127, 131
                     ]
    for l in range(2, 400):
        m = int(M / l)
        n = int(N / l)
        if (m * n) <= 400 and m >= 14 and n >= 12:
            return l, m, n


def scScale(img, m, n):
    M, N = img.shape
    # l, m, n = getL(M, N)
    # m = int(M/l)
    # n = int(N/l)
    result = np.zeros((m + 1, n + 1))
    l = max(int(M / m), int(N / n))
    for i in range(0, M, l):
        for j in range(0, N, l):
            sum = 0
            count = 0
            for k in range(i, min(i + l, M)):
                for h in range(j, min(j + l, N)):
                    sum += img[k, h]
                    count += 1
            result[int(i / l), int(j / l)] = sum / count
    return result


def DFT(img, p):
    M, N = img.shape
    FpmCos = np.zeros((p, M))
    FpmSin = np.zeros((p, M))
    FnpCos = np.zeros((N, p))
    FnpSin = np.zeros((N, p))
    for i in range(0, p):
        for j in range(0, M):
            FpmCos[i][j], FpmSin[i][j] = cos(2 * pi / M * i * j), sin(2 * pi / M * i * j)

    for i in range(0, N):
        for j in range(0, p):
            FnpCos[i][j], FnpSin[i][j] = cos(2 * pi / N * i * j), sin(2 * pi / N * i * j)
    FXcos = (FpmCos.dot(img))
    FXsin = (FpmSin.dot(img))
    Creal = FXcos.dot(FnpCos) - FXsin.dot(FnpSin)
    Cimag = FXcos.dot(FnpSin) + FXsin.dot(FnpCos)
    tmp = np.square(Creal) + np.square(Cimag)
    C = np.sqrt(tmp)
    return C


def DCT(img, p):
    M, N = img.shape
    Tpm = np.zeros((p, M))
    Tnp = np.zeros((N, p))

    for j in range(0, M):
        Tpm[0, j] = 1 / sqrt(M)
    for i in range(1, p):
        for j in range(0, M):
            Tpm[i, j] = sqrt(2 / M) * cos((pi * (2 * j + 1) * i) / (2 * M))

    for i in range(0, N):
        Tnp[i, 0] = 1 / sqrt(N)
    for i in range(0, N):
        for j in range(0, p):
            Tnp[i, j] = sqrt(2 / N) * cos((pi * (2 * i + 1) * j) / (2 * N))

    C = (Tpm.dot(img)).dot(Tnp)
    return C


def histogram(img, BIN):
    Hi = [0 for _ in range(256)]
    M, N = img.shape
    for i in range(0, M):
        for j in range(0, N):
            Hi[img[i, j]] += 1

    Hb = [0 for _ in range(BIN)]
    for i in range(0, BIN):
        for j in range(int(i * 256 / BIN), int((i + 1) * 256 / BIN)):
            Hb[i] += Hi[j]
    HbNorm = [Hb[i] / (M * N) for i in range(BIN)]
    return [Hi, Hb, HbNorm]


def gradient(img, W, S, type=COLS):
    M, N = img.shape
    result = []

    if type == COLS:
        lastRow = img[0:W]
        for i in range(S, M - W + 1, S):
            row = img[i:(i + W)]
            diff = abs(np.linalg.norm(lastRow - row))
            lastRow = row
            result.append(diff)
        return result
    elif type == ROWS:
        lastCol = img[:, 0:W]
        for i in range(S, N - W, S):
            col = img[:, i:i + W]
            diff = abs(np.linalg.norm(lastCol - col))
            lastCol = col
            result.append(diff)
        return result


def distance(test, template):
    return abs(np.linalg.norm(np.array(test) - np.array(template)))


dataToWrite = []
# Обучение
for pDFT in range(2, 6):
    f = open('templates1.json', 'w')
    dataSet = []
    #m, n, pDFT, pDCT, BIN, S, W = [37, 30, 10, 16, 32, 3, 12]
    for typeOfFaceNum in range(1, 2):
        for faceNum in range(1, 41):
            img = cv2.imread('orl_faces/s' + str(faceNum) + '/' + str(typeOfFaceNum) + '.pgm')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray1 = np.float32(gray)
            # scale = np.uint8(scScale(gray, m, n))
            # dct = DCT(gray, pDCT)
            dft = DFT(gray, pDFT)
            # Hi, Hb, HbNorm = histogram(gray, BIN)
            #grad = gradient(gray, W, S)
            data = {
                'img_num': faceNum,
                'img': img.tolist(),
                # 'scale': scale.tolist(),
                'dft': dft.tolist(),
                #   'dct': dct.tolist(),
                #   'Hi': Hi,
                #   'Hb': Hb,
                #  'HbNorm': HbNorm,
                #'grad': grad
            }
            dataSet.append(data)
    f.write(json.dumps(dataSet))

    # Тестирование
    f = open('templates1.json', 'r')
    resFile = open('pdft1.txt', 'w')
    dataSet = json.loads(f.read())
    #m, n, pDFT, pDCT, BIN, S, W = [37, 30, 10, 16, 32, 3, 12]
    # m = int(112/l)
    # n = int(92/l)
    correct = 0
    totalCorrect = 0
    totalAmauntOfTestImages = 0
    for typeOfFaceNum in range(2, 11):
        correct = 0
        for faceNum in range(1, 41):
            img = cv2.imread('orl_faces/s' + str(faceNum) + '/' + str(typeOfFaceNum) + '.pgm')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray1 = np.float32(gray)
            # scale = np.uint8(scScale(gray, m, n))

            # dct = DCT(gray, pDCT)
            dft = DFT(gray, pDFT)
            # Hi, Hb, HbNorm = histogram(gray, BIN)
            # grad = gradient(gray, W, S)
            minNormD = 9999999999999999
            totalAmauntOfTestImages += 1

            for setCount in range(0, len(dataSet)):
                set = dataSet[setCount]
                d = [
                          distance(dft, set['dft']),
                    #      distance(grad, set['grad']),
                    # distance(scale, set['scale']),
                    #    distance(dct, set['dct']),
                    #    distance(HbNorm, set['HbNorm']),
                    #distance(grad, set['grad'])
                ]
                normD = np.linalg.norm(d)
                if normD < minNormD:
                    minNormD = normD
                    pict = np.array(set['img'])
                    resNum = set['img_num']
            if resNum == faceNum:

                correct += 1
        totalCorrect += correct

    dataToWrite.append({
        'Количество изображений в обучающей выборке': c * 40,
        # 'l' : int(112/m),
        # 'm': m,
        # 'n': n,
        # 'pDCT': pDCT,
        # 'pDCT': pDCT,
        # 'BIN': BIN,
        # 'w': W,
        # 's': S,
        'Результат': totalCorrect / totalAmauntOfTestImages * 100,
    })
resFile.write(json.dumps(dataToWrite))
