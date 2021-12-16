import math
import random
import sys

from PIL import Image
import numpy as np


class StructuringElement:
    elements = None
    width = 0
    height = 0
    origin = None
    ignoreElements = None

    def __init__(self, width=None, height=None, origin=None):
        self.width = width

        self.height = height
        if (origin.real < 0 or origin.real >= width or origin.imag < 0 or origin.imag >= height):
            self.origin = complex(0, 0)
        else:
            self.origin = origin
        self.elements = np.zeros([width, height])
        self.ignoreElements = []


class ImageManager2:

    def __init__(self, width=None, height=None, bitDepth=None, img=None, data=None, origin=None):
        self.width = width
        self.height = height
        self.bitDepth = bitDepth
        self.img = img
        self.data = data
        self.origin = origin

    def read(self, fileName):
        self.img = Image.open(fileName)
        self.data = np.array(self.img)
        self.original = np.copy(self.data)
        self.width = self.data.shape[0]
        self.height = self.data.shape[1]
        mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32,
                       "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
        bitDepth = mode_to_bpp[self.img.mode]

        print(
            "Image %s with %s x %s pixels (%s bits per pixel)  data has been read!" % (
                self.img.filename, self.width, self.height, bitDepth))

    def write(self, fileName):
        img = Image.fromarray(self.data)
        try:
            img.save(fileName)
        except:
            print("Write file error")
        else:
            print("Image %s has been written!" % (fileName))

    def restoreToOriginal(self):
        self.data = np.copy(self.original)

    def rgb2gray(self):
        for x in range(len(self.data)):
            for y in range(len(self.data[x])):
                graysclae = (self.data[x, y, 0] * 0.299) + (self.data[x, y, 1] * 0.587) + (self.data[x, y, 2] * 0.114)
                self.data[x, y] = graysclae

    def convertToBlue(self):
        for y in range(self.height):
            for x in range(self.width):
                self.data[x, y, 0] = 0
                self.data[x, y, 1] = 0
                self.data[x, y, 1] = 0

    def dilation(self, se):
        self.rgb2gray()
        data_zeropaded = np.zeros([self.width + se.width * 2, self.height + se.height * 2, 3])
        data_zeropaded[se.width - 1:self.width + se.width - 1, se.height - 1:self.height + se.height - 1, :] = self.data
        for y in range(se.height - 1, se.height + self.height - 1):
            for x in range(se.width - 1, se.width + self.width - 1):
                subData = data_zeropaded[x - int(se.origin.real):x - int(se.origin.real) + se.width,
                          y - int(se.origin.imag):y - int(se.origin.imag) + se.height, 0:1]
                subData = subData.reshape(3, -1)

                for point in se.ignoreElements:
                    subData[int(point.real), int(point.imag)] = se.elements[int(point.real), int(point.imag)]
                max = np.amax(se.elements[se.elements > 0])
                subData = np.subtract(subData, np.flip(se.elements))
                if (0 <= x - int(se.origin.real) - 1 < self.width and 0 <= y - int(se.origin.imag) - 1 < self.height):
                    if (0 in subData):
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = max
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = max
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = max
                    else:
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = 0
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = 0
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = 0

    def erosion(self, se):
        self.rgb2gray()
        data_zeropaded = np.zeros([self.width + se.width * 2, self.height + se.height * 2, 3])
        data_zeropaded[se.width - 1:self.width + se.width - 1, se.height - 1:self.height + se.height - 1, :] = self.data
        for y in range(se.height - 1, se.height + self.height - 1):
            for x in range(se.width - 1, se.width + self.width - 1):
                subData = data_zeropaded[x - int(se.origin.real):x - int(se.origin.real) + se.width,
                          y - int(se.origin.imag):y - int(se.origin.imag) + se.height, 0:1]
                subData = subData.reshape(3, -1)

                for point in se.ignoreElements:
                    subData[int(point.real), int(point.imag)] = se.elements[int(point.real), int(point.imag)]
                min = np.amin(se.elements[se.elements > 0])

                if (0 <= x - int(se.origin.real) - 1 < self.width and 0 <= y - int(se.origin.imag) - 1 < self.height):
                    if (np.array_equal(subData, se.elements)):
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = min
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = min
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = min
                    else:
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = 0
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = 0
                        self.data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = 0

    def boundaryExtraction(self, se):
        self.erosion(se)
        temp = self.data

        self.restoreToOriginal()
        self.rgb2gray()
        for x in range(len(self.data)):
            for y in range(len(self.data[x])):
                boundex = (self.data[x, y, 0] - temp[x, y, 0]) + (self.data[x, y, 1] - temp[x, y, 1]) + (self.data[x, y, 2] - temp[x, y, 2])
                self.data[x, y] = boundex

    def thresholding(self, threshold):
        self.rgb2gray()
        for y in range(self.height):
            for x in range(self.width):
                gray = self.data[x, y, 0]
                gray = 0 if gray < threshold else 255
                self.data[x, y, 0] = gray
                self.data[x, y, 1] = gray
                self.data[x, y, 2] = gray

    def otsuThreshold(self):
        self.rgb2gray()
        histogram = np.zeros(256)

        for y in range(self.height):
            for x in range(self.width):
                histogram[self.data[x, y, 0]] += 1

        histogramNorm = np.zeros(len(histogram))
        pixelNum = self.width * self.height

        for i in range(len(histogramNorm)):
            histogramNorm[i] = histogram[i] / pixelNum

        histogramCS = np.zeros(len(histogram))
        histogramMean = np.zeros(len(histogram))

        for i in range(len(histogramNorm)):
            if (i == 0):
                histogramCS[i] = histogramNorm[i]
                histogramMean[i] = 0
            else:
                histogramCS[i] = histogramCS[i - 1] + histogramNorm[i]
                histogramMean[i] = histogramMean[i - 1] + histogramNorm[i] * i

        globalMean = histogramMean[len(histogramMean) - 1]
        max = sys.float_info.min
        maxVariance = sys.float_info.min
        countMax = 0

        for i in range(len(histogramCS)):
            if (histogramCS[i] < 1 and histogramCS[i] > 0):
                variance = ((globalMean * histogramCS[i] - histogramMean[i]) ** 2) / (histogramCS[i] * (1 - histogramCS[i]))
            if (variance > maxVariance):
                maxVariance = variance
                max = i
                countMax = 1
            elif (variance == maxVariance):
                countMax = countMax + 1
                max = ((max * (countMax - 1)) + i) / countMax
        self.thresholding(round(max))

    def linearSpatialFilter(self, kernel, size):
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return

        data_zeropaded = np.zeros([self.width + int(size / 2) * 2, self.height + int(size / 2) * 2, 3])
        data_zeropaded[int(size / 2):self.width + int(size / 2), int(size / 2):self.height + int(size / 2), :] = self.data

        for y in range(int(size / 2), int(size / 2) + self.height):
            for x in range(int(size / 2), int(size / 2) + self.width):
                subData = data_zeropaded[x - int(size / 2):x + int(size / 2) + 1,y - int(size / 2):y + int(size / 2) + 1, :]

                sumRed = np.sum(np.multiply(subData[:, :, 0:1].flatten(), kernel))
                sumGreen = np.sum(np.multiply(subData[:, :, 1:2].flatten(), kernel))
                sumBlue = np.sum(np.multiply(subData[:, :, 2:3].flatten(), kernel))

                sumRed = 255 if sumRed > 255 else sumRed
                sumRed = 0 if sumRed < 0 else sumRed

                sumGreen = 255 if sumGreen > 255 else sumGreen
                sumGreen = 0 if sumGreen < 0 else sumGreen

                sumBlue = 255 if sumBlue > 255 else sumBlue
                sumBlue = 0 if sumBlue < 0 else sumBlue

                self.data[x - int(size / 2), y - int(size / 2), 0] = sumRed
                self.data[x - int(size / 2), y - int(size / 2), 1] = sumGreen
                self.data[x - int(size / 2), y - int(size / 2), 2] = sumBlue

    def cannyEdgeDetector(self, lower, upper):
        # Step 1 - Apply 5 x 5 Gaussian filter

        gaussian = [2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0,
                    4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
                    5.0 / 159.0, 12.0 / 159.0, 15.0 / 159.0, 12.0 / 159.0, 5.0 / 159.0,
                    4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
                    2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0]

        self.linearSpatialFilter(gaussian, 5)
        self.rgb2gray()

        # Step 2 - Find intensity gradient
        sobelX = [1, 0, -1,
                  2, 0, -2,
                  1, 0, -1]
        sobelY = [1, 2, 1,
                  0, 0, 0,
                  -1, -2, -1]

        magnitude = np.zeros([self.width, self.height])
        direction = np.zeros([self.width, self.height])

        data_zeropaded = np.zeros([self.width + 2, self.height + 2, 3])
        data_zeropaded[1:self.width + 1, 1:self.height + 1, :] = self.data

        for y in range(1, self.height + 1):
            for x in range(1, self.width + 1):
                gx = 0
                gy = 0

                subData = data_zeropaded[x - 1:x + 2, y - 1:y + 2, :]

                gx = np.sum(np.multiply(subData[:, :, 0:1].flatten(), sobelX))
                gy = np.sum(np.multiply(subData[:, :, 0:1].flatten(), sobelY))

                magnitude[x - 1, y - 1] = math.sqrt(gx * gx + gy * gy)
                direction[x - 1, y - 1] = math.atan2(gy, gx) * 180 / math.pi

        # Step 3 - Nonmaxima Suppression
        gn = np.zeros([self.width, self.height])
        for y in range(3, self.height - 3):
            for x in range(3, self.width - 3):
                targetX = 0
                targetY = 0

                # find closest direction
                if (direction[x - 1, y - 1] <= -157.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= -112.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= -67.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= -22.5):
                    targetX = 1
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 22.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= 67.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= 112.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 157.5):
                    targetX = 1
                    targetY = 1
                else:
                    targetX = 1
                    targetY = 0

                print(targetX,targetY)

                if (y + targetY >= 0 and y + targetY < self.height and x + targetX >= 0 and x + targetX < self.width and magnitude[x - 1, y - 1] < magnitude[x + targetY - 1, y + targetX - 1]):
                    gn[x - 1, y - 1] = 0

                elif (y - targetY >= 0 and y - targetY < self.height and x - targetX >= 0 and x - targetX < self.width and magnitude[x - 1, y - 1] < magnitude[x - targetY - 1, y - targetX - 1]):
                    gn[x - 1, y - 1] = 0

                else:
                    gn[x - 1, y - 1] = magnitude[x - 1, y - 1]

                # set back first
                gn[x - 1, y - 1] = 255 if gn[x - 1, y - 1] > 255 else gn[x - 1, y - 1]
                gn[x - 1, y - 1] = 0 if gn[x - 1, y - 1] < 0 else gn[x - 1, y - 1]

                self.data[x - 1, y - 1, 0] = gn[x - 1, y - 1]
                self.data[x - 1, y - 1, 1] = gn[x - 1, y - 1]
                self.data[x - 1, y - 1, 2] = gn[x - 1, y - 1]

        # Step 4 - Hysteresis Thresholding
        # upper threshold checking with recursive
        for y in range(self.height):
            for x in range(self.width):
                if (self.data[x, y, 0] >= upper):
                    self.data[x, y, 0] = 255
                    self.data[x, y, 1] = 255
                    self.data[x, y, 2] = 255

                    self.hystConnect(x, y, lower)

        # clear unwanted values
        for y in range(self.height):
            for x in range(self.width):
                if (self.data[x, y, 0] != 255):
                    self.data[x, y, 0] = 0
                    self.data[x, y, 1] = 0
                    self.data[x, y, 2] = 0


    def hystConnect(self, x, y, threshold):
        for i in range(y - 1, y + 2):
            for j in range(x - 1, x + 2):
                if ((j < self.width) and (i < self.height) and (j >= 0) and (i >= 0) and (j != x) and (i != y)):
                    value = self.data[j, i, 0]
                    if (value != 255):
                        if (value >= threshold):
                            self.data[j, i, 0] = 255
                            self.data[j, i, 1] = 255
                            self.data[j, i, 2] = 255

                        else:
                            self.data[j, i, 0] = 0
                            self.data[j, i, 1] = 0
                            self.data[j, i, 2] = 0

    def houghTransform(self, percent):
        # The image should be converted to edge map first
        # Work out how the hough space is quantized
        numOfTheta = 720
        thetaStep = math.pi / numOfTheta
        highestR = int(round(max(self.width, self.height) * math.sqrt(2)))
        centreX = int(self.width / 2)
        centreY = int(self.height / 2)
        print("Hough array w: %s height: %s" % (numOfTheta, (2 * highestR)))
        # Create the hough array and initialize to zero
        houghArray = np.zeros([numOfTheta, 2 * highestR])
        # Step 1 - find each edge pixel
        # Find edge points and vote in array
        for y in range(3, self.height - 3):
            for x in range(3, self.width - 3):
                pointColor = self.data[x, y, 0]

                if (pointColor != 0):
                # Edge pixel found
                    for i in range(numOfTheta):
                        # Step 2 - Apply the line equation and update hough array
                        # Work out the r values for each theta step
                        r = int((x - centreX) * math.cos(i * thetaStep) + (y - centreY) * math.sin(i * thetaStep))

                        # Move all values into positive range for display purposes
                        r = r + highestR
                        if (r < 0 or r >= 2 * highestR):
                            continue
                        # Increment hough array
                        houghArray[i, r] = houghArray[i, r] + 1

        maxHough = np.amax(houghArray)
        threshold = percent * maxHough
        # Step 4 - Draw lines
        # Search for local peaks above threshold to draw
        for i in range(numOfTheta):
            for j in range(2 * highestR):
                # only consider points above threshold
                if (houghArray[i, j] >= threshold):
                    # see if local maxima
                    draw = True
                    peak = houghArray[i, j]

                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            # not seeing itself
                            if (k == 0 and l == 0):
                                continue
                            testTheta = i + k
                            testOffset = j + l
                            if (testOffset < 0 or testOffset >= 2 * highestR):
                                continue
                            if (testTheta < 0):
                                testTheta = testTheta + numOfTheta
                            if (testTheta >= numOfTheta):
                                testTheta = testTheta - numOfTheta
                            if (houghArray[testTheta][testOffset] > peak):
                                # found bigger point
                                draw = False
                                break

                    # point found is not local maxima
                    if (not (draw)):
                        continue

                    # if local maxima, draw red back
                    tsin = math.sin(i * thetaStep)
                    tcos = math.cos(i * thetaStep)

                    if (i <= numOfTheta / 4 or i >= (3 * numOfTheta) / 4):
                        for y in range(self.height):
                    # vertical line

                            x = int((((j - highestR) - ((y - centreY) * tsin)) / tcos) + centreX)

                            if (x < self.width and x >= 0):
                                self.data[x, y, 0] = 255
                                self.data[x, y, 1] = 0
                                self.data[x, y, 2] = 0

                    else:
                        for x in range(self.width):
                            # horizontal line

                            y = int((((j - highestR) - ((x - centreX) * tcos)) / tsin) + centreY)

                        if (y < self.height and y >= 0):
                            self.data[x, y, 0] = 255
                            self.data[x, y, 1] = 0
                            self.data[x, y, 2] = 0

    def ADIAbsolute(self, sequences, threshold, step):

            data_temp = np.zeros([self.width,self.height, 3])
            data_temp = np.copy(self.data)
            self.data[self.data > 0] = 0
            for n in range(len(sequences)):
                # read file
                otherImage = Image.open(sequences[n])
                otherData = np.array(otherImage)
                for y in range(self.height):
                    for x in range(self.width):
                        dr = int(data_temp[x, y, 0]) - int(otherData[x, y, 0])
                        dg = int(data_temp[x, y, 1]) - int(otherData[x, y, 1])
                        db = int(data_temp[x, y, 2]) - int(otherData[x, y, 2])
                        dGray = int(round((0.2126 * dr) + int(0.7152 * dg) + int(0.0722 * db)))
                        if (abs(dGray) > threshold):
                            newColor = self.data[x, y, 0] + step
                            newColor = 255 if newColor > 255 else newColor
                            newColor = 0 if newColor < 0 else newColor

                            self.data[x, y, 0] = newColor
                            self.data[x, y, 1] = newColor
                            self.data[x, y, 2] = newColor