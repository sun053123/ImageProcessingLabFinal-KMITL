import math
import random

from PIL import Image
import numpy as np


class ImageManager:

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

    def convertToGreen(self):
        for y in range(self.height):
            for x in range(self.width):
                self.data[x, y, 0] = 0
                self.data[x, y, 2] = 0

    def convertToBlue(self):
        for y in range(self.height):
            for x in range(self.width):
                self.data[x, y, 0] = 0
                self.data[x, y, 1] = 0

    def rgb2gray(self):
        for x in range(len(self.data)):
            for y in range(len(self.data[x])):
                graysclae = (self.data[x, y, 0] * 0.299) + (self.data[x, y, 1] * 0.587) + (self.data[x, y, 2] * 0.114)
                self.data[x, y] = graysclae

    def adjustBrightness(self, brightness):

        for y in range(self.height):
            for x in range(self.width):
                r = self.data[x, y, 0]
                g = self.data[x, y, 1]
                b = self.data[x, y, 2]
                r = r + brightness
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r
                g = g + brightness
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g
                b = b + brightness
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b
                self.data[x, y, 0] = r
                self.data[x, y, 1] = g
                self.data[x, y, 2] = b

    def adjustGamma(self, Gamma):
        for y in range(self.height):
            for x in range(self.width):
                r = self.data[x, y, 0]
                g = self.data[x, y, 1]
                b = self.data[x, y, 2]

                r = pow(r / 255, Gamma) * 255
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = pow(g / 255, Gamma) * 255
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g

                b = pow(b / 255, Gamma) * 255
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b

                self.data[x, y, 0] = r
                self.data[x, y, 1] = g
                self.data[x, y, 2] = b

    def invert(self):
        for y in range(self.height):
            for x in range(self.width):
                r = self.data[x, y, 0]
                g = self.data[x, y, 1]
                b = self.data[x, y, 2]
                r = 255 - r
                g = 255 - g
                b = 255 - b
                self.data[x, y, 0] = r
                self.data[x, y, 1] = g
                self.data[x, y, 2] = b

    def grayscaleHistogramEqualisation(self):
        histogram = np.array([0] * 256)

        for y in range(self.height):
            for x in range(self.width):
                r = self.data[x, y, 0]
                g = self.data[x, y, 1]
                b = self.data[x, y, 2]

                gray = int((0.2126 * r) + int(0.7152 * g) + int(0.0722 * b))

                histogram[gray] += 1

        histogramCDF = np.array([0] * 256)
        cdfMin = 0

        for i in range(len(histogram)):
            if (i == 0):
                histogramCDF[i] = histogram[i]
            else:
                histogramCDF[i] = histogramCDF[i - 1] + histogram[i]
            if (histogram[i] > 0 and cdfMin == 0):
                cdfMin = i

        for y in range(self.height):
            for x in range(self.width):
                r = self.data[x, y, 0]
                g = self.data[x, y, 1]
                b = self.data[x, y, 2]

                gray = (int)((0.2126 * r) + int(0.7152 * g) + int(0.0722 * b))
                gray = (int)(round(255.0 * (histogramCDF[gray] - cdfMin) / (self.width * self.height - cdfMin)))
                gray = 255 if gray > 255 else gray
                gray = 0 if gray < 0 else gray

                self.data[x, y, 0] = gray
                self.data[x, y, 1] = gray
                self.data[x, y, 2] = gray

    def setTemperature(self, rTemp, gTemp, bTemp):
        for y in range(self.height):
            for x in range(self.width):
                r = self.data[x, y, 0]
                g = self.data[x, y, 1]
                b = self.data[x, y, 2]

                r *= (rTemp / 255.0)
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g *= (gTemp / 255.0)
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g

                b *= (bTemp / 255.0)
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b

                self.data[x, y, 0] = r
                self.data[x, y, 1] = g
                self.data[x, y, 2] = b

    def getGrayscaleHistogram(self):
        self.rgb2gray()

        histogram = np.array([0] * 256)
        for y in range(self.height):
            for x in range(self.width):
                histogram[self.data[x, y, 0]] += 1
        self.restoreToOriginal()
        return histogram

    def getContrast(self):
        contrast = 0.0

        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = self.width * self.height

        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i
        avgIntensity /= pixelNum
        for y in range(self.height):
            for x in range(self.width):
                contrast += (self.data[x, y, 0] - avgIntensity) ** 2
        contrast = (contrast / pixelNum) ** 0.5
        return contrast

    def adjustContrast(self, contrast):
        currentContrast = self.getContrast()
        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = self.width * self.height

        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i

        avgIntensity /= pixelNum
        min = avgIntensity - currentContrast
        max = avgIntensity + currentContrast

        newMin = avgIntensity - currentContrast - contrast / 2
        newMax = avgIntensity + currentContrast + contrast / 2

        newMin = 0 if newMin < 0 else newMin
        newMax = 0 if newMax < 0 else newMax
        newMin = 255 if newMin > 255 else newMin
        newMax = 255 if newMax > 255 else newMax

        if (newMin > newMax):
            temp = newMax
            newMax = newMin
            newMin = temp

        contrastFactor = (newMax - newMin) / (max - min)

        for y in range(self.height):
            for x in range(self.width):
                r = self.data[x, y, 0]
                g = self.data[x, y, 1]
                b = self.data[x, y, 2]
                contrast += (self.data[x, y, 0] - avgIntensity) ** 2
                r = (int)((r - min) * contrastFactor + newMin)
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r
                g = (int)((g - min) * contrastFactor + newMin)
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g
                b = (int)((b - min) * contrastFactor + newMin)
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b
                self.data[x, y, 0] = r
                self.data[x, y, 1] = g
                self.data[x, y, 2] = b

    def median_filter(self, kernel):
        temp = []
        index = kernel // 2
        data_final = []
        data_final = np.zeros((len(self.data), len(self.data[0])))

        for i in range(len(self.data)):

            for j in range(len(self.data[0])):

                for z in range(kernel):
                    if i + z - index < 0 or i + z - index > len(self.data) - 1:
                        for c in range(kernel):
                            temp.append(0)
                    else:
                        if j + z - index < 0 or j + index > len(self.data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(kernel):
                                temp.append(self.data[i + z - index][j + k - index])

                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
                self.data = data_final

    def addSaltNoise(self, percent):
        noOfPX = self.height * self.width
        noiseAdded = (int)(percent * noOfPX)
        whiteColor = 255

        for i in range(noiseAdded):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.data[x, y, 0] = whiteColor
            self.data[x, y, 1] = whiteColor
            self.data[x, y, 2] = whiteColor

    def addPepperNoise(self, percent):
        noOfPX = self.height * self.width
        noiseAdded = (int)(percent * noOfPX)
        blackColor = 0

        for i in range(noiseAdded):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.data[x, y, 0] = blackColor
            self.data[x, y, 1] = blackColor
            self.data[x, y, 2] = blackColor

    def addUniformNoise(self, percent, distribution):
        noOfPX = self.height * self.width
        noiseAdded = (int)(percent * noOfPX)

        for i in range(noiseAdded):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            gray = self.data[x, y, 0]
            gray += (random.randint(0, distribution * 2 - 1) - distribution)
            gray = 255 if gray > 255 else gray
            gray = 0 if gray < 0 else gray
            self.data[x, y, 0] = gray
            self.data[x, y, 1] = gray
            self.data[x, y, 2] = gray

    def addsaltpeppernoise(self,x,y):
        self.addSaltNoise(x)
        self.addPepperNoise(y)

    def contraharmonicFilter(self, size, Q):
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return

        data_temp = np.zeros([self.width, self.height, 3])
        data_temp = self.data.copy()
        for y in range(self.height):
            for x in range(self.width):
                sumRedAbove = 0
                sumGreenAbove = 0
                sumBlueAbove = 0
                sumRedBelow = 0
                sumGreenBelow = 0
                sumBlueBelow = 0
                subData = data_temp[x - int(size / 2):x + int(size / 2) + 1, y - int(size / 2):y + int(size / 2) + 1,
                          :].copy()
                subData = subData ** (Q + 1)
                sumRedAbove = np.sum(subData[:, :, 0:1], axis=None)
                sumGreenAbove = np.sum(subData[:, :, 1:2], axis=None)
                sumBlueAbove = np.sum(subData[:, :, 2:3], axis=None)
                subData = data_temp[x - int(size / 2):x + int(size / 2) + 1, y - int(size / 2):y + int(size / 2) + 1,:].copy()
                subData = subData ** Q
                sumRedBelow = np.sum(subData[:, :, 0:1], axis=None)
                sumGreenBelow = np.sum(subData[:, :, 1:2], axis=None)
                sumBlueBelow = np.sum(subData[:, :, 2:3], axis=None)
                if (sumRedBelow != 0): sumRedAbove /= sumRedBelow
                sumRedAbove = 255 if sumRedAbove > 255 else sumRedAbove
                sumRedAbove = 0 if sumRedAbove < 0 else sumRedAbove
                if (math.isnan(sumRedAbove)): sumRedAbove = 0
                if (sumGreenBelow != 0): sumGreenAbove /= sumGreenBelow
                sumGreenAbove = 255 if sumGreenAbove > 255 else sumGreenAbove
                sumGreenAbove = 0 if sumGreenAbove < 0 else sumGreenAbove
                if (math.isnan(sumGreenAbove)): sumGreenAbove = 0
                if (sumBlueBelow != 0): sumBlueAbove /= sumBlueBelow
                sumBlueAbove = 255 if sumBlueAbove > 255 else sumBlueAbove
                sumBlueAbove = 0 if sumBlueAbove < 0 else sumBlueAbove
                if (math.isnan(sumBlueAbove)): sumBlueAbove = 0

                self.data[x, y, 0] = sumRedAbove
                self.data[x, y, 1] = sumGreenAbove
                self.data[x, y, 2] = sumBlueAbove

    def unsharpmask(self):

        kernel = [[1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9]]

        amount = 2

        # Middle of the kernel
        offset = len(kernel) // 2

        for x in range(offset, self.width - offset):
            for y in range(offset, self.height - offset):
                original_pixel = self.data[x, y]
                acc = [0, 0, 0]
                for a in range(len(kernel)):
                    for b in range(len(kernel)):
                        xn = x + a - offset
                        yn = y + b - offset
                        pixel = self.data[xn, yn]
                        acc[0] += pixel[0] * kernel[a][b]
                        acc[1] += pixel[1] * kernel[a][b]
                        acc[2] += pixel[2] * kernel[a][b]

                    new_pixel = (
                        int(original_pixel[0] + (original_pixel[0] - acc[0]) * amount),
                        int(original_pixel[1] + (original_pixel[1] - acc[1]) * amount),
                        int(original_pixel[2] + (original_pixel[2] - acc[2]) * amount)
                    )
                    self.data[x, y] = new_pixel

    def resizeNearestNeighbour(self, scaleX, scaleY):
        newWidth = (int)(round(self.width * scaleX))
        newHeight = (int)(round(self.height * scaleY))
        data_temp = np.zeros([self.width, self.height, 3])
        data_temp = self.data.copy()
        data = np.resize(self.data, [newWidth, newHeight, 3])
        for y in range(newHeight):
            for x in range(newWidth):
                xNearest = (int)(round(x / scaleX))
                yNearest = (int)(round(y / scaleY))
                xNearest = self.width - 1 if xNearest >= self.width else xNearest
                xNearest = 0 if xNearest < 0 else xNearest
                yNearest = self.height - 1 if yNearest >= self.height else yNearest
                yNearest = 0 if yNearest < 0 else yNearest
                self.data[x, y, :] = data_temp[xNearest, yNearest, :]

    def resizeBilinear(self, scaleX, scaleY):
        newWidth = (int)(round(self.width * scaleX))
        newHeight = (int)(round(self.height * scaleY))
        data_temp = np.zeros([self.width, self.height, 3])
        data_temp = self.data.copy()
        self.data = np.resize(self.data, [newWidth, newHeight, 3])
        for y in range(newHeight):
            for x in range(newWidth):
                oldX = x / scaleX
                oldY = y / scaleY
                # get 4 coordinates
                x1 = min((int)(np.floor(oldX)), self.width - 1)
                y1 = min((int)(np.floor(oldY)), self.height - 1)
                x2 = min((int)(np.ceil(oldX)), self.width - 1)
                y2 = min((int)(np.ceil(oldY)), self.height - 1)
                # get colours
                color11 = np.array(data_temp[x1, y1, :])
                color12 = np.array(data_temp[x1, y2, :])
                color21 = np.array(data_temp[x2, y1, :])
                color22 = np.array(data_temp[x2, y2, :])
                # interpolate x
                P1 = (x2 - oldX) * color11 + (oldX - x1) * color21
                P2 = (x2 - oldX) * color12 + (oldX - x1) * color22
                if x1 == x2:
                    P1 = color11
                    P2 = color22

                # interpolate y
                P = (y2 - oldY) * P1 + (oldY - y1) * P2

                if y1 == y2:
                    P = P1

                P = np.round(P)

                self.data[x, y, :] = P