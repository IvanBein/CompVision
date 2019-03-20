import dlib
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
plt.rcParams['figure.figsize'] = 16, 13
import cv2
from math import sqrt
from math import acos
from math import cos
from math import sin
from math import radians
from math import pow
from math import fabs
from math import floor
from math import degrees
from imutils import face_utils
from scipy import ndimage
import numpy as np
from PIL import Image
import sys
from skimage import io
from scipy.spatial import distance
predictor= dlib.shape_predictor('C:\Test\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('C:\Test\dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

# Импортируем наш интерфейс из файла
from Form3 import *
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QApplication,QFileDialog)
from PyQt5.QtGui import QPixmap


class MyWin(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global img_in
        # Здесь прописываем событие нажатия на кнопку
        self.ui.pushButton.clicked.connect(self.loading)
        self.ui.pushButton_2.clicked.connect(self.MainFunction)

    def loading(self): #загрузка изображения
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        hbox = QHBoxLayout(self)
        pixmap = QPixmap(fname)
        pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        self.ui.label.setPixmap(pixmap)
        self.ui.pushButton_2.setDisabled(False)

    def Webloading(self): #загрузка изображения с вебкамеры
        cap = cv2.VideoCapture(0)
        # "Прогреваем" камеру, чтобы снимок не был тёмным
        for i in range(30):
            cap.read()
        # Делаем снимок
        ret, frame = cap.read()
        # Записываем в файл
        cv2.imwrite('cam.jpg', frame)
        # Отключаем камеру
        cap.release()
        fname = np.array(Image.open('cam.jpg'))
        hbox = QHBoxLayout(self)
        pixmap = QPixmap(fname)
        pixmap = pixmap.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        self.ui.label.setPixmap(pixmap)
        self.ui.pushButton_2.setDisabled(False)
		
	def FuncAttribute(img): # функция вычасления признаков для лиц
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        for k, d in enumerate(dets):
            shape = predictor(gray, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return face_descriptor
	#Нанесение антропометрических точек		
	def ShapeFunc
		img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        rects = detector(img_gray, 0)
        shape = predictor(img_gray, rects[0])
        shape = face_utils.shape_to_np(shape)
		return shape

    def rotate(A, B, C):
        return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
		
	def CalcNasFold(dot2,dot3,dot4)
		rx = dot3[0] - dot2[0]
        ry = dot3[1] - dot2[1]
        dot4[0] = dot2[0] + rx * cos(radians(-125)) - ry * sin(radians(-125))
        dot4[1] = dot2[1] + rx * sin(radians(-125)) + ry * cos(radians(-125))
        lenAB = sqrt((dot4[0] - dot2[0]) ** 2 + (dot4[1] - dot2[1]) ** 2)
        dot4[0] = dot4[0] + (dot4[0] - dot2[0]) / lenAB * 50
        dot4[1] = dot4[1] + (dot4[1] - dot2[1]) / lenAB * 50
        dot2 = shape[35].tolist()
        dot3 = shape[54].tolist()
        a = sqrt((dot2[0] - dot4[0]) ** 2 + (dot2[1] - dot4[1]) ** 2)
        c = sqrt((dot2[0] - dot3[0]) ** 2 + (dot2[1] - dot3[1]) ** 2)
        b = sqrt((dot3[0] - dot4[0]) ** 2 + (dot3[1] - dot4[1]) ** 2)
        bOb = degrees(acos((pow(a, 2) + pow(c, 2) - pow(b, 2)) / (2 * a * c)))
		if rotate(dot2, dot4, dot3) > 0:
			bOb = -1 * bl
		return bOb
		
	def func(dot1, dot2): #функция рассчета угла отклонения ( но не для носогубных складок)
		#  print("Гипотенуза")
		Hypotenuse = sqrt((dot1[0] - dot2[0]) ** 2 + (dot1[1] - dot2[1]) ** 2)
		# print("Катет b")
		Cathetusb = sqrt((dot1[0] - dot2[0]) ** 2)
		# print("Катет a")
		Cathetusa = sqrt((dot1[1] - dot2[1]) ** 2)
		#  print("Угол")
		return degrees(acos((pow(Hypotenuse, 2) + pow(Cathetusb, 2) - pow(Cathetusa, 2)) / (2 * Hypotenuse * Cathetusb)))

    def MainFunction(self):
	#сравнение лица тестируемого человека с "усредненным лицом" полученным из лиц людей определенного психологического типа по MBTI
        img_input = np.array(Image.open(fname))
        if self.ui.radioButton.isChecked():
            face_descriptor1 = FuncAttribute(img_input)

            img = np.array(Image.open('IntM.jpg'))
            face_descriptor2 = FuncAttribute(img)
            IntM = distance.euclidean(face_descriptor1, face_descriptor2)

            img = np.array(Image.open('ExtM.jpg'))
            face_descriptor2 = FuncAttribute(img)
            ExtM = distance.euclidean(face_descriptor1, face_descriptor2)
            if IntM < ExtM:
                self.ui.lineEdit.setText("Интроверт: 1")
            else:
                self.ui.lineEdit.setText("Экстраверт: 0")

            img = np.array(Image.open('JudM.jpg'))
            face_descriptor2 = FuncAttribute(img)
            JudM = distance.euclidean(face_descriptor1, face_descriptor2)

            img = np.array(Image.open('PerM.jpg'))
            face_descriptor2 = FuncAttribute(img)
            PerM = distance.euclidean(face_descriptor1, face_descriptor2)
            if JudM < PerM:
                self.ui.lineEdit_2.setText("Иррационал: 0")
            else:
                self.ui.lineEdit_2.setText("Рационал: 1")

        if self.ui.radioButton_2.isChecked():
            face_descriptor1 = FuncAttribute(img_input)

            img = np.array(Image.open('IntW.jpg'))
            face_descriptor2 = FuncAttribute(img)
            IntM = distance.euclidean(face_descriptor1, face_descriptor2)

            img = np.array(Image.open('ExtW.jpg'))
            face_descriptor2 = FuncAttribute(img)
            ExtM = distance.euclidean(face_descriptor1, face_descriptor2)
            self.ui.lineEdit.setText("Интроверт: 1") if IntM < ExtM else self.ui.lineEdit.setText("Экстраверт: 0")

            img = np.array(Image.open('JudW.jpg'))
            face_descriptor2 = FuncAttribute(img)
            JudM = distance.euclidean(face_descriptor1, face_descriptor2)

            img = np.array(Image.open('PerW.jpg'))
            face_descriptor2 = FuncAttribute(img)
            PerM = distance.euclidean(face_descriptor1, face_descriptor2)
            self.ui.lineEdit_2.setText("Иррационал: 0") if JudM < PerM else self.ui.lineEdit_2.setText("Рационал: 1")

#часть кода отвечающая за разворот фотографии до положения когда лицо будет смотреть прямо
		shape=ShapeFunc()
        a = shape[27].tolist()
        b = shape[30].tolist()
        c = shape[31].tolist()
        d = shape[35].tolist()
        while a[0] - b[0] != 0 or c[1] - d[1] != 0:
            while (a[0] - b[0] != 0):
                if fabs(a[0] - b[0]) > 3:
                    if a[0] - b[0] > 0:
                        img_input = ndimage.rotate(img_input, 1)
                    else:
                        img_input = ndimage.rotate(img_input, -1)
                else:
                    if a[0] - b[0] > 0:
                        img_input = ndimage.rotate(img_input, 0.1)
                    else:
                        img_input = ndimage.rotate(img_input, -0.1)
                print(fabs(a[0] - b[0]))				
				shape=ShapeFunc()
                a = shape[27].tolist()
                b = shape[30].tolist()
            while (c[1] - d[1] != 0):
                if fabs(c[1] - d[1]) > 3:
                    if c[1] - d[1] > 0:
                        img_input = ndimage.rotate(img_input, -1)
                    else:
                        img_input = ndimage.rotate(img_input, 1)
                else:
                    if c[1] - d[1] > 0:
                        img_input = ndimage.rotate(img_input, -0.1)
                    else:
                        img_input = ndimage.rotate(img_input, 0.1)
                print(fabs(c[1] - d[1]))
				shape=ShapeFunc()
				#рисует антропометрические точки на выводимом изображении
                img_tmp = img_input.copy()
                for x, y in shape:
                    cv2.circle(img_tmp, (x, y), 1, (0, 0, 255), -1)
                c = shape[31].tolist()
                d = shape[35].tolist()
#################################################################################################################################

       # print("Левый глаз")
        dot1 = shape[42].tolist()
        dot2 = shape[45].tolist()
        if dot1[1] > dot2[1]:
            al = func(dot1, dot2)
        else:
            al = -1 * func(dot1, dot2)
      #  print(al)
       # print("Правый глаз")
        dot1 = shape[39].tolist()
        dot2 = shape[36].tolist()
        if dot1[1] > dot2[1]:
            ar = func(dot1, dot2)
        else:
            ar = -1 * func(dot1, dot2)
      #  print(ar)
       # print("Левая носагубная складка")
        dot2 = shape[35].tolist()
        dot3 = shape[33].tolist()
        dot3[1] = dot2[1]
        dot4 = shape[35].tolist()
		bl=CalcNasFold(dot2,dot3,dot4)
       # print("bl ", bl)
        cv2.line(img_tmp, (dot2[0], dot2[1]), (floor(dot4[0]), floor(dot4[1])), (0, 0, 255), 2)
       # print("Правая носагубная складка")
        dot2 = shape[31].tolist()
        dot3 = shape[33].tolist()
        dot3[1] = dot2[1]
        dot4 = shape[35].tolist()
		br=CalcNasFold(dot2,dot3,dot4)
      #  print("br ", br)
        cv2.line(img_tmp, (dot2[0], dot2[1]), (floor(dot4[0]), floor(dot4[1])), (0, 0, 255), 2)
       # print("Левый кончик губы")
        dot1 = shape[62].tolist()
        dot2 = shape[54].tolist()
        if dot1[1] > dot2[1]:
            Yl = func(dot1, dot2)
        else:
            Yl = -1 * func(dot1, dot2)
      #  print(Yl)
       # print("Правый кончик губы")
        dot1 = shape[62].tolist()
        dot2 = shape[48].tolist()
        if dot1[1] > dot2[1]:
            Yr = func(dot1, dot2)
        else:
            Yr = -1 * func(dot1, dot2)
      #  Расчет коэффициентов
        Da = ar - al
        Db = br - bl
        DY = Yr - Yl
        Sa = (ar - al) / 2
        Sb = (br - bl) / 2
        SY = (Yr - Yl) / 2
        plt.imshow(img_tmp)
        dom = 0
        if Da <= -3.5:
            dom = dom - 2
        elif -3.5 < Da <= 0:
            dom = dom - 1
        elif 0 < Da <= 3.5:
            dom = dom + 1
        elif Da > 3.5:
            dom = dom + 2

        if Db <= -3.5:
            dom = dom - 2
        elif -3.5 < Db <= 0:
            dom = dom - 1
        elif 0 < Db <= 3.5:
            dom = dom + 1
        elif Db > 3.5:
            dom = dom + 2

        if DY <= -3.5:
            dom = dom - 2
        elif -3.5 < DY <= 0:
            dom = dom - 1
        elif 0 < DY <= 3.5:
            dom = dom + 1
        elif DY > 3.5:
            dom = dom + 2

        sv = 0
        if Sa < -1:
            sv = sv + 2
        elif -1 <= Sa < 0:
            sv = sv + 1
        elif 0 <= Sa < 1:
            sv = sv - 1
        elif Sa >= 1:
            sv = sv - 2

        if Sb < -1:
            sv = sv + 2
        elif -1 <= Sb < 0:
            sv = sv + 1
        elif 0 <= Sb < 1:
            sv = sv - 1
        elif Sb >= 1:
            sv = sv - 2

        if SY < -1:
            sv = sv + 2
        elif -1 <= SY < 0:
            sv = sv + 1
        elif 0 <= SY < 1:
            sv = sv - 1
        elif SY >= 1:
            sv = sv - 2
        print(dom)
        print(sv)
        if dom>0:
            self.ui.lineEdit_3.setText("Правое: 0")
        else:
            self.ui.lineEdit_3.setText("Левое: 1")
        if sv > 0:
            self.ui.lineEdit_4.setText("Cтабильность: 1")
        else:
            self.ui.lineEdit_4.setText("Дестабильность: 0")
        self.ui.lineEdit_5.setText("2")
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    sys.exit(app.exec_( ))