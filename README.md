# CompVision
main -Это главный файл в нем содержатся все функции, по нахождению и обработке лиц и ключевых точек.
Form3- файл в котором содержится код пользовательского интерфейса.

jpg файлы- нужны для работы алгоритма определения предладполагаемого психологическо типа :D

Подключены 3 нейронные сети:

predictor - нахождения 68 ключевых антропометрически точек лица.

facerec - вычисляет дескриптор(число) с помощью которого можно сравнивать лица на похожесть

detector - встроенная нейронная сеть библиотеки dlib по нахождения лиц на изображениия, работает на основе метода HOG(Гистограммы направленных градиентов)
