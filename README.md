# CompVision
Программа для оценки рисков кредитования по психофизическому портрету клиента
Программа оценивающей риски кредитования клиента финансовых организаций и предоставляющей менеджеру рекомендации. Для оценки рисков используется два метода психометрического анализа на основе лицевых данных, определение типа по типологии MBTI с помощью композитных изображений составленных из лиц людей с определенной психологической функцией данной типологии и “Алгоритм психодиагностики по асимметрии лица” А.Н Ануашвили. Поиск лиц на изображении осуществляется с помощью гистограмм направленных градиентов. Преобразования изображения для подготовки перед обработкой алгоритмами и вычисления асимметрии лица для алгоритма Ануашвили происходит с использованием активных моделей внешнего вида. Сравнение лица тестируемого человека с композитными лицами происходит с помощью сверточной нейронной сети, обученной по алгоритму FaceNet. 

main -Это главный файл в нем содержатся все функции, по нахождению и обработке лиц и ключевых точек.
Form3- файл в котором содержится код пользовательского интерфейса.

jpg файлы- нужны для работы алгоритма определения предладполагаемого психологическо типа :D

Подключены 3 нейронные сети:

predictor - нахождения 68 ключевых антропометрически точек лица.

facerec - вычисляет дескриптор(число) с помощью которого можно сравнивать лица на похожесть

detector - встроенная нейронная сеть библиотеки dlib по нахождения лиц на изображениия, работает на основе метода HOG(Гистограммы направленных градиентов)
