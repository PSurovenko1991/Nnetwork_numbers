import numpy as np
from keras.layers import Dense  #слои
from keras.models import Sequential #модель
from keras.utils import np_utils # утилиты для массивов
from keras.datasets import mnist #dataset



np.random.seed(42) # 


(x_train, y_train),(x_test, y_test) = mnist.load_data()


# Formalization:
#print(x_train.shape)(60000,28,28)
x_train = x_train.reshape(60000, 784)  
#print(x_train.shape)(60000, 784)
x_train = x_train.astype("float32") # интенсивность пикселей
x_train /= 255
#print(x_train[10])

y_train = np_utils.to_categorical(y_train, 10) # приводим к категориальному виду




model = Sequential() # создаем модель
# добавляем слои:
model.add(Dense(800, input_dim = 784, init = "normal", activation="relu")) # слой Dense - все нейроны соеденены со всеми следующего слоя
                                                                            # init - распределение весов
                                                                            # Relu(activation): f(x) = max(0,x), softplas: f(x)=ln(1+e^x)
model.add(Dense(10, init = "normal", activation="softmax")) # softmax - сумма значений всех нейронов =1, хорошо подходит для категориального распределения
# компилируем модель:
model.compile(loss = "categorical_crossentropy", optimizer = "SGD", metrics = ["accuracy"]) # loss - мера ошибки,
                                                                                            # optimizer - метод обучени, SGD - стохастический градиентный спуск
# основные параметры модели:                                                                                            # metrics (точность)
print(model.summary())

# Обучаем модель:
model.fit(x_train, y_train, validation_split=0.2 batch_size=200, nb_epoch = 30, verbose=1) # batch_size - кол-во обьектов после которых происходит определение направления градиентов и корректировка весов
                                                                        # nb_epoch - кол-во эпох обучения
                                                                        # verbose - отображение инф. во время обучения
                                                                        #validation_split - размер проверочной выборки, используемой, в процессе обучения
        
        
prediction = model.predict(x_train) # работа модели

prediction = prediction.round(0) # определяем лидирующий нейрон, подходит так как использован "softmax" 
# print(prediction[:15])
# print(y_train[:15])        
        
        

# качество обучения:        
        
        
x_test = x_test.reshape(10000,784)
x_test = x_test.astype("float32") 
x_test /= 255
y_test = np_utils.to_categorical(y_test, 10)
#print(x_test.shape)
#print(y_test.shape)
score = model.evaluate(x_test,y_test) # оценка качества обучения
print("!!!!!!!!!!!!!!!!!!!!!!!")
print(score[1]*100)
