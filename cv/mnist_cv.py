import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt #графика
#from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

#нерабочая хуйня
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

path = './mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']  
x_test, y_test = f['x_test'], f['y_test']  
f.close()

path = './cifra.npz'
f = np.load(path)
ourx_test = f['arr_0']
f.close()

# стандартизация входных данных
x_train = x_train / 255 # делим на 255, так как в этих векторах минимальное значение - 0, максимальное - 255
x_test = x_test / 255
ourx_test = ourx_test/255

#перевод из такого типа:
#5
#в такой
#[0,0,0,0,0,1,0,0,0,0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
plt.show()

#создаем модель нейронной сети
model = keras.Sequential([
    #flatten - класс, который создает слой, который преобразовывает входную матрицу(28х28) в вектор(784 элемента)
    #1 - 1 байт(1 пиксель), представляющий собой значение от 0 до 255 (градация серого)
    Flatten(input_shape=(28, 28, 1)),
    #создаем слой нейронов полносвязной нейронной сети с помощью класса Dence
    #units - количество нейронов в скрытом слое(128)
    #Dense связывает все 128 нейронов скрытого слоя с входные значения(из flatten)
    Dense(112, activation='relu'),
    #связываем 10 нейронов со всеми нейронами скрытого слоя
    Dense(10, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль

#компилируем нейронную сеть оптимиз adam ф-ия потерь котегориальная кроссэнтропия(она лучше работает с softmax)
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])#метрика, которая нужна заказщику(показатель,нужный нам) 

#обучаем сеть, подавая на выход входное обуч мн-во, выходное обуч мн-во, размер батча(после каждых 32 изображений мы корректирем весовые кэфы)
#validation_split разбиваем обучающую выборку на обучающую и валидации (20% - выборка валид)

print("начало обучения")
model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2,verbose = 0)
print("конец обучения\n")


#plt.imshow(ourx_test, cmap=plt.cm.binary)
#plt.show()
x = np.expand_dims(ourx_test,axis=0)
print(x)
res = model.predict(x)
print("Распознанная цифра: ",np.argmax(res))
#plt.imshow(ourx_test, cmap=plt.cm.binary)
#plt.show()

print("Критерий кач-ва loss и метрика качества accuracy:\n")
model.evaluate(x_test, y_test_cat)#считаем критерий  качества и метрику accuracy
print("")


#считываем 1 цифру

'''n = 0
#axis = 0
#типо model.predict(x) на вход должен получать несколько изображений, поэтому, когда мы подаем 1 двумерный массив мы добавляем еще 3 измерение
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res )#10 выходов, которая дала нейронная сеть
print( "Распознанная цифра: ",np.argmax(res) )#выводим индекс максимального значения среди этих выходов

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()'''


# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
# pred - вектор с 10000 значениями, которые являются результатом работы нейронки

#print(pred.shape)

#выводим первые 20 цифр
print(pred[:10])
print(y_test[:10])

# Выделение неверных вариантов из первых 20ти цифр
mask = pred == y_test
print(mask[:10])

#считаем сколько неправильных ответов дала сеть
x_false = x_test[~mask]
y_false = pred[~mask]
 
print(x_false.shape)
a = int(input("Введи число и сеть покажет первую попавшуюся картинку с этим числом:"))
while a<10:
    index = np.where(pred == a)
    x = int(index[0][0])
    plt.imshow(x_test[x], cmap=plt.cm.binary)
    plt.show()
    a = int(input("Введи число и сеть покажет первую попавшуюся картинку с этим числом:"))
print("Где ошиблась сеть\n")

# Вывод первых 25 неверных результатов
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)
plt.show()
for i in range(5):
    print("Значение сети: " + str(y_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
    
