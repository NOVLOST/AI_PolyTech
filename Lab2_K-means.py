import pandas as pd
from sklearn.cluster import KMeans

import math
import time
import random as r
list_city = []
distance_matrix = []
line = ['0','0']
k_max = int(input("Введите кол-во класстеров "))
sum_x = 0
sum_y = 0
value_city = 0
index_city = 0
with open('city2.csv') as file: #читаем файл с городами
    while len(line) != 1 :
        line=file.readline().split(',')

        try:
            list_city.append((index_city,int(line[1]),int(line[2][0:-1]))) #при формировании списка городов заменяем название города на индекс
            index_city +=1
        except:
            pass


    for first_name,fx,fy in list_city: # формируем матрицу смежности
        row_matrix = []
        sum_x += fx
        sum_y += fy
        value_city += 1
        for second_name,sx,sy in list_city:
                x = fx - sx
                y = fy - sy


                distance = (x**2)+(y**2)
                distance = round(math.sqrt(distance))
                row_matrix.append(distance)

        distance_matrix.append(row_matrix)
    # for row in distance_matrix: # вывод матрицы
    #     print(row)

def claster_maker(list_city,distance_matrix,sum_x,sum_y,value_city,k_max): # самописный алгоритм создания кластеров

    mid_x = round(sum_x/value_city)
    mid_y = round(sum_y/value_city)
    distance_dict = {}
    claster_dict = {}


    for name ,city_x,city_y in list_city: # вычисляем дистанции городов до средней точки
        x = city_x - mid_x
        y = city_y - mid_y
        distance = round((x ** 2) + (y ** 2))
        distance_dict[name] = distance

    for i in range(1,k_max+1): #  находим самые удаленные точки
        key = max(distance_dict,key=distance_dict.get)
        claster_dict[key] = []
        del distance_dict[key]


    best_dict={}
    # каждая строка это определенный город ,в строке растояния до других городов
    # из строки мы выбираем растояния до городов центроидов и ищем минимльное из них ,тоесть к кому ближе город
    for row in distance_matrix:
        for i in claster_dict.keys():
            best_dict[i]=row[i]

        key = min(best_dict,key=best_dict.get)
        name_city = distance_matrix.index(row)
        # if name_city not in claster_dict.keys(): # если нужно чтобы центроид не был в кластере
        claster_dict[key].append(name_city)
    print(f"ФИНИШ Кластеры:{claster_dict}")

def sci_method(k_max): # создание кластеров через SCIkit-learn

    # Задаём количество кластеров
    K = k_max

    # Загружаем данные из CSV-файла
    df = pd.read_csv('citySCI2.csv')

    #Извлекаем координаты для кластеризации
    X = df[['x', 'y']].values

    #Создаём модель KMeans и обучаем её
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)

    # Получаем метки кластеров для каждого города
    df['cluster'] = kmeans.labels_

    # Выводим результат
    print(df)
def wcss_maker():# метод локтя для k-means
    old_sum_distance = 1
    procent_distence =0
    value_k = 0
    while procent_distence < 5:
        value_k += 1
        sum_distance = k_means_madeself(k_max=value_k,list_city=list_city,distance_matrix=distance_matrix,wcss_maker_flag=True)
        procent_distence = ((old_sum_distance - sum_distance)/old_sum_distance)*100
        old_sum_distance = sum_distance

    return value_k

def k_means_madeself(k_max,list_city,distance_matrix,wcss_maker_flag):
    claster_dict = {} # словарь для кластеров
    old_claster_dict = {} # сохряняем старый набор кластеров
    sum_distance = 0 #cумма квадратов растояния для метода локтя
    for i in range(1,k_max+1): # выбираем случайные точки
        key = r.randint(0,len(list_city)-1)
        claster_dict[key] = []
    if wcss_maker_flag == False:
        print(f'первый {claster_dict}\n')
    while old_claster_dict.keys() != claster_dict.keys():
        best_dict = {}

        #опредление городов в кластеры
        for row in distance_matrix:
            for i in claster_dict.keys():
                best_dict[i] = row[i]

            key = min(best_dict, key=best_dict.get)
            name_city = distance_matrix.index(row)
            # if name_city not in claster_dict.keys(): # если нужно чтобы центроиды не были в кластере
            claster_dict[key].append(name_city)



        sum_x=0
        sum_y=0
        sum_distance = 0
        list_next_centroid = []
        # для каждого кластера находим среднюю точку и ближайший к ней город
        for key in claster_dict:
            for city in claster_dict[key]:
                sum_x += list_city[city][1]
                sum_y += list_city[city][2]
            mid_x = round(sum_x / value_city)
            mid_y = round(sum_y / value_city)
            min_distance = None
            next_centroid_claster = 0
            for city in claster_dict[key]: #поиск наиболее приближенного города к средней координате кластера
                x = list_city[city][1] - mid_x
                y = list_city[city][2] - mid_y
                distance = round((x ** 2) + (y ** 2))
                if wcss_maker_flag:
                    sum_distance += distance
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    next_centroid_claster=city
            sum_x=0
            sum_y=0
            list_next_centroid.append(next_centroid_claster)
        old_claster_dict = claster_dict.copy() # сохраняем старый набор кластеров
        claster_dict = {}
        for key in list_next_centroid: # создаем новый набор кластеров
            claster_dict[key] = []
        if wcss_maker_flag == False:
            print(f'новый {claster_dict}')
            print(f'старый {old_claster_dict}\n')
        else:
            return sum_distance


    print("СОВПАДЕНИЕ! конец")

start_time = time.time()
claster_maker(list_city=list_city,distance_matrix=distance_matrix,sum_x=sum_x,sum_y=sum_y,value_city=value_city,k_max=k_max)
end_time = time.time()
print(f"Время выполнения собственной функции {end_time-start_time:.4f}")
start_time = time.time()
sci_method(k_max=k_max)
end_time = time.time()
print(f"Время выполнения через SCIKIT {end_time-start_time:.4f}")
print("--------------")
elbow_k_max = wcss_maker()
start_time = time.time()
k_means_madeself(k_max=elbow_k_max,list_city=list_city,distance_matrix=distance_matrix,wcss_maker_flag=False)
end_time = time.time()
print(f"Время выполнения через K-Means {end_time-start_time:.4f}")
