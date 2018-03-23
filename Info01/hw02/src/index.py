# coding: utf-8

import re
import array
import sys

import mmh3
import mmap

import numpy as np

import datetime

#Почему-то array.array('L') при extend не принимает числа > 2^31. Поэтому ограничим этим числом хэши
def Hash(val,  signed=True):
    return mmh3.hash(val,  signed) % 2147483648 #2^31 = 2147483648
    
SPLIT_RGX = re.compile(r'\w+', re.U)
def extract_words(text):
    words = re.findall(SPLIT_RGX, text)
    #Добавим unt8 --> bytes
    return map(lambda s: s.lower().encode('utf-8'), words)

#Записывает int в массив из 4 байт (от младших разрядов к старшим)
#Возвращает <4 байта, представляющие int>
def IntToFourBytes(int_value):
    value_bytes = array.array('B')
    for i in xrange(4):
        value_bytes.append(int_value % 256)
        int_value = int_value // 256
    return value_bytes
    
def FourBytesToInt(int_bytes):
    if type(int_bytes) == str:
        bytes = array.array('B')
        bytes.fromstring(int_bytes)
    else:
        bytes = int_bytes
    
    result = 0
    current_pow = 1
    for byte in bytes:
        result += current_pow*byte
        current_pow *= 256
    return int(result)

#Классы для сжатия, реальзующие методы PackList, UnpackList
#Могут использоваться как compression_object в NewIndex
#PackArray - упаковывает array (unsigned int, 4 байта - 'L') в массив байт ('B')
#UnpackArray - распаковывает бинарную строку в np.array (uint32)

class Varbyte:
    def PackArray(self,  values_array):
        source_data = np.array(values_array,  dtype=np.uint32)
        #Размеры (в байтах) записи исходных чисел в Varbyte, позиции, в которых оканчиваются числа
        final_sizes = np.floor(np.log(source_data)/np.log(128)) + 1
        final_sizes[np.isinf(final_sizes)] = 1 #Поправка на нули
        final_sizes = final_sizes.astype(np.int8)
        current_positions = np.cumsum(final_sizes) - 1
        #Numpy конечных байтов
        result_numpy = np.zeros((int(final_sizes.sum()),  ),  dtype=np.uint8)
        #Помечаем окончания чисел
        result_numpy[current_positions] += 128
        
        #Кодирование чисел (final_sizes - отражает сколько байт текущего числа уже закодировано (> 0, если еще нужно писать),
        #values_positions - позиции текущей записи)
        for i in xrange(final_sizes.max()):
            mask = final_sizes > 0
            result_numpy[current_positions[mask]] += source_data[mask] % 128
            
            source_data = source_data // 128
            final_sizes -= 1
            current_positions -= 1
        
        result_array = array.array('B') # 'B' unsigned integer 1 byte
        result_array.fromstring(result_numpy.tobytes())
        return result_array
    
    def UnpackArray(selfs,  binary_data):
        source_array = array.array('B')
        source_array.fromstring(binary_data)
        source_numpy = np.array(source_array,  dtype=np.uint32)
        
        #Позиции окончаний чисел
        current_positions = np.where(source_numpy >= 128)[0] #т.к. tuple
        result_numpy = np.zeros((len(current_positions),  ),  dtype=np.uint32)
        #Размеры чисел
        sizes = np.insert(np.diff(current_positions),  0,  [current_positions[0]+1])
        
        #Декодирование
        current_pow = 1
        source_numpy[current_positions] -= 128 #Удаление меток окончаний чисел
        for i in xrange(sizes.max()):
            mask = sizes > 0
            result_numpy[np.where(mask)[0]] += source_numpy[current_positions[mask]] * current_pow
            
            sizes -= 1
            current_positions -= 1
            current_pow *= 128
        
        return result_numpy

#<Битовая маска> (= количество чисел % 16): <упаковка>
#8 = 0001: 1 28-и битное число
#7 = 0010: 2 14-и битных числа
#6 = 0011: 3 9-и битных числа
#5 = 0100: 4 7-и битных числа
#4 = 0101: 5 5-и битных чисел
#3 = 0110: 7 4-х битных чисел
#2 = 1001: 9 3-х битных чисел
#1 = 1110: 14 2-х битных чисел
#0 = 1100: 28 1 битных чисел
class Simple9:
    #Размечает исходный массив на квартеты Simple9
    #Возвращает массив с номерами в упаковке вида [0, 1, 2, 0, 1, 2, 3, ...], где 0 - начало нового квартета simple9,
    #число != 0 - номер данного числа в байте simple9
    def MarkArray(self,  source_data):
        # Вычисляем для каждой ячейки, сколько таких чисел как оно, может быть записано в 1 байт
        sizes_array = np.floor(np.log(source_data)/np.log(2)) + 1
        sizes_array[np.isinf(sizes_array)] = 1 #Поправка на нули
        sizes_array = np.floor(28.0 / sizes_array)
        sizes_array = sizes_array.astype(np.uint8)
        if np.sum(sizes_array == 0) > 0:
            raise ValueError("Array has too big values for simple9!")
        
        # Заполняем массив, описывающий кодирование (для каждго числа - его позиция в байте, начиная с 0)
        positions = np.zeros(source_data.shape,  dtype=np.uint8)
        counter = 0
        max_size = 0
        for i in xrange(len(source_data)):
            if counter < max_size: #Продолжается число
                if sizes_array[i] >= max_size: #Следующее число маленькое => берем его спокойно
                    positions[i] = counter
                    counter += 1
                    continue
                elif sizes_array[i] > counter: #Число уменьшает нам "запас", но его еще можно добавить
                    max_size = sizes_array[i]
                    positions[i] = counter
                    counter += 1
                    continue
            #Начинаем новое число
            max_size = sizes_array[i]
            positions[i] = 0
            counter = 1
        
        return positions
    
    def PackArray(self,  values_array):
        source_data = np.array(values_array,  dtype=np.uint32)
        
        positions = self.MarkArray(source_data)
        #Количество чисел в каждом квартете
        lengths = np.diff(np.append(np.where(positions == 0)[0],  [len(positions)])).astype(np.uint8)
        bit_sizes = np.floor(28.0 / lengths).astype(np.uint8) #Количество бит на число в каждом квартете
        result_numpy = np.zeros(lengths.shape,  dtype=np.uint32)
        
        #Записываем маски
        current_powers = np.full(lengths.shape,  268435456,  dtype=np.uint32) #268435456 = 2^28
        result_numpy += current_powers * (lengths % 16) #т.к. 4 бита, 28 --> 1100
        
        #Кодируем числа
        powers_step = np.power(2,  bit_sizes,  dtype=np.uint32) #Смещения для записи чисел в квартеты
        for i in xrange(np.max(lengths)):
            current_powers /= powers_step
            result_numpy[current_powers > 0] += source_data[positions == i] * current_powers[current_powers > 0]
        
        result_array = array.array('B')
        result_array.fromstring(result_numpy.tobytes())
        return result_array
    
    def UnpackArray(selfs,  binary_data):
        source_array = array.array('L')
        source_array.fromstring(binary_data)
        source_numpy = np.array(source_array,  dtype=np.uint32)
        
        #Восстанавливаем количества чисел, извлекаемых из квартетов
        lengths = np.floor(source_numpy / 268435456).astype(np.uint8) #268435456 = 2^28
        lengths[lengths == 12] = 28 #Исключение - битовая маска 1100 обозначает 28
        bit_sizes = np.floor(28.0 / lengths).astype(np.uint8) #Количество бит на число в каждом квартете
        result_numpy = np.zeros((np.sum(lengths),  ),  dtype=np.uint32)
        
        #Распаковка чисел
        current_powers = np.full(lengths.shape,  268435456,  dtype=np.uint32)
        powers_step = np.power(2,  bit_sizes,  dtype=np.uint32)
        current_positions = np.insert(np.cumsum(lengths)[:-1],  0,  [0])
        source_numpy = source_numpy % current_powers #Удаляем маски
        for i in xrange(np.max(lengths)):
            current_powers /= powers_step
            current_mask = current_powers > 0
            result_numpy[current_positions[current_mask]] = np.floor(source_numpy[current_mask] / current_powers[current_mask])
            current_positions += 1
            source_numpy[current_mask] = source_numpy[current_mask] % current_powers[current_mask] #Удаляем предыдущие числа
        
        return result_numpy
        
#Упаковывает и распаковывает массив uint, 4 bytes без сжатия
class NoCompression:
    def PackArray(self,  values_array):
        source_array = array.array('L')
        source_array.extend(values_array)
        
        result_array = array.array('B') 
        result_array.fromstring(values_array.tostring())
        return result_array
    
    def UnpackArray(self,  binary_data):
        result_array = array.array('L')
        result_array.fromstring(binary_data)
        return result_array
        
#Список возможных упаковщиков для поддержки поля <тип упаковки> в файлах индекса
#Тип упаковки = позиция упаковщика в списке
COMPRESSION_OBJECTS = [Varbyte(),  Simple9()]

#Словарь для сохранения индекса в файл (при индексации документов)
class NewDictionary:
    #Т.к. создается только при сохранении индекса, то мы уже знаем количество термов в словаре
    def __init__(self,  count_terms,  compression_object=Varbyte()):
        self.count_baskets = count_terms // 4096 + 1 # 4096 ключей в корзине в среднем, 1 корзина минимально
        self.terms_hashes = [array.array('L')] * self.count_baskets
        self.terms_shifts = [array.array('L')] * self.count_baskets
        
        self.compression_object = compression_object
    
    #Формат хранения
    #<Хэши термов/Положения термов> <Положения корзин> <N>
    #(корзины перемещены в конец для возможности записи корзинок по очереди)
    #<N> - количество корзин с ключами, в каждой, в среднем, до 4096 ключей (uint, 4 байта)
    #<Положения корзин> - N чисел uint (4 байта) - положение содержимого каждой корзины в файле (в обратном порядке)
    #<Хэши термов/Положения термов> - содержимое корзины (возможно, сжатое),
    #представляет из себя две равные половины: первая половина - int (unsigned, 4 bytes) - хэши термов,
    #вторая половина - позиции в индексе (записанные в том же порядке, unsigned, 4 bytes). Хэши записаны в порядке возрастания
    def SaveToFile(self,  dict_filename):
        with open(dict_filename,  'w+b') as f:
            basket_info = array.array('B')
            basket_info.extend([0] * 4*self.count_baskets) #Место для записи положений корзин
            basket_info.extend(IntToFourBytes(self.count_baskets)) #Количество корзин
            
            #Начинаем писать с начала файла
            current_position = 0
            tmp_array = array.array('B')
            for i in xrange(self.count_baskets):
                del tmp_array[:]
                
                #Сохраняем положение текущей корзины
                basket_info[-8 - i*4 : -4 - i*4] = IntToFourBytes(current_position)
                
                #Сортируем содержимое корзины
                order = np.argsort(self.terms_hashes[i])
                hashes_sorted = np.array(self.terms_hashes[i],  dtype=np.uint32)[order]
                shifts_sorted = np.array(self.terms_shifts[i],  dtype=np.uint32)[order]
                
                #Записываем результат
                basket = np.hstack((hashes_sorted,  shifts_sorted))
                basket_compressed = self.compression_object.PackArray(basket)
                tmp_array.extend(basket_compressed)
                f.write(tmp_array.tostring())
                
                current_position += len(basket_compressed)
                
            f.write(basket_info.tostring())
        
    def AddTerm(self,  term,  shift):
        hash = Hash(term,  signed=False)
        basket_id = hash % self.count_baskets
        self.terms_hashes[basket_id].append(hash)
        self.terms_shifts[basket_id].append(shift)
        
    def GetSize(self):
        return sys.getsizeof(self.terms_hashes)+sys.getsizeof(self.terms_shifts)
        
#Словарь, загруженный из файла (для навигации по существующему индексу)
class LoadedDictionary:
    def __init__(self,  dict_filename,  compression_object=Varbyte()):
        self.compression_object = compression_object
        
        self.dict_file = open(dict_filename,  'r+b')
        self.mapped_dict = mmap.mmap(self.dict_file.fileno(),  0)
        
        #Корзины записаны в конце файла!
        self.count_baskets = FourBytesToInt(self.mapped_dict[-4:None])
        self.baskets_positions = np.zeros((self.count_baskets, ),  dtype=int)
        for i in xrange(self.count_baskets):
            self.baskets_positions[i] = FourBytesToInt(self.mapped_dict[-8 - i*4 : -4 - i*4])
    
    #Возвращает shift для term или -1, если он не найден в словаре
    def GetTermPosition(self, term):
        hash = Hash(term,  signed=False)
        
        #Ищем границы корзинки для распаковки
        basket_id = hash % self.count_baskets
        left = self.baskets_positions[basket_id]
        if basket_id != self.count_baskets-1:
            right = self.baskets_positions[basket_id+1]
        else:
            right = None #Срез до последнего элемента
        
        #Распаковываем корзинку и ищем элемент
        basket = self.compression_object.UnpackArray(self.mapped_dict[left:right])
        basket_items_count = len(basket) / 2 #Количество хэшей = количеству сдвигов, все в одной корзинке
        hash_position = np.searchsorted(basket[:basket_items_count],  hash)
        if basket[hash_position] != hash: #В словаре терм не найден
            return -1
        else:
            return basket[hash_position + basket_items_count] #Сдвиг находится во второй половине корзинки
        
    def Close(self):
        self.mapped_dict.close()
        self.dict_file.close()

#Класс для индексации набора документов
class NewIndex:
    #compression_object - объект класса, реальзующего методы PackList, UnpackList
    def __init__(self,  compression_object):
        self.compression_object = compression_object
        self.index = dict()
        
        self.count_ids = 0 #Для учета размеров списков в индексе
        
    #Формат хранения:
    #<Тип упаковки (1 байт, индекс упаковщика в массиве COMPRESSION_OBJECTS)
    #Далее - <Длина сжатого списка в байтах (4 байта)><Список docid сжатый, в бинарной форме>
    def SaveToFile(self,  index_filename,  dict_filename):
        dictionary = NewDictionary(len(self.index))
        
        with open(index_filename,  'w+b') as index_file:
            tmp_array = array.array('B')
            tmp_array.append(COMPRESSION_OBJECTS.index(self.compression_object))
            index_file.write(tmp_array.tostring())
            
            current_position = 1 #Т.к. записали тип упаковки
            for term,  doclist in self.index.iteritems():
                del tmp_array[:]
                dictionary.AddTerm(term,  current_position)
                
                packed_list = self.compression_object.PackArray(doclist)
                list_len = len(packed_list) #Количество байт = длине, т.к. packed_list - строка
                list_len_bytes = IntToFourBytes(list_len)  #само представление длины - 4 байта
                tmp_array.extend(list_len_bytes) #<Длина сжатого списка, представленная в 4 байтах>
                tmp_array.extend(packed_list) #Сам сжатый список
                index_file.write(tmp_array.tostring())
                
                current_position += (list_len+4)
            
        #print "Saving dictionary ...",  datetime.datetime.now()
        dictionary.SaveToFile(dict_filename)
        
    def IndexDocument(self,  docid,  text):
        words = extract_words(text)
        for w in words:
            try:
                if self.index[w][-1] != docid:
                    self.index[w].append(docid)
                    self.count_ids += 1 #4 байта - новый docid
            except KeyError:
                self.index.update({w: array.array('L')}) #uint, 4 bytes
                self.index[w].append(docid)
                self.count_ids += 1 #4 байта - новый docid
                
    # С поправкой на сохраняемый словарь
    def GetSize(self):
        #На каждый терм в словаре: хэш+индекс по 4 байта
        return self.count_ids*4 + sys.getsizeof(self.index) + len(self.index)*2*4
    
#Класс для индекса, загруженного из файла
#Используется для булева поиска и оптимизации индекса
class LoadedIndex:
    def __init__(self,  index_filename,  dict_filename):
        self.index_file = open(index_filename,  'r+b')
        self.index_mapped = mmap.mmap(self.index_file.fileno(),  0)
        self.compression_object = COMPRESSION_OBJECTS[ord(self.index_mapped[0])] #Восстановление типа упаковки
        
        self.dictionary = LoadedDictionary(dict_filename)
    
    #Возвращает индекс (в сжатом или развернутом виде) для переданного term
    #None, если терм не представлен в индексе
    def GetIndexForTerm(self,  term,  compressed=False):
        position = self.dictionary.GetTermPosition(term)
        if position == -1:
            return None
        
        #4 байта - длина сжатого списка
        length = FourBytesToInt(self.index_mapped[position : position+4])
        if compressed == True:
            return self.index_mapped[position+4 : position+4+length]
        else:
            return self.compression_object.UnpackArray(self.index_mapped[position+4 : position+4+length])
        
    def Close(self):
        self.dictionary.Close()
        
        self.index_mapped.close()
        self.index_file.close()

#Класс индекса, собирающего другие во время оптимизации
#Занимает промежуточное положение между NewIndex и LoadedIndex: хранит dict() {term: <Сжатые индексы>}
class OptimizatingIndex:
    def __init__(self,  compression_object):
        self.compression_object = compression_object
        self.index = dict()
        
        #Группа обслуживания дополняемого файла индекса
        self.dictionary = dict()
        self.current_position = 0
    
    #Используется для сохранения текущего состояния индекса в файл при его оптимизации
    #Eсли файл индекса существует, то данные помещаются в конец файла
    #Не создает словарь, его нужно создать отдельно!
    def FlushIndexToFile(self,  index_filename):
        with open(index_filename,  'a+b') as index_file:
            tmp_array = array.array('B')
            
            #Если это запись начала файла
            if self.current_position == 0:
                tmp_array.append(COMPRESSION_OBJECTS.index(self.compression_object))
                self.current_position = 1 #Т.к. записали тип упаковки
                index_file.write(tmp_array.tostring())
                
            for term,  compressed_list in self.index.iteritems():
                del tmp_array[:]
                self.dictionary[term] = self.current_position #Терм уже регистрировали в словаре смещений при добавлении
                
                compressed_list_len = len(compressed_list) #Количество байт = длине, т.к. compressed_list - строка
                compressed_list_len_bytes = IntToFourBytes(compressed_list_len)  #само представление длины - 4 байта
                tmp_array.extend(compressed_list_len_bytes) #<Длина сжатого списка, представленная в 4 байтах>
                tmp_array.extend(compressed_list) #Сам сжатый список
                index_file.write(tmp_array.tostring())
                
                self.current_position += (compressed_list_len+4)
            
            self.index.clear()
            
    def SaveDict(self,  dict_filename):
        new_dictionary = NewDictionary(len(self.dictionary))
        for term,  shift in self.dictionary.iteritems():
            new_dictionary.AddTerm(term,  shift)
        new_dictionary.SaveToFile(dict_filename)
        
    def ExtendCompressedIndex(self,  term,  new_index):
        """
        Дополняет имеющийся индекс для терма term индексом, переданным в параметре new_index
        Предполагает, что new_index - сжатое представление индекса, совместимое
        с self.compression_type (полученное. например, из LoadedIndex.GetIndexForTerm c compressed=True)
        Если term нет в текущем индексе, то он добавляется
        Также предполагает, что new_index будут добавляться по мере возрастания docid
        """
        try:
            self.index[term].extend(new_index)
        except KeyError:
            self.index.update({term: array.array('B')})
            self.index[term].extend(new_index)
            
        #Проверяем, нет ли повтора терма после flush, регистрируем его, если это новый терм
        try:
            if self.dictionary[term] != 0:
                raise RuntimeError("Cannot extend index for term after it was flushed!")
        except KeyError:
            self.dictionary.update({term: 0})
        
    def GetSize(self):
        return sys.getsizeof(self.index)+sys.getsizeof(self.dictionary)

#arr1 = np.array([5,  3,  5,  6,  4,])
#arr1[np.array([0,  1])][False,  True] += arr1[np.array([0,  1])][0] % 2
#print arr1

#arr = np.array([0,  1,  3,  4])
#arr = array.array('L')
#arr.extend([29, 83, 239, 292, 301, 547])#, 579, 696, 715, 759, 1027, 1116, 1145, 1173,  1249, 1274, 1296, 1333, 1374, 1458])
#packed = Simple9().PackArray(arr)
#print Simple9().UnpackArray(packed.tostring())
#order = np.array([3,  2,  0,  1])
#print arr[:None]
#
#f = open('test.txt',  'w')
#f.write('ABCDE')
#f.close()
#f = open('test.txt',  'a')
#f.write('012345')
#f.close()
#f = open('test.txt',  'a')
#f.seek(0,  0)
#f.write('-*/+')
#f.close()

#arr = range(0,  100)
#print arr[-8:-4]
#print arr[-4:None]
