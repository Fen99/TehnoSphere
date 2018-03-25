# coding: utf-8

import re
import array
import sys
import struct

import mmh3
import mmap

import numpy as np
from collections import OrderedDict

import datetime

#Почему-то array.array('L') при extend не принимает числа > 2^31. Поэтому ограничим этим числом хэши
def Hash(val,  signed=True):
    return mmh3.hash(val,  signed) % 2147483648 #2^31 = 2147483648
    
SPLIT_RGX = re.compile(r'\w+', re.U)
def extract_words(text):
    words = re.findall(SPLIT_RGX, text)
    #Добавим unt8 --> bytes
    return map(lambda s: s.lower().encode('utf-8'), words)

#Классы для сжатия, реальзующие методы PackList, UnpackList
#Могут использоваться как compression_class в NewIndex
#PackArray - упаковывает array (unsigned int, 4 байта - 'L') в массив байт ('B') - в виде строки (!)
#UnpackArray - распаковывает бинарную строку в np.array (uint32)
class Varbyte:
    @staticmethod
    def PackArray(values_array):
        source_data = np.array(values_array,  dtype=np.uint32)
        #Размеры (в байтах) записи исходных чисел в Varbyte, позиции, в которых оканчиваются числа
        final_sizes = np.ones(source_data.shape,  dtype=np.uint8)
        source_copy = np.copy(source_data)
        while np.any(source_copy > 0):
            source_copy = source_copy // 128
            final_sizes += 1
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
        
        return result_numpy
    
    @staticmethod
    def UnpackArray(binary_data):
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
#!!! Ошибочная упаковка в Simple 9 !!!
#Данная реализация упаковывает неправильно случаи, когда обрывается последовательность из >15 однобитовых
#и 12 двухбитовых чисел (т.к. для них невозможно записать количество чисел в маске, а 12 зарезервировано под 28 чисел)
#Но эти случаи очень маловероятны для нашей задачи

#Количества бит на число и максимальные количества чисел
COUNT_BITS = np.array([1,  2,  3,  4,  5,  7,  9,  14,  28],  dtype=np.uint8)
COUNT_VALUES = COUNT_BITS[::-1]
#2^(количество_бит_на_число) - первое число, непредставимое данным количеством бит, степени для возведения
MAX_VALUES = np.array([2,  4,  8,  16,  32,  128,  512,  16384,  268435456],  dtype=np.uint32)
class Simple9:
    #Размечает исходный массив на квартеты Simple9
    #Возвращает массив с номерами в упаковке вида [0, 1, 2, 0, 1, 2, 3, ...], где 0 - начало нового квартета simple9,
    #число - номер данного числа в байте simple9
    @staticmethod
    def __MarkArray(source_data):
        # Вычисляем для каждой ячейки, сколько таких чисел как оно, может быть записано в 1 байт
        sizes_array = COUNT_VALUES[np.searchsorted(MAX_VALUES - 1,  source_data)]
        
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
    
    @staticmethod
    def PackArray(values_array):
        source_data = np.array(values_array,  dtype=np.uint32)
        
        positions = Simple9.__MarkArray(source_data)
        #Количество чисел в каждом квартете
        lengths = np.diff(np.append(np.where(positions == 0)[0],  [len(positions)])).astype(np.uint8)
        bit_sizes = np.floor(28.0 / lengths).astype(np.uint8) #Количество бит на число в каждом квартете
        result_numpy = np.zeros(lengths.shape,  dtype=np.uint32)
        
        #Записываем маски
        current_powers = np.full(lengths.shape,  268435456,  dtype=np.uint32) #268435456 = 2^28
        result_numpy += current_powers * (lengths % 16) #т.к. 4 бита, 28 --> 1100
        
        #Кодируем числа
        #powers_step = np.power(2,  bit_sizes,  dtype=np.uint32) #Смещения для записи чисел в квартеты
        powers_step = MAX_VALUES[np.searchsorted(COUNT_BITS,  bit_sizes)]
        del bit_sizes
        for i in xrange(np.max(lengths)):
            current_powers /= powers_step
            mask = lengths > 0
            result_numpy[mask] += source_data[positions == i] * current_powers[mask]
            lengths[mask] -= 1
        
        return result_numpy
    
    @staticmethod
    def UnpackArray(binary_data):
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
        powers_step = MAX_VALUES[np.searchsorted(COUNT_BITS,  bit_sizes)]
        current_positions = np.insert(np.cumsum(lengths)[:-1],  0,  [0])
        source_numpy = source_numpy % current_powers #Удаляем маски
        for i in xrange(np.max(lengths)):
            current_powers /= powers_step
            current_mask = lengths > 0
            result_numpy[current_positions[current_mask]] = np.floor(source_numpy[current_mask] / current_powers[current_mask])
            current_positions += 1
            source_numpy[current_mask] = source_numpy[current_mask] % current_powers[current_mask] #Удаляем предыдущие числа
            lengths[current_mask] -= 1
        
        return result_numpy
        
#Упаковывает и распаковывает массив uint, 4 bytes без сжатия
class NoCompression:
    @staticmethod
    def PackArray(values_array):
        return np.array(values_array,  dtype=np.uint32)
    
    @staticmethod
    def UnpackArray(binary_data):
        return np.frombuffer(binary_data,  dtype=np.uint32)
        
#Список возможных упаковщиков для поддержки поля <тип упаковки> в файлах индекса
#Тип упаковки = позиция упаковщика в списке
COMPRESSION_CLASSES = [Simple9,  Simple9]
DICTIONARY_COMPRESSION = NoCompression

#Словарь для сохранения индекса в файл (при индексации документов)
class NewDictionary:
    #Т.к. создается только при сохранении индекса, то мы уже знаем количество термов в словаре
    def __CreateArray(self):
        return [array.array('L') for i in xrange(self.count_baskets)]
    
    def __init__(self,  count_terms,  compression_class=DICTIONARY_COMPRESSION):
        self.count_baskets = count_terms // 1024 + 1 # 1024 ключей в корзине в среднем, 1 корзина минимально
        self.terms_hashes = self.__CreateArray()
        self.terms_shifts = self.__CreateArray()
        self.terms_ends = self.__CreateArray()
        
        self.compression_class = DICTIONARY_COMPRESSION
    
    #Формат хранения
    #<Хэши термов/Положения термов> <Положения корзин> <N>
    #(корзины перемещены в конец для возможности записи корзинок по очереди)
    #<N> - количество корзин с ключами, в каждой, в среднем, до 1024 ключей (uint, 4 байта)
    #<Положения корзин> - N чисел uint (4 байта) - положение содержимого каждой корзины в файле (в обратном порядке)
    #<Хэши термов/Положения термов> - содержимое корзины (возможно, сжатое),
    #представляет из себя две равные половины: первая половина - int (unsigned, 4 bytes) - хэши термов,
    #вторая половина - позиции в индексе (записанные в том же порядке, unsigned, 4 bytes). Хэши записаны в порядке возрастания
    def SaveToFile(self,  dict_filename):
        with open(dict_filename,  'w+b') as f:
            #basket_info = array.array('B')
            #basket_info.extend([0 for i in xrange(4*self.count_baskets)]) #Место для записи положений корзин
            #Еще 4 байта - на колчиество корзин
            basket_info = ""
            
            #Начинаем писать с начала файла
            current_position = 0
            for i in xrange(self.count_baskets):
                #Сохраняем положение текущей корзины
                basket_info = struct.pack(">I", current_position)+basket_info
                
                #Сортируем содержимое корзины
                order = np.argsort(self.terms_hashes[i])
                hashes_sorted = np.array(self.terms_hashes[i],  dtype=np.uint32)[order]
                shifts_sorted = np.array(self.terms_shifts[i],  dtype=np.uint32)[order]
                ends_sorted = np.array(self.terms_ends[i],  dtype=np.uint32)[order]
                
                #Записываем результат
                basket = np.hstack((hashes_sorted,  shifts_sorted,  ends_sorted))
                basket_compressed = self.compression_class.PackArray(basket).tobytes()
                f.write(basket_compressed)
                
                current_position += len(basket_compressed)
            basket_info = struct.pack(">I", current_position)+basket_info #Дописываем последнюю корзину
                
            f.write(basket_info)
            f.write(struct.pack(">I", self.count_baskets))
        
    def AddTerm(self,  term,  shift,  end):
        hash = Hash(term,  signed=False)
        basket_id = hash % self.count_baskets
        self.terms_hashes[basket_id].append(hash)
        self.terms_shifts[basket_id].append(shift)
        self.terms_ends[basket_id].append(end)
        
    def GetSize(self):
        return sys.getsizeof(self.terms_hashes)+sys.getsizeof(self.terms_ends)+sys.getsizeof(self.lengths)
        
#Словарь, загруженный из файла (для навигации по существующему индексу)
class LoadedDictionary:
    def __init__(self,  dict_filename,  compression_class=DICTIONARY_COMPRESSION):
        self.compression_class = DICTIONARY_COMPRESSION
        
        self.dict_file = open(dict_filename,  'r+b')
        self.mapped_dict = self.dict_file.read()
        
        #Корзины записаны в конце файла!
        self.count_baskets = struct.unpack(">I",  self.mapped_dict[-4:None])[0]
        self.baskets_positions = np.zeros((self.count_baskets+1, ),  dtype=np.uint32)
        for i in xrange(self.count_baskets+1): #Т.к. дописано еще окончание последней корзины
            self.baskets_positions[i] = struct.unpack(">I", self.mapped_dict[-8 - i*4 : -4 - i*4])[0]
    
    #Возвращает shift, end для term или -1, если он не найден в словаре
    def GetTermPosition(self, term):
        hash = Hash(term,  signed=False)
        
        #Ищем границы корзинки для распаковки
        basket_id = hash % self.count_baskets
        left = self.baskets_positions[basket_id]
        right = self.baskets_positions[basket_id+1] #Т.к. дописана "фиктивная" корзина
        
        #Распаковываем корзинку и ищем элемент
        basket = self.compression_class.UnpackArray(self.mapped_dict[left:right])
        basket_items_count = len(basket) / 3 #Количество хэшей = количеству сдвигов = к-ву длин, все в одной корзинке
        hash_position = np.searchsorted(basket[:basket_items_count],  hash)
        if basket[hash_position] != hash: #В словаре терм не найден
            return -1,  -1
        else:
            return basket[hash_position + basket_items_count],  basket[hash_position + basket_items_count*2]
        
    def Close(self):
        #self.mapped_dict.close()
        self.dict_file.close()

#Класс для индексации набора документов
#Константа для обозначения конца индекса при сжатии
MAX_INDEX = 268435455 
#Соответствующее ей число в Simple9
#Varbyte 127 127 127 255
MAX_SIMPLE9 = 536870911 # = 268435455+268435456
class NewIndex:
    #compression_class - класс, реальзующий методы PackList, UnpackList
    def __init__(self,  compression_class):
        self.compression_class = compression_class
        self.index = OrderedDict()
        
        self.count_ids = 0 #Для учета размеров списков в индексе
        
    #Формат хранения:
    #<Тип упаковки (1 байт, индекс упаковщика в массиве COMPRESSION_CLASSES)
    #Далее - <Длина сжатого списка в байтах (4 байта)><Список docid сжатый, в бинарной форме>
    def SaveToFile(self,  index_filename,  dict_filename):
        dictionary = NewDictionary(len(self.index),  self.compression_class)
        
        with open(index_filename,  'w+b') as index_file:
            tmp_array = array.array('B')
            tmp_array.append(COMPRESSION_CLASSES.index(self.compression_class))
            index_file.write(tmp_array.tostring()) #!+1
            
            #Строим индекс
            #all_index = array.array('L')
            all_arrays = []
            for term,  doclist in self.index.iteritems():
                doclist.append(268435455) #268435455 = 2^28 - 1 - обозначение окончания индекса - 1 байт Simple9
                all_arrays.append(doclist)
            all_index = np.concatenate(tuple(all_arrays))
            #print "Concated",  datetime.datetime.now()
            for arr in all_arrays:
                del arr[:]
            packed = self.compression_class.PackArray(all_index)
            del all_index
            index_file.write(packed.tobytes())
        #print "Packed+deleted 2",  datetime.datetime.now()
        #Расчитываем позиции окончаний списков
        #additional_value_len - сколько места занимает терминатор (268435455) при данном способе кодирования
        if self.compression_class is Simple9:
            end_positions = np.where(packed == MAX_SIMPLE9)[0] * 4 #Т.к. байты, а packed - массив uint32
        if self.compression_class is Varbyte:
            end_positions = np.where(packed == 255)[0]
            for i in xrange(1,  4): #Еще 3 байта по 127
                end_positions = end_positions[packed[end_positions-i] == 127]
            end_positions -= 3 #перемещаем на начало 
            
        #Составляем словарь (+1 - первый бит документа)
        shift = 1
        for i,  term in enumerate(self.index):
            dictionary.AddTerm(term, shift,  end_positions[i]+1)
            shift = end_positions[i]+5 #+4+1
            
        #print "Saving dictionary ...",  datetime.datetime.now()
        dictionary.SaveToFile(dict_filename)
        
    def IndexDocument(self,  docid,  text):
        words = extract_words(text)
        for w in words:
            doclist = self.index.get(w)
            if doclist is None:
                self.index.update({w: array.array('L')}) #uint, 4 bytes
                self.index[w].append(docid)
                self.count_ids += 1
            elif self.index[w][-1] != docid:
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
        self.compression_class = COMPRESSION_CLASSES[ord(self.index_mapped[0])] #Восстановление типа упаковки
        
        self.dictionary = LoadedDictionary(dict_filename,  self.compression_class)
    
    #Возвращает индекс (в сжатом или развернутом виде) для переданного term
    #None, если терм не представлен в индексе
    def GetIndexForTerm(self,  term,  compressed=False):
        shift,  end = self.dictionary.GetTermPosition(term)
        if shift == -1:
            return None
        
        if compressed == True:
            return self.index_mapped[shift : end]
        else:
            return self.compression_class.UnpackArray(self.index_mapped[shift : end])
        
    def Close(self):
        self.dictionary.Close()
        self.index_mapped.close()
        self.index_file.close()
        
#Класс индекса, собирающего другие во время оптимизации
#Занимает промежуточное положение между NewIndex и LoadedIndex: хранит dict() {term: <Сжатые индексы>}
#class OptimizatingIndex:
#    def __init__(self,  compression_class):
#        self.compression_class = compression_class
#        self.index = dict()
#        
#        #Группа обслуживания дополняемого файла индекса
#        self.dictionary = dict()
#        self.current_position = 0
#    
#    #Используется для сохранения текущего состояния индекса в файл при его оптимизации
#    #Eсли файл индекса существует, то данные помещаются в конец файла
#    #Не создает словарь, его нужно создать отдельно!
#    def FlushIndexToFile(self,  index_filename):
#        with open(index_filename,  'a+b') as index_file:
#            tmp_array = array.array('B')
#            
#            #Если это запись начала файла
#            if self.current_position == 0:
#                tmp_array.append(COMPRESSION_CLASSES.index(self.compression_class))
#                self.current_position = 1 #Т.к. записали тип упаковки
#                index_file.write(tmp_array.tostring())
#                
#            for term,  compressed_list in self.index.iteritems():
#                self.dictionary[term] = self.current_position #Терм уже регистрировали в словаре смещений при добавлении
#                
#                compressed_list_len = len(compressed_list) #Количество байт = длине, т.к. compressed_list - строка
#                compressed_list_len_bytes = IntToFourBytes(compressed_list_len).tostring()  #само представление длины - 4 байта
#                index_file.write(compressed_list_len_bytes) #<Длина сжатого списка, представленная в 4 байтах>
#                index_file.write(compressed_list) #Сам сжатый список
#                
#                self.current_position += (compressed_list_len+4)
#            
#            self.index.clear()
#            
#    def SaveDict(self,  dict_filename):
#        new_dictionary = NewDictionary(len(self.dictionary))
#        for term,  shift in self.dictionary.iteritems():
#            new_dictionary.AddTerm(term,  shift)
#        new_dictionary.SaveToFile(dict_filename)
#        
#    def ExtendCompressedIndex(self,  term,  new_index):
#        """
#        Дополняет имеющийся индекс для терма term индексом, переданным в параметре new_index
#        Предполагает, что new_index - сжатое представление индекса, совместимое
#        с self.compression_type (полученное. например, из LoadedIndex.GetIndexForTerm c compressed=True)
#        Если term нет в текущем индексе, то он добавляется
#        Также предполагает, что new_index будут добавляться по мере возрастания docid
#        """
#        try:
#            self.index[term].extend(new_index)
#        except KeyError:
#            self.index.update({term: array.array('B')})
#            self.index[term].extend(new_index)
#            
#        #Проверяем, нет ли повтора терма после flush, регистрируем его, если это новый терм
#        try:
#            if self.dictionary[term] != 0:
#                raise RuntimeError("Cannot extend index for term after it was flushed!")
#        except KeyError:
#            self.dictionary.update({term: 0})
#        
#    def GetSize(self):
#        return sys.getsizeof(self.index)+sys.getsizeof(self.dictionary)

#arr1 = np.array([5,  3,  5,  6,  4,])
#arr1[np.array([0,  1])][False,  True] += arr1[np.array([0,  1])][0] % 2
#print arr1
#val = Simple9
#if val is Varbyte:
#    print "A"

#arr = np.array([0,  1, 3,  4])
#print arr[2:-1]
#arr = array.array('L')
#arr.extend([0,  2,  423,  434,  22,  11,  22,  35,  4,  268435455])
#packed = NoCompression.PackArray(arr)
#print packed
#print NoCompression.UnpackArray(packed.tostring())
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
