# Модуль для превращения песен в коэффициенты

import numpy as np
import pandas as pd
import itertools

import python_speech_features as psf
import librosa as lb

# Для PlotConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns

def _AddStatistics(audio_files, numcep, winlen, winstep):
    statistics_index = ['name_statistic', 'func_statistic']
    statistics = [pd.Series(data=['mean', lambda x: x.mean(axis=0)], index=statistics_index),
                  pd.Series(data=['std', lambda x: x.std(axis=0)], index=statistics_index),
                  pd.Series(data=['max', lambda x: x.max(axis=0)], index=statistics_index),
                  pd.Series(data=['min', lambda x: x.min(axis=0)], index=statistics_index),
                  pd.Series(data=['diff_mean', lambda x: np.diff(x, axis=1).mean(axis=0)], index=statistics_index),
                  pd.Series(data=['diff_std',  lambda x: np.diff(x, axis=1).std(axis=0)], index=statistics_index)]
    
    for statistic in statistics:
        audio_files.loc[:, statistic.name_statistic] = object() #Создание колонки под статистику в датасете

    for i in xrange(audio_files.shape[0]):
        audio_file = audio_files.loc[i,:]
        mfcc = psf.mfcc(audio_file.audio_data, samplerate=audio_file.samplerate, numcep=numcep,
                        winlen=winlen, winstep=winstep, nfft=1024)
        for statistic in statistics:
            audio_file[statistic.name_statistic] = statistic.func_statistic(mfcc)
        audio_files.loc[i,:] = audio_file
    
    return audio_files
    
def GetFilenameList(system_list_output, OSTYPE, current_dir=None):
    """
    Возвращает имена файлов для обработки на основании вывода find/dir
    system_list_output - результат команды "!find <директория_с_файлами> -type f | sort" для LINUX или "!dir <директория_с_файлами> /B /S" для Windows
    current_dir - результат выполнения "!echo %cd%" (нулевой элемент массива), нужен для Windows
    
    В пути к файлу не должно быть точек в названиях директорий!z
    """

    file_names = list()
    for file_name in system_list_output:
        if not (file_name.find('.') == -1): #Это должен быть файл, а не директория
            if OSTYPE == 'Windows':
                file_name = file_name.replace(current_dir+'\\', '')
            file_names.append(file_name)
    return file_names

def LoadAudio(file_names, has_genres, OSTYPE, crop=False, numcep=30, winlen=0.025, winstep=0.01):
    """
        file_names - строка или список
        Если has_genres == True:
            Загружает аудио из папок вида "жанр"\(/)"название песни" (пример: jazz\001.jazz.wav - для Windows, / для Linux).
        Иначе:
            Загружает файлы по именам
        Ожидает увидеть то, что может загрузить librosa.
        crop - обрезка большого аудио (эвристическая)
        Параметры соответствуют свойствам преобразования mfcc
    """
    columns_audio_file = ['genre', 'file_name', 'samplerate', 'duration', 'audio_data'] #Информация о файле и его содержимом в pd.DataFrame
    audio_files = pd.DataFrame(columns=columns_audio_file)
    
    if type(file_names) == str:
        file_names = [file_names]
    for k, file_name in enumerate(file_names):
        audio_file = pd.Series(index=columns_audio_file)
        splited = file_name.split('/' if (OSTYPE == 'Linux') else '\\')
        if has_genres == True:
            audio_file.genre = splited[-2]
        audio_file.file_name = splited[-1] 
        
        audio_data_librosa, audio_file.samplerate = lb.load(file_name)
        audio_file.duration = len(audio_data_librosa)/audio_file.samplerate
        audio_file.audio_data = np.array(audio_data_librosa)
        if crop == False or audio_file.duration <= 60.0:
            audio_file.audio_data = np.array(audio_data_librosa)
        else:
            #Отрезаем по 10 сек с начала и конца, оставшиеся делим на 15 сек фрагменты (возм. последний меньше)
            #Берем по 2 случайных из каждых четырех+1 из остатка
            audio_data_list = list()
            left_borders = np.arange(10*audio_file.samplerate, len(audio_data_librosa)-10*audio_file.samplerate, 15*audio_file.samplerate, dtype=int)
            count_left = len(left_borders)
            left_borders = np.array(list(left_borders)+[len(audio_data_librosa)-10*audio_file.samplerate])
            for i in xrange(0, count_left, 4):
                if i + 4 > count_left:
                    break
                random_fragments = np.random.choice(4, 2)
                audio_data_list += list(audio_data_librosa[left_borders[i+random_fragments[0]]:left_borders[i+random_fragments[0]+1]])
                #audio_data_list += list(audio_data_librosa[left_borders[i+random_fragments[1]]:left_borders[i+random_fragments[1]+1]])
            if i + 4 != count_left: #Разделилось на фрагменты по 4 без остатка
                random_fragment = np.random.choice(len(left_borders) - i)
                audio_data_list += list(audio_data_librosa[left_borders[random_fragment]:left_borders[random_fragment+1]])
            audio_file.audio_data = np.array(audio_data_list)
            
        audio_files = audio_files.append(audio_file, ignore_index=True)
    
    # Удаление колонки audio_data после расчета статистик
    return _AddStatistics(audio_files, numcep, winlen, winstep).drop(labels=["audio_data"], axis=1)
    
def PrepareDataForModel(dataframe, labels_dict, normalize=True, shuffle=True):
    """
    Превращает датасет, полученный функцией LoadAudio в X и y для fit и predict.
    Если жанров в датасете нет, то будет возвращен y = [None]*n_samples
    Преобразование названий жанров в датасете - согласно labels_dict (словарь - имя-id)
    """

    X = dataframe.iloc[:,4:].values
    #.any() = True, если есть хоть один True => должен быть False
    if not dataframe.iloc[:,0].isnull().values.any():
        y = np.array([labels_dict[value] for value in dataframe.iloc[:,0].values]) #Преобразование согласно меткам
    else:
        y = dataframe.iloc[:,0].values
    
    # Объединение списков фич в единый список
    res = list()
    for x in X:
        res_prep = list()
        for element in x:
            res_prep += list(element)
        res.append(res_prep)
    X = np.array(res, dtype=np.float32)
    del res
    
    #Добавим нормировку и центрирование
    if normalize == True:
        X = (X-X.mean(axis=0))/X.std(axis=0)
    
    if shuffle == True:
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]
    return X, y

def PlotConfusionMatrix(cm, classes, threshold_ratio=0.5, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Функция печатает и рисует (т.е. остается вызвать только plt.show()) ConfusionMatrix.
    cm - ConfusionMatrix из sklearn (n classes*n classes)
    classes - массив меток классов, соответствующих cm
    threshold_ratio - доля от макс. значения в cm, после которого цифра станет белой
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.rcParams["axes.grid"] = False #Удалить белые линии внутри ячеек
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() * threshold_ratio
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def StringToList(string):
    string = string.replace("\n", "")
    string = string[1:-1] # Убираем квадратные скобки

    result = list()
    for s in string.split(" "):
        if s == "":
            continue
        else:
            result.append(float(s))

    return result