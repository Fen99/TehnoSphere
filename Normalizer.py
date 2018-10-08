# -*- coding: utf-8 -*-

import numpy as np
import subprocess
import os
import re

import Stemmer

STEMMER = 'yandex' #pystem, yandex, nostem

EXTRACT_DIGITS = re.compile("(\d+)")
NOT_DIGIT_OR_LETTER = re.compile("\W+")

# Предварительно нормализует текст
def FirstReplaces(text, preserve_tabs=False):
    text = text.replace("-\n", "") # Склейка переносов
    text = re.sub(NOT_DIGIT_OR_LETTER, r' ', text) # Удаляем все небуквенные и нециферные символы
    text = ' '.join(re.split(EXTRACT_DIGITS, text)) # Разделяем цифры со словами
    return text

def YandexStemming(text, preserve_tabs=False):
    text_hash = str(np.abs(hash(text)))

    # Сохраняем текст в файл, вызываем стеммер
    input = open(text_hash+".in", 'w', encoding="utf-8")
    input.write(text)
    input.close()
    subprocess.call(["mystem.exe", "-ldc", text_hash+".in", text_hash+".out"])

    # Читаем текст из выходного файла, удаляем все лишние символы
    output = open(text_hash+".out", 'r', encoding='utf-8')
    text_out = output.read()
    text_out = re.sub(NOT_DIGIT_OR_LETTER, r' ', text_out) # Удаляем все небуквенные и нециферные символы
    output.close()

    # Удаляем лишние файлы
    os.remove(text_hash+".in")
    os.remove(text_hash+".out")
    return text_out

def PystemStemming(text):
    stemmer_rus = Stemmer.Stemmer('russian')
    stemmer_en = Stemmer.Stemmer('english')
    words = text.split(" ")
    words_out = stemmer_en.stemWords(stemmer_rus.stemWords(words))
    return " ".join(words_out)

# Нормализует текст с помощью стеминга
def Steming(text):
    if STEMMER == 'yandex':
        return YandexStemming(text)
    if STEMMER == 'pystem':
        return PystemStemming(text)
    if STEMMER == 'nostem':
        return text

# Окончательно нормализует слово
def SecondaryReplaces(word):
    word = word.replace(u'ё', u'е')
    word = word.replace('_', '')
    word = word.strip()
    return word
