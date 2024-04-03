from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from time import sleep as pause
from random import *

from Configs.ConfigParser import *

# Импорт моделей
from Database.Models.Classes.HeadphonesClassDns import Headphones as HP

# Подключение API базы данных
from Database.PyDBControl import Database


def __create_elem_table__(elements, url, db, driver):
    tmp = HP()

    tmp.name = elements.find_element(By.CSS_SELECTOR, f"h1.{elemTitle}").text

    parent = elements.find_element(By.CSS_SELECTOR, f"div.{parentPrice}")
    tmp.price = parent.find_element(By.CSS_SELECTOR, f"span.{elemPrice}").text
    # не количество цен а количество оценок

    pause(randint(7, 15))

    driver.get(url + "otzyvy/")

    parent = driver.find_element(By.CSS_SELECTOR, f"span.{elemParentRate}")
    tmp.averageRate = parent.find_element(By.CSS_SELECTOR, f"span.{elemRate}").text

    parent = driver.find_element(By.CSS_SELECTOR, f"div.{elemParentCountRate}")
    tmp.countRate = parent.find_elements(By.CSS_SELECTOR, f"span.{elemCount}")[0].text[:-8]



    pause(randint(5, 12))

    driver.get(url + "properties/")

    # точка проблемы

    parent = driver.find_element(By.CSS_SELECTOR, f"div.{elemParentProperty}")
    titles = parent.find_elements(By.CSS_SELECTOR, f"span.{elemPropertyTitle}")
    values = parent.find_elements(By.CSS_SELECTOR, f"span.{elemPropertyValPare}")

    for i in range(len(titles)):
        if titles[i].text == "Бренд":
            tmp.brand = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Модель":
            tmp.modelTitle = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Крепление":
            tmp.fasteningMethod = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Тип конструкции":
            tmp.typeConstruction = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Тип соединения":
            tmp.typeConnect = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Регулятор громкости":
            tmp.volumeControl = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Тип регулятора громкости":
            tmp.typeVolControl = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Тип, акустический":
            tmp.typeAcousticType = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Тип звукоизлучателя":
            tmp.typeAcousticDesign = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Диаметр головок излучателей":
            tmp.diametr = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Чувствительность":
            tmp.sense = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Диапазон воспроизводимых частот":
            tmp.freqRange = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Крепление микрофона":
            tmp.microphone = True
            tmp.microphoneMount = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Подключение":
            tmp.typeCon = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Версия Bluetooth":
            tmp.versionBluetooth = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Поддержка профиля":
            tmp.codecs = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Радиус действия":
            tmp.radius = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Подключение кабеля":
            tmp.connectCable = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Зарядка от USB":
            tmp.ChargingUsb = True
        if titles[i].text == "Емкость аккумулятора наушников":
            tmp.battery = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Время работы":
            tmp.time = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Время зарядки наушников":
            tmp.chargingCase = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Питание наушников":
            tmp.typeCharging = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Режим объемного звучания":
            tmp.typeSoundScheme = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Степень защиты":
            tmp.ipyz = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Цвет":
            tmp.mainColor = values[i].find_element(By.TAG_NAME, "span").text
        if titles[i].text == "Вес упаковки(ед)":
            tmp.weight = values[i].find_element(By.TAG_NAME, "span").text

    tmp.url = url

    pause(randint(6, 14))

    db.__insert__(nameTable, tmp, HP)


class Parser:
    def __init__(self, name):
        self.productsUrls = []
        self.dbmodule = Database(name)
        self.dbmodule.__table_create__(nameTable, HP)
        self.driver = webdriver.Chrome()
        self.indexPage = 11

    def __get_urls_elems__(self):
        while self.indexPage < max_page + 1:
            self.driver.get(webresource + str(self.indexPage))
            self.indexPage += 1
            pause(randint(5, 12))
            parent_element = self.driver.find_element(By.CSS_SELECTOR, f"div.{itemParent}")
            elements = parent_element.find_elements(By.CSS_SELECTOR, f"a.{itemPage}")
            for i in elements:
                self.productsUrls.append(i.get_attribute("href"))

    def __print_elems__(self):
        for i in self.productsUrls:
            print(i)

    def __getElms__(self):
        for i in self.productsUrls:
            self.driver.get(i)
            pause(randint(11, 15))
            elements = self.driver.find_element(By.CSS_SELECTOR, f"div.{elemGood}")
            __create_elem_table__(elements, i, self.dbmodule, self.driver)

    def __del__(self):
        self.productsUrls = []
        self.indexPage = 1
        self.driver.close()
        self.driver.quit()


if __name__ == "__main__":
    p = Parser("db.db")
    p.__get_urls_elems__()
    p.__getElms__()
    p.__del__()
