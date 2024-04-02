from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from time import sleep as pause
from random import *

from Configs.ConfigParser import *

# Импорт моделей
from Database.Models.Classes.HeadphonesClassDns import HeadphonesCit as HP

# Подключение API базы данных
from Database.PyDBControl import Database


def __create_elem_table__(elements, url, db, driver):
    tmp = HP()

    tmp.name = elements.find_element(By.CSS_SELECTOR, f"h1.{elemTitle}").text

    parent = elements.find_element(By.CSS_SELECTOR, f"div.{parentPrice}")
    tmp.price = parent.find_element(By.CSS_SELECTOR, f"span.{elemPrice}").text
#не количество цен а количество оценок
    parent = elements.find_element(By.CSS_SELECTOR, f"label.{elemParentCountRate}")
    # tmp.countRate = parent.find_element(By.CSS_SELECTOR, f"span.{elemCountRate}").text[:-8]
    tmp.countRate = parent.find_element(By.TAG_NAME, "span").text[:-8]
    parent = elements.find_element(By.CSS_SELECTOR, f"div.{elemParentRate}")
    tmp.averageRate = parent.find_element(By.CSS_SELECTOR, f"span.{elemRate}").text
    pause(randint(5, 12))
    print(tmp)
    driver.get(url + "properties/")

    parent = driver.find_element(By.CSS_SELECTOR, f"div.{elemParentProperty}")
    titles = parent.find_elements(By.CSS_SELECTOR, f"span.{elemPropertyTitle}")
    values = parent.find_elements(By.CSS_SELECTOR, f"span.{elemPropertyValPare}")
    tmps = []

    for i in range(len(titles)):
        tuple_element = (titles[i].text, values[i].find_element(By.TAG_NAME, "span").text)
        tmps.append(tuple_element)
    tmp.properties = str(dict(tmps))

    tmp.url = url

    pause(randint(5, 12))
    driver.get(url + "otzyvy/")

    db.__insert__(nameTable, tmp, HP)

    tmp = None


class Parser:
    def __init__(self, name):
        self.productsUrls = []
        self.dbmodule = Database(name)
        self.dbmodule.__table_create__(nameTable, HP)
        self.driver = webdriver.Chrome()
        self.indexPage = 1

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
            pause(randint(5, 12))
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
