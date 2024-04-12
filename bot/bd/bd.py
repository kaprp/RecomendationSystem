import pyodbc
import logging
from bot.bd.user import User

#logging.basicConfig(level=logging.ERROR, filename="Errors.log",filemode="w",
#                    format="%(asctime)s %(levelname)s %(message)s")


# Адрес до БД
path_bd = "bd/bd.accdb"
# Подключение к БД
config_connection = "Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=C:%s;" % (path_bd)
# Добавление пользователя в БД
def add_user(user:User):
    try:
       connect = pyodbc.connect(config_connection)
       cursor = connect.cursor() 


       cursor.execute("insert into Пользователи (FirstName, LastName, ID, Status) values (?,?,?,?)", user.name, user.lastname, user.id, user.status)
       connect.commit()
       #Забираем БДшный айди пользователя
       cursor.execute("select UserID from Пользователи where ID = ? ", user.id)
       user_id = int(cursor.fetchone().UserID)
       #Задаём стандартные настройки
       cursor.execute("insert into Настройки (ID, use_link, markets_link, viewed_marketplaces) values (?)", user_id)
       connect.commit()

       cursor.close()
       connect.close()
    except pyodbc.Error as err:
        logging.error("Can't connection to DB while try add user", exc_info=True)
        print("Error in connection while try add user")
# Удаление пользователя из БД
# Требует доработки удаления настроек пользователя
def delete_user(id):
    try:
       connect = pyodbc.connect(config_connection)
       cursor = connect.cursor() 

       cursor.execute("delete from Пользователи where id = ?", id)
       connect.commit()

       cursor.close()
       connect.close()
    except pyodbc.Error as err:
        logging.error("Can't connection to DB while try delete user", exc_info=True)
        print("Error in connection while try delete user")


#Поиск пользователя в БД и сбор данных о нём
def search_user(id:int):
    try:
       connect = pyodbc.connect(config_connection)
       cursor = connect.cursor() 

       cursor.execute("select FirstName, LastName, ID, Status from Пользователи where ID = ? ", str(id))
       row = cursor.fetchone()
       if row == None:
           cursor.close()
           connect.close()
           return None
       else:
        user = User(row.FirstName, row.LastName, row.ID, row.Status)

        cursor.close()
        connect.close()

        return user
    except pyodbc.Error as err:
        logging.error("Can't connection to DB while try search user", exc_info=True)
        print("Error in connection while try search user")
#Проверка наличия пользователя в БД
def check_user(id:int):
    try:
       connect = pyodbc.connect(config_connection)
       cursor = connect.cursor() 

       cursor.execute("select ID from Пользователи where ID = ? ", str(id))
       row = cursor.fetchone()
       if row == None:
           cursor.close()
           connect.close()
           return False
       else:

        cursor.close()
        connect.close()

        return True
    except pyodbc.Error as err:
        logging.error("Can't connection to DB while try search user", exc_info=True)
        print("Error in connection while try search user")

#Извлечение сведений о настройках пользователя
def get_settings(id:int):
    try:
        connect = pyodbc.connect(config_connection)
        cursor = connect.cursor()

        #Забираем БДшный айди пользователя
        cursor.execute("select UserID from Пользователи where ID = ? ", id)
        user_id = int(cursor.fetchone().UserID)

        cursor.close()
        connect.close()

    except pyodbc.Error as err:
        logging.error("Can't connection to DB while try get user's settings", exc_info=True)
        print("Error in connection while try get user's settings")
