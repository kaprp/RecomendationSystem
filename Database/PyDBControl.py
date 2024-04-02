import sqlite3


def get_attributes(Class):
    return [(attr, str(type(getattr(Class, attr)).__name__)) for attr in dir(Class) if
            not callable(getattr(Class, attr)) and not attr.startswith("__")]


def get_attribute_values(instance):
    return [getattr(instance, attr) for attr in dir(instance) if not callable(getattr(instance, attr))
            and not attr.startswith("__")]


class Database:
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.cur = self.conn.cursor()

    def __to_type__(self, elem):
        return self.cur.execute("SELECT ?", (elem,)).fetchone()[0]

    def __get_all__(self, table):
        self.cur.execute(f"SELECT * FROM {table}")
        data = self.cur.fetchall()
        return data

    def __table_create__(self, nametable, model):
        qur = (f"CREATE TABLE IF NOT EXISTS {nametable} "
               f"( ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, ")
        print(len(get_attributes(model)))
        columns = ', '.join(f"{value} {self.__to_type__(type_value)}"
                            for value, type_value in get_attributes(model))
        qur += columns + ");"
        self.cur.execute(qur)
        self.conn.commit()

    def __insert__(self, nametable, values, model):
        qur = f"INSERT INTO {nametable} ("
        columns = ', '.join(f"{value} "
                            for value, _ in get_attributes(model))
        placeholders = ', '.join('?' for _ in get_attributes(model))
        qur += columns + f") VALUES ({placeholders});"
        values = get_attribute_values(values)
        print(qur)
        self.cur.execute(qur, values)
        self.conn.commit()

    def __delete_condition__(self, nametable, condition):
        qur = f"DELETE FROM {nametable} WHERE {condition};"
        self.cur.execute(qur)
        self.conn.commit()

    def __all_delete__(self, nametable):
        qur = f"DROP TABLE {nametable};"
        self.cur.execute(qur)
        self.conn.commit()

    def __update__(self, nametable, column, new_value, condition, condition_value):
        qur = f"UPDATE {nametable} SET {column} = ? WHERE {condition} ?;"
        self.cur.execute(qur, (new_value, condition_value))
        self.conn.commit()

    def __get_elems__(self, nametable, columns, condition=None):
        query = f"SELECT {columns} FROM {nametable}"
        if condition:
            query += f" WHERE {condition}"
        self.cur.execute(query)
        return [row[0] for row in self.cur.fetchall()]

    def __close__(self):
        self.conn.close()
        self.cur = None

