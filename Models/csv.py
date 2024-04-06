def read_csv_to_array(file_path):
    data_array = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            row = line.strip().split(',')  # Разделение строки по запятой
            data_array.append(row)
    return data_array

def save_array_to_text_file(data_array, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for row in data_array:
            t = ','.join(row) + '",\n'
            file.write(f'"{t}')

def main():
    csv_file_path = 'negative.csv'  # Замените на путь к вашему CSV файлу
    output_text_file = 'output.txt'  # Имя файла для сохранения текстового массива
    data = read_csv_to_array(csv_file_path)
    save_array_to_text_file(data, output_text_file)
    print("Массив успешно сохранен в текстовый файл:", output_text_file)

if __name__ == "__main__":
    main()