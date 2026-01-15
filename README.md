pip install python-dotenv psycopg2
Requirement already satisfied: python-dotenv in /opt/anaconda3/lib/python3.12/site-packages (0.21.0)
Requirement already satisfied: psycopg2 in /opt/anaconda3/lib/python3.12/site-packages (2.9.11)
Note: you may need to restart the kernel to use updated packages.
pip install sqlalchemy==2.0
Requirement already satisfied: sqlalchemy==2.0 in /opt/anaconda3/lib/python3.12/site-packages (2.0.0)
Requirement already satisfied: typing-extensions>=4.2.0 in /opt/anaconda3/lib/python3.12/site-packages (from sqlalchemy==2.0) (4.11.0)
Requirement already satisfied: greenlet!=0.4.17 in /opt/anaconda3/lib/python3.12/site-packages (from sqlalchemy==2.0) (3.0.1)
Note: you may need to restart the kernel to use updated packages.

1. Для начала импортируем все библиотеки, которые могут понадобиться
2. import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import sqlite3
import requests
import sqlalchemy
from typing import List, Optional
from sqlalchemy import Column, String, Text, ForeignKey, func, exists
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

2. Подключаемся к supabase
3. USER = 'postgres.cwjwifniusnhiltixqff'
PASSWORD = 'n19ht15DGN$'
HOST = 'aws-1-eu-central-1.pooler.supabase.com'
PORT = '6543'
DBNAME = 'postgres'
DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"
engine = sqlalchemy.create_engine(DATABASE_URL)
os.getcwd()
'/Users/oksanatebieva/Downloads'
sqlite_connector = sqlite3.connect('test.db')

3. Создаем подключение к API Всемирного банка
4. url = 'https://api.worldbank.org/v2/country'  # endpoint (API метод) для получения стран

params = {
            'format': 'json',  # Формат JSON
            'per_page': 296  # Количество записей на странице (296 стран всего)
}
response = requests.get(url, params=params)  # Получаем данные с endpoint
data = response.json()  # Ответ конвертируем в JSON
data[0]
{'page': 1, 'pages': 1, 'per_page': '296', 'total': 296}
data[1][0]
{'id': 'ABW',
 'iso2Code': 'AW',
 'name': 'Aruba',
 'region': {'id': 'LCN',
  'iso2code': 'ZJ',
  'value': 'Latin America & Caribbean '},
 'adminregion': {'id': '', 'iso2code': '', 'value': ''},
 'incomeLevel': {'id': 'HIC', 'iso2code': 'XD', 'value': 'High income'},
 'lendingType': {'id': 'LNX', 'iso2code': 'XX', 'value': 'Not classified'},
 'capitalCity': 'Oranjestad',
 'longitude': '-70.0167',
 'latitude': '12.5167'}
 countries = data[1]
 
 4. Создаем датафреймы, вытаскиваем словари из JSON
 pd.DataFrame(countries).sample(3)
temp_df = pd.json_normalize(countries)
temp_df.head(5)
temp_df.to_csv('temp_df.csv', encoding='utf-8') # сохраняем файл
def fetch_worldbank_data(indicators: List[str],
                         countries: List[str],
                         start_year: int,
                         end_year: int,
                         language: str = 'en') -> pd.DataFrame:

    '''
    Получает данные показателей из API Всемирного банка.

    Args:
        indicators: Список кодов показателей (например, ['EN.GHG.CO2.IC.MT.CE.AR5'])
        countries: Список кодов стран в формате ISO 2 или ISO 3 (например, ['RU', 'US'])
        start_year: Год начала периода
        end_year: Год окончания периода
        language: Язык данных ('en', 'ru' и т.д.)

    Returns:
        pandas.DataFrame с данными показателей
    '''

    base_url = "https://api.worldbank.org/v2" # endpoint API - адрес по которому мы обращаемся за данными

    # преобразовываем страны в строку с разделителем точка с запятой (для запроса данных)
    countries_str = ';'.join(countries)
    # список для хранения данных о показателях
    all_data = []

    try:
        for indicator in indicators:
            # Формируем URL для запроса
            url = f"{base_url}/{language}/country/{countries_str}/indicator/{indicator}"
            params = {
                'format': 'json',
                'date': f"{start_year}:{end_year}",
                'per_page': 10000  # Большое значение для получения всех данных
            }

            # Выполняем запрос к API
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # API возвращает массив, где первый элемент - метаданные, второй - данные
    if len(data) > 1 and isinstance(data[1], list):
                for item in data[1]:
                    if item.get('value') is not None:
                        all_data.append({
                            'country': item['country']['value'],
                            'country_code': item['countryiso3code'],
                            'indicator': item['indicator']['value'],
                            'indicator_code': item['indicator']['id'],
                            'year': int(item['date']),
                            'value': item['value']
                        })
    # Создаем DataFrame
        df = pd.DataFrame(all_data)

        if df.empty:
            print("Предупреждение: Не получено данных для указанных параметров")
            return df

        return df

    # Обрабатываем возможные ошибки при работе с АПИ
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
        return pd.DataFrame()
    except (KeyError, IndexError, ValueError) as e:
        print(f"Ошибка при обработке данных: {e}")
        return pd.DataFrame()
def get_countries(
        url: str='https://api.worldbank.org/v2/country', 
        params: dict={'format': 'json', 'per_page': 296}
        ) -> pd.DataFrame:
    
    # проверяем на наличие переданных параметров в функцию
    if params is not None:
        response = requests.get(url, params=params)

    else:
        # если параметры не переданы, используем только endpoint
        response = requests.get(url)
    
    # переделываем ответ в JSON
    data = response.json()

    # нормализуем JSON в датафрейм pandas
    countries = pd.json_normalize(data[1])

    # обрабатываем столбцы: чтобы привести их к snake_case необходимо заменить
    # точки и заглавные буква на нижнее подчеркивание (в принципе - это хардкод,
    # но данные всемирного банка вряд ли будут меняться, поэтому можно)
    countries.columns = (
        countries.columns.str.replace('.', '_')
        .str.replace('C', '_c')
        .str.replace('L', '_l')
        .str.replace('T', '_t')
        )
     # убираем агрегированные регионы (например, африка или юго-восточная азия)
    countries = countries[countries['lending_type_value'] != 'Aggregates']

    return countries

countries_df = get_countries()
countries_df.head(5)
countries_df.to_csv('countries_df.csv', encoding='utf-8')
# Получаем данные по выбросам CO2 для РФ, США, Китая и Японии за 2015-2024 год
indicators = ['EN.GHG.CO2.IC.MT.CE.AR5']  # Промышленные выбросы CO2
countries = ['RU', 'US', 'CN', 'JP']  # Россия, США, Китай, Япония
start_year = 2015
end_year = 2024

indicators_df = fetch_worldbank_data(indicators, countries, start_year, end_year)

indicators_df.to_csv('indicators_df.csv', encoding='utf-8')
indicators_df.head(6)
# Получаем данные по численности населения для РФ, США, Китая и Японии за 2015-2024 год
indicators_pop = ['SP.POP.TOTL']
countries = ['RU', 'US', 'CN', 'JP']
start_year = 2015
end_year = 2024

indicators_pop = fetch_worldbank_data(indicators_pop, countries, start_year, end_year)

indicators_pop.to_csv('indicators_pop.csv', encoding='utf-8')
indicators_pop.head(6)


Посмотрим распределение стран по уровню дохода

s = (
    temp_df.loc[temp_df['incomeLevel.value'] != "Aggregates"]     
             .groupby('incomeLevel.value')
             .size()
)

ax = s.plot(kind='pie', autopct='%1.0f%%', figsize=(6, 6), legend=False)
ax.set(ylabel=None)
plt.title('Распределение стран по уровню дохода')
plt.show()
40% стран имеют высокий показатель входящих средств, кажется, очень высокий показатель



Посмотрим, в какой стране был самый высокий выброс СО2 в 2024 году

# Берём данные только за 2024 год
data_2024 = indicators_df[indicators_df['year'] == 2024]

# Суммируем выбросы по странам
country_2024 = (
    data_2024
    .groupby('country')['value']
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(7, 3))
country_2024.plot(kind='bar', color='skyblue')

plt.title('Выбросы CO2 по странам в 2024 году')
plt.ylabel('Выбросы CO2')
plt.xlabel('Страна')
plt.xticks(rotation=33)
plt.tight_layout()
plt.show()
Китай уверенно лидирует по выбросам углекислого газа


А теперь сравним численность населения по странам и годам
# Делаем сводную таблицу: строки — годы, столбцы — страны, значения — население
pop_pivot = (
    indicators_pop
    .pivot(index='year', columns='country', values='value')
    .sort_index()
)

plt.figure(figsize=(8, 4))
sns.heatmap(pop_pivot.T, annot=False, fmt='.0f', cmap='Blues', cbar_kws={'label': 'Население, млн чел.'})
plt.title('Численность населения')
plt.xlabel('Год')
plt.ylabel('Страна')
plt.tight_layout()
plt.show()

