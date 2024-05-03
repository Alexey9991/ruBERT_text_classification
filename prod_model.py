import torch
from transformers import BertTokenizer
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sqlalchemy import create_engine


# Необходимо указать: 
# путь модели / model_path
# Данные для авторизации в БД
# названия выходного csv файла / output_file_name

# Названия по умолчанию

model_path = ''
data_base_autorization = ''
output_file_name = 'results_text_classification.csv'

# Загрузка модели
rubert_model_name = 'DeepPavlov/rubert-base-cased' 
model = BertForSequenceClassification.from_pretrained(rubert_model_name, num_labels=12)
# Токенизация 
tokenizer = BertTokenizer.from_pretrained(rubert_model_name)

# Загружаем состояния модели
state_dict = torch.load(model_path + 'ruBERT_model_2.pt')
model.load_state_dict(state_dict)
model.eval()

# Создаем запрос на выгрузку данных
engine = create_engine(data_base_autorization)
sql = '''
    SELECT DISTINCT permitted_use_document
    FROM cadastr.parcels_nspd
    WHERE permitted_use_document IS NOT NULL;
'''

# Формируем датафрем
df = pd.read_sql(sql, engine)  # Путь к вашему CSV-файлу

# Список классов
label_map = {'Жилая застройка ИЖС': 0,
 'Предпринимательство': 1,
 'обработать руками': 2,
 'С/Х': 3,
 'Коммунальное обслуживание': 4,
 'Садоводство': 5,
 'Производственная деятельность': 6,
 'Транспортная инфраструктура': 7,
 'Жилая застройка МКД': 8,
 'Хранение автотранспорта': 9,
 'Подсобное хозяйство': 10,
 'Жилая застройка': 11}

# функцию преобразования описания в класс 
def classify_text(text):
    # Токенизация текста
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Передача данных в модель
    with torch.no_grad():
        outputs = model(**inputs)

    # Получение предсказаний
    predictions = torch.argmax(outputs.logits, dim=-1)
    сlass_name = [list(label_map.keys())[i] for i in predictions]

    # Возвращаем номер класса
    return сlass_name[0]



# Применяем функцию классификации к каждому описанию
df['class'] = df['permitted_use_document'].apply(classify_text)

# Сохранение данных
df.to_csv(output_file_name)