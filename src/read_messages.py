import configparser
import pickle
import pandas
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest

# Считываем учетные данные
config = configparser.ConfigParser()
config.read('config.ini')

print(config)
# Присваиваем значения внутренним переменным
api_id = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']
username = config['Telegram']['username']

client = TelegramClient(username, api_id, api_hash)
client.start()

st_values = set()


async def dump_all_messages(channel):
    """Записывает все сообщения каналов/чатов"""

    offset_msg = 0  # номер записи, с которой начинается считывание
    limit_msg = 100  # максимальное число записей, передаваемых за один раз

    all_messages = []  # список всех сообщений
    total_messages = 0
    total_count_limit = 5

    val = False
    while True:
        history = await client(GetHistoryRequest(
            peer=channel,
            offset_id=offset_msg,
            offset_date=None, add_offset=0,
            limit=min(limit_msg, total_count_limit), max_id=0, min_id=0,
            hash=0))
        if not history.messages:
            break
        messages = history.messages
        # val = datetime.now()
        for message in messages:
            if message.to_dict()['_'] == 'Message':
                if message.to_dict()['message'] == 'https://youtu.be/cEq_qmrpNu4':
                    val = True
                    break
                if message.to_dict()['id'] not in st_values:
                    st_values.add(message.to_dict()['id'])
                    all_messages.append(message.to_dict())
                else:
                    val = True
                    break
        if val:
            break
        offset_msg = messages[len(messages) - 1].id
        total_count_limit -= limit_msg
        if total_count_limit <= 0:
            break
    return all_messages


urls = []


def set_urls():
    global urls

    urls = []

    # получим объект файла
    file1 = open("resource/configs/config.txt", "r")

    while True:
        # считываем строку
        line = file1.readline()

        # прерываем цикл, если строка пустая
        if not line:
            break

        if line[-1] == '\n':
            urls.append(line.rstrip(line[-1]))
        else:
            urls.append(line)
        # выводим строку

    # закрываем файл
    file1.close()


channels = []


async def add_channel(url):
    return await client.get_entity(url)


async def get_all_messages():
    global channels
    if len(channels) == 0:
        channels.extend(await client.get_entity(urls))
    arr = []
    for ch in channels:
        arr += await dump_all_messages(ch)
    return arr


def save_val(array, name):
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_val(name):
    with open(name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
        return val


def get_all_messages_with_():
    with client:
        return client.loop.run_until_complete(get_all_messages())

def save_train_data():
    """
    this function for update train data for model
    """
    arr = get_all_messages_with_()

    nArr = []
    yArr = []
    i = 0
    while i < len(arr):
        nArr.append(arr[i]['message'])
        print(nArr[-1])
        if i + 1 >= len(arr) or (arr[i + 1]['message'] != '0' and arr[i + 1]['message'] != '1'):
            yArr.append('0')
            i += 1
        else:
            yArr.append(arr[i + 1]['message'])

            i += 2

    date = pandas.DataFrame(data=
                            {'data': nArr,
                             'target': yArr})
    print(len(nArr), len(yArr))
    date.to_csv("all_messages.csv")
