import asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types
from src import modelshell, read_messages

# Объект бота
bot = Bot(token="")
# Диспетчер для бота
dp = Dispatcher(bot)
# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

chat_ids = {}


@dp.message_handler(commands="save")
async def save_users(message: types.Message):
    read_messages.save_val(chat_ids, "chats")
    await message.answer("save users")


@dp.message_handler(commands="load")
async def load_users(message: types.Message):
    global chat_ids
    chat_ids = read_messages.load_val("chats")
    await message.answer("load users")


@dp.message_handler(commands="reset")
async def reset_bot(message: types.Message):
    print("reset")
    read_messages.urls = []
    read_messages.channels = []
    read_messages.set_urls()


@dp.message_handler()
async def set_user(message: types.Message):
    chat_ids[message.from_user.id] = message.from_user
    text = f'Hey (" you re id is {message.message_id}"), {message.from_user.first_name} {message.from_user.last_name}'
    await message.reply(text, reply=False)


async def periodic(sleep_for):
    while True:
        read_messages.set_urls()
        await asyncio.sleep(sleep_for)
        arr = []
        arr += await read_messages.get_all_messages()
        for i in arr:
            model = get_actual_model()
            if i['message'] != "":
                if model.predict(i['message']):
                    for id in chat_ids:
                        await bot.send_message(id, i['message'], disable_notification=True)


def get_actual_model():
    # read config
    with open("resource/configs/config_model.txt", 'r') as handle:
        model_path = handle.readline()
        modelClassifier = modelshell.ModelClassifier(name=model_path, load=True)
        return modelClassifier


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(periodic(2))
    executor.start_polling(dp, skip_updates=True)
