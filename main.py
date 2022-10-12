import asyncio
import datetime
import logging
import pickle
from aiogram import Bot, Dispatcher, executor, types
import alg
import readAc, lstmModel

# Объект бота
bot = Bot(token="5160756931:AAEbzmy3Tm41NDscVfq_2R0YcIIBKeWc1xU")
# Диспетчер для бота
dp = Dispatcher(bot)
# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

sp = [
    'https://t.me/kommersant',
    'https://t.me/markettwits',
    'https://t.me/prime1',
    'https://t.me/bcs_express',
    'https://t.me/vedomosti',
    'https://t.me/AbzalovD',
    'https://t.me/newssmartlab'
]

chat_ids = {}


@dp.message_handler(commands="save")
async def save_users(message: types.Message):
    readAc.save_val(chat_ids, "chats")
    await message.answer("save users")


@dp.message_handler(commands="load")
async def load_users(message: types.Message):
    global chat_ids
    chat_ids = readAc.load_val("chats")
    await message.answer("load users")


# Хэндлер на команду /test1
@dp.message_handler(commands="add")
async def cmd_test1(message: types.Message):
    stringVal = message.text[4::].strip()
    arr = await readAc.get_all_messages(stringVal)
    for i in arr:
        if lstmModel.predict(i['message']):
            await message.answer(
                i['message']
            )


@dp.message_handler(commands="reset")
async def reset_bot(message: types.Message):
    print("reset")
    readAc.urls = []
    readAc.channels = []
    readAc.set_urls()


# Хэндлер на команду /test1
@dp.message_handler(commands="news")
async def cmd_test1(message: types.Message):
    arr = []
    print("gf")
    for v in sp:
        arr += await readAc.get_all_messages(v)
    for i in arr:
        if i['message'] != "":
            if lstmModel.predict(i['message']):
                await message.answer(
                    i['message']
                )



@dp.message_handler()
async def set_user(message: types.Message):
    chat_ids[message.from_user.id] = message.from_user
    text = f'Hey (" you re id is {message.message_id}"), {message.from_user.first_name} {message.from_user.last_name}'
    await message.reply(text, reply=False)


async def periodic(sleep_for):
    while True:
        readAc.set_urls()
        await asyncio.sleep(sleep_for)
        arr = []
        arr += await readAc.get_all_messages(sp)
        for i in arr:
            if i['message'] != "":
                if lstmModel.predict(i['message']):
                    for id in chat_ids:
                        await bot.send_message(id, i['message'], disable_notification=True)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(periodic(40))
    executor.start_polling(dp, skip_updates=True)
