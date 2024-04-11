from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message

from filters.chat_type import ChatTypeFilter 
# Подключаем фильтр на всё кроме личной переписки
router2 = Router()
router2.message.filter(
    ChatTypeFilter(chat_type=[ "group", "supergroup", "channel"])
) 
#Даём ответ на любое сообщение или команду /start
@router2.message(F.text)
@router2.message(Command("start")) 
async def cmd_start(message: Message):
    await message.answer(
    f"Извините, но я работую только в личной переписке\n",
    parse_mode="HTML"
    )
    