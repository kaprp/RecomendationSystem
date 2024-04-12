from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

from bot.filters.chat_type import ChatTypeFilter

from bot.keyboards.for_start import get_start_keyboard
from bot.keyboards.main_menu import get_main_menu

from bot.bd.bd import add_user, search_user
from bot.bd.user import User



router = Router()
router.message.filter(
    # Выставляем настройки фильтра на тип чата приватный
    ChatTypeFilter(chat_type=["private"])
) 

class Choosing_city(StatesGroup):
    choosing_city_name = State()
    check_city_name = State()
#Ответ на команду /start
@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот помогающий в выборе техники.",
         reply_markup=get_start_keyboard()
         )
    # Забираем доступную о пользователе информация
    user = User(message.chat.first_name, message.chat.last_name, message.chat.id)
    if search_user(message.chat.id) == None:
        # Если пользователь новый, то добавляем его в БД
        add_user(user)
    else:
        pass

@router.message(F.text.lower() == "вернуться в главное меню")
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Привет! Я бот помогающий в выборе техники.",
         reply_markup=get_start_keyboard()
         )
#Ответ на команду /help или её аналог из клавиатуры
@router.message(Command("help"))
@router.message(F.text.lower() == "список доступных команд")
async def cmd_help(message: Message):
    await message.answer(
    f"В данный момент доступны команды:\n"
    f"/start - открыть главное меню\n"
    f"/assist - получить рекомендацию в выборе наушников\n"
    f"/about_me - узнать обо мне\n",
    parse_mode="HTML",
    reply_markup=get_main_menu()
    )
#Ответ на команду /assist или её аналог из клавиатуры
#@router.message(Command("assist"))
#@router.message(F.text.lower() == "получить рекомендацию")
#async def message_help(message: Message):
#    await message.answer(
#    f"Извините, но в данный момент этот раздел находится в разработке.\n",
#    parse_mode="HTML"
#    )

    
#Ответ на команду /about_me или её аналог из клавиатуры
@router.message(Command("about_me"))
@router.message(F.text.lower() == "узнать обо мне")
async def message_creaters(message: Message):
    await message.answer(
    f"Я нахожусь на стадии тестирования.\n"
    f"Я создан для облегчения выбора инресующей вас техники.\n"
    f"Меня разрабатывают и тестируют студенты из НГТУ им. Р.Е. Алексеева.\n"
    f"Меня создали на языке Python с помощью фреймворка aiogram 3.1.1\n",
    parse_mode="HTML",
    reply_markup=get_main_menu()
    )


#Даём ответ на любое сообщение которое не является командой
@router.message(F.text)
async def message_help(message: Message):
    await message.answer(
    f"Извините, но я не знаю такой команды\n"
    f"Воспользуйтесь командой /help для того, чтобы узнать список известных мне команд.\n"
    f"Или командой /start для того, чтобы открыть главное меню",
    parse_mode="HTML"
    )
