from aiogram import Router, F
from aiogram.fsm.state import StatesGroup, State
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

from filters.chat_type import ChatTypeFilter 
from keyboards.for_settings import *
from keyboards.for_start import get_start_keyboard
from bd.bd import get_settings, change_settings
from bd.settings import Settings
from handlers.start_private import router




settings_router = Router()
settings_router.message.filter(
    # Выставляем настройки фильтра на тип чата приватный
    ChatTypeFilter(chat_type=["private"])
) 

class Change_settings(StatesGroup):
    change = State()
    use_link = State()
    markets_link = State()
    viewed_marketplaces = State()
    add_viewed_marketplaces = State()
    finish = State()

use_l = ["прикреплять", "не прикреплять"] #Варианты допустимых ответ при выборе прикрепления ссылок
markets_l_use = ["dns"]       # можно прикряплять ссылки только на маркетплейсы из этого списка
markets_l_not_use = [ "эльдорадо", "мвидео", "ситилинк"]# нельзя прикряплять ссылки на маркетплейсы из этого списка
viewed_m_use = ["dns", "эльдорадо"]      # список используемых для анализа маркетплесов
viewed_m_not_use = ["мвидео", "ситилинк"]# список не используемых для анализа маркетплесов


@settings_router.message(Command("settings"))
@settings_router.message(F.text.lower() == "пользовательские настройки")
async def settings(message: Message, state: FSMContext):
    #Достаём из БД настройки пользователя
    user_settings = get_settings(message.chat.id)

    switch = {
        True:"да",
        False:"нет"
    }

    await message.answer(
    f"Ваши текущие настройки:\n"
    f"Прикреплять ссылки на товар - {switch.get(user_settings.use_link)}\n"
    f"Ссылки на товар на площадках {user_settings.markets_link}\n"
    f"При оценки использовать отзывы с площадок {user_settings.viewed_marketplaces}\n",
    parse_mode="HTML",
    reply_markup=get_setting_keyboard(),
    )
    await state.set_state(Change_settings.change)# Переводим состояние пользователя в выбор настроек

#Использовалось при тестировании, можно удалить
#@settings_router.message(Command(commands=["pr"]))
#async def print_date(message: Message, state: FSMContext):
#    user_data = await state.get_data()
#    await message.answer(
#        text=f"{user_data}"
#    )

# На любом этапе пользователь может отменить изменение настроек
@settings_router.message(Command(commands=["cancel"]))
@settings_router.message(F.text.lower() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear() # обнуляем/очищаем состояние пользователя
    await message.answer(
        text="Действие отменено",
        reply_markup=get_start_keyboard()
    )

@settings_router.message(Command(commands=["сhange_settings"]))
@settings_router.message(F.text.lower() == "изменить настройки")
async def use_link(message: Message, state: FSMContext):
    await message.answer(
        text="Прикреплять ссылки к рекомендованному товару? \n",
        reply_markup=use_link_keyboard()
    )
    await state.set_state(Change_settings.use_link)


@settings_router.message(F.text.lower().in_(use_l),Change_settings.use_link)
async def market_link(message: Message, state: FSMContext):
    await state.update_data(use_link=message.text.lower())
    await message.answer(
        text="С какого маркетплейса прикреплять ссылки? \n",
        reply_markup=markets_link_keyboard()
    )
    await state.set_state(Change_settings.markets_link)


@settings_router.message(Change_settings.use_link)
async def market_link_er(message: Message, state: FSMContext):
    await message.answer(
        text="Я не понял вашего ответа.\nПопробуйте снова или прекратите изменение настроек командой /cancel",
        reply_markup=use_link_keyboard()
    )

@settings_router.message(F.text.lower().in_(markets_l_use),Change_settings.markets_link)
async def viewed_markets(message: Message, state: FSMContext):
    if message.text.lower() == "dns":
        market = "DNS"
    else:
        market = message.text.title()
    await state.update_data(market_link=market)
    await message.answer(
        text="При оценке товара с каких маркетплейсов использовать отзывы? \n",
        reply_markup=viewed_marketplaces_keyboard()
    )
    await state.set_state(Change_settings.viewed_marketplaces)

@settings_router.message(F.text.lower().in_(markets_l_not_use),Change_settings.markets_link)
async def viewed_markets(message: Message, state: FSMContext):
    await message.answer(
        text=f"В данный момент этот маркетплейс недоступен. Разработчики работают над этой проблемой!\nВ данный момент доступен только DNS",
        reply_markup=markets_link_keyboard()
    )

@settings_router.message(Change_settings.markets_link)
async def viewed_markets_er(message: Message, state: FSMContext):
    await message.answer(
        text="Я не понял вашего ответа.\nПопробуйте снова или прекратите изменение настроек командой /cancel",
        reply_markup=markets_link_keyboard()
    )


@settings_router.message(F.text.lower().in_(viewed_m_use),Change_settings.viewed_marketplaces)
async def viewed_markets(message: Message, state: FSMContext):
    if message.text.lower() == "dns":
        market = "DNS"
    else:
        market = message.text.title()
    await state.update_data(viewed_markets=market)
    await message.answer(
        text=f"Желаете добавить ещё маркетплейсы?",
        reply_markup=add()
    )
    await state.set_state(Change_settings.add_viewed_marketplaces)

@settings_router.message(F.text.lower() == "добавить",Change_settings.add_viewed_marketplaces)
async def viewed_markets(message: Message, state: FSMContext):
    await message.answer(
        text="Какой маркетплейс добавить? \n",
        reply_markup=viewed_marketplaces_keyboard()
    )

@settings_router.message(F.text.lower().in_(viewed_m_use),Change_settings.add_viewed_marketplaces)
async def viewed_markets(message: Message, state: FSMContext):
    user_data = await state.get_data()
    if message.text.lower() == "dns":
        market = "DNS"
    else:
        market = message.text.title()
    if market in user_data["viewed_markets"]:
        await message.answer(
        text=f"Этот маркетплейс уже указан. Желаете добавить ещё маркетплейсы?",
        reply_markup=add()
        )
        return
    await state.update_data(viewed_markets=user_data["viewed_markets"]+", "+market)
    await message.answer(
        text=f"Желаете добавить ещё маркетплейсы?",
        reply_markup=add()
    )
    await state.set_state(Change_settings.add_viewed_marketplaces)

@settings_router.message(F.text.lower().in_(viewed_m_not_use),Change_settings.viewed_marketplaces)
@settings_router.message(F.text.lower().in_(viewed_m_not_use),Change_settings.add_viewed_marketplaces)
async def viewed_markets(message: Message, state: FSMContext):
    await message.answer(
        text=f"В данный момент этот маркетплейс недоступен. Разработчики работают над этой проблемой!\nВ данный момент доступен только DNS",
        reply_markup=viewed_marketplaces_keyboard()
    )


@settings_router.message(F.text.lower() == "не добавлять",Change_settings.viewed_marketplaces)
@settings_router.message(F.text.lower() == "не добавлять",Change_settings.add_viewed_marketplaces)
async def viewed_markets(message: Message, state: FSMContext):
    switch = {
        "прикреплять":True,
        "не прикреплять":False
    }
    switch2 = {
        True:"да",
        False:"нет"
    }
    user_data = await state.get_data()
    market_l = user_data["market_link"]
    #viewed_markets = ", ".join(word.capitalize() for word in user_data["viewed_markets"].split(', '))
    user_settings = Settings(switch.get(user_data["use_link"]),market_l,user_data["viewed_markets"])
    await message.answer(
        text="Новые настройки сохранены! \n",
        reply_markup=get_start_keyboard()
    )
    change_settings(message.chat.id,user_settings)
    await message.answer(
    f"Ваши текущие настройки:\n"
    f"Прикреплять ссылки на товар - {switch2.get(user_settings.use_link)}\n"
    f"Ссылки на товар на площадках {user_settings.markets_link}\n"
    f"При оценки использовать отзывы с площадок {user_settings.viewed_marketplaces}\n",
    parse_mode="HTML"
    )
    await state.clear()


@settings_router.message(Change_settings.viewed_marketplaces)
@settings_router.message(Change_settings.add_viewed_marketplaces)
async def viewed_markets_er(message: Message, state: FSMContext):
    await message.answer(
        text="Я не понял вашего ответа.\nПопробуйте снова или прекратите изменение настроек командой /cancel",
        reply_markup=viewed_marketplaces_keyboard()
    )