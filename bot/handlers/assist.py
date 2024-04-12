from aiogram import Router, F
from aiogram.fsm.state import StatesGroup, State
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

from bot.filters.chat_type import ChatTypeFilter
from bot.keyboards.for_assist import *
from bot.keyboards.for_start import get_start_keyboard

import logging
from dataclasses import dataclass
from bot.Database.PyDBControl import Database
from bot.Models.Model import *
import string

b = Database("db.db")
data = b.__get_all__("Goods")

data_handler = Data()
data_handler.load_wordvec_model("testing.bin")
data_handler.load_keras_model("keras_model2.keras")


@dataclass
class FiltersData:
    mincost: float = 0.0
    maxcost: float = 0.0
    type: str = ""
    typeconnect: str = ""


assist_router = Router()
assist_router.message.filter(
    # Выставляем настройки фильтра на тип чата приватный
    ChatTypeFilter(chat_type=["private"])
)


def check_empty(text):
    strs = text
    if len(strs.replace(" ", "").replace("\n", "")) == 0:
        return "0"
    else:
        return text


def get_result(text):
    text = replace_punctuation_with_spaces(text)
    text = check_empty(text)
    print(text)
    if data_handler.keras_model.predict(data_handler.get_common_vectors([text])) >= 0.5:
        return data_handler.keras_model.predict(data_handler.get_common_vectors([text]))
    else:
        return data_handler.keras_model.predict(data_handler.get_common_vectors([text])) - 1


def replace_punctuation_with_spaces(text):
    # Получить все знаки препинания
    punctuation = string.punctuation

    # Заменить все знаки препинания на пробелы
    for char in punctuation:
        text = text.replace(char, ' ')

    return text


class Assist_settings(StatesGroup):
    assist = State()
    category = State()
    budget_min = State()
    budget_max = State()
    category_sett1 = State()
    category_sett2 = State()
    category_sett3 = State()
    finish = State()


category_use = ["наушники"]  # можно прикряплять ссылки только на маркетплейсы из этого списка
# category_not_use = [ "монитор", "клавиатура", "мышь"]# нельзя прикряплять ссылки на маркетплейсы из этого списка
category_sett1 = ["вкладыши", "внутриканальные", "мониторные", "накладные"]
category_sett2 = ["беспроводные", "проводные"]
f = FiltersData()

"""@assist_router.message(Command("assist"))
@assist_router.message(F.text.lower() == "получить рекомендацию")
async def assist(message: Message, state: FSMContext):
    await message.answer(
    f"Выберите одну из доступных категорий",
    parse_mode="HTML",
    reply_markup=category_keyboard(),
    )
    await state.set_state(Assist_settings.category)# Переводим состояние пользователя в выбор настроек для рекомендаций
"""


# Использовалось при тестировании, можно удалить
# @assist_router.message(Command(commands=["pr"]))
# async def print_date(message: Message, state: FSMContext):
#    user_data = await state.get_data()
#    await message.answer(
#        text=f"{user_data}"
#    )

# На любом этапе пользователь может отменить получение рекомендаций
@assist_router.message(Command(commands=["cancel"]))
@assist_router.message(F.text.lower() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()  # обнуляем/очищаем состояние пользователя
    await message.answer(
        text="Действие отменено",
        reply_markup=get_start_keyboard()
    )


@assist_router.message(Command("assist"))
@assist_router.message(F.text.lower() == "рекомендации по наушникам")
# @assist_router.message(F.text.lower().in_(category_use),Assist_settings.category)
async def category_catch(message: Message, state: FSMContext):
    await state.update_data(category="наушники")
    await message.answer(
        text="Выберите минимальную стоимость товара \n"
    )
    await state.set_state(Assist_settings.budget_min)


# Временно отключено до дальнейшего развития.
# @assist_router.message(F.text.lower().in_(category_not_use),Assist_settings.category)
# async def category_catch_not_use(message: Message, state: FSMContext):
#    await state.update_data(category=message.text.lower())
#    await message.answer(
#        text="В данный момент эта категория недоступна\nПриносим свои извинения.\n"
#        "Выберите другую категорию. Или вернитесь в главное меню с помощью команды /cancel",
#        reply_markup=category_keyboard()
#    )
#    await state.set_state(Assist_settings.category)
#
# @assist_router.message(Assist_settings.category)
# async def category_catch_er(message: Message, state: FSMContext):
#    await message.answer(
#        text="Я не понял вашего ответа.\nПопробуйте снова или вернитесь в главное меню с помощью команды /cancel",
#        reply_markup=category_keyboard()
#    )

@assist_router.message(Assist_settings.budget_min)
async def budget_min_catch(message: Message, state: FSMContext):
    try:
        min = int(message.text)
        if min < 0:
            await message.answer(
                text="Число должно быть больше или равно 0, попробуйте снова \n"
            )
        else:
            f.mincost = float(min)
            await state.update_data(budget_min=min)
            await message.answer(
                text="Выберите максимальную стоимость товара \n"
            )
            await state.set_state(Assist_settings.budget_max)
    except ValueError:
        logging.error("Budget must be int", exc_info=True)
        print("Error while user try input budget_min")
        await message.answer(
            text="Вы ввели не число.\n"
                 "Попробуйте снова или вернитесь в главное меню с помощью команды /cancel"
        )


@assist_router.message(Assist_settings.budget_max)
async def budget_max_catch(message: Message, state: FSMContext):
    try:
        max = int(message.text)
        user_data = await state.get_data()
        min = int(user_data["budget_min"])
        if max < min:
            await message.answer(
                text=f"Число должно быть больше мин стоимости товара равно {min}, попробуйте снова \n"
                     "Или вернитесь в главное меню с помощью команды /cancel"
            )
        else:
            f.maxcost = float(max)
            recs = [record for record in data if f.mincost <= float(str(record[18]).replace(" ", "")) <= f.maxcost]

            records = [(i, get_result(str(i[20])) if i[20] and i else 0.5) for i in recs]

            recs = sorted(records,
                   key=lambda x: round(
                       (float((x[1]))) * float(str(x[0][8])) + float(str(x[0][2])) * float(str(x[0][8])), 3))

            await state.update_data(category_sett2=message.text.lower())
            # Тут должна быть функция которая получает список товаров
            if len(recs) > 0:
                await message.answer(
                    text="Вот список товаров в соответствии с вашим запросом (от худшего к лучшему):"
                )
                for i in recs:
                    await message.answer(
                        text=f"Название : {i[0][17]}\n"
                             f"Общий рейтинг : {(i[1] + 1) * float(str(i[0][8])) + float(str(i[0][2])) * float(str(i[0][8]))}\n"
                             f"Цена: {i[0][18]}\n"
                             f"{i[0][32]}"
                    )
            else:
                await message.answer(text="К сожалению ничего не найдено")
            await state.clear()
    except ValueError:
        logging.error("Budget must be int", exc_info=True)
        print("Error while user try input budget_max")
        await message.answer(
            text="Вы ввели не число.\n"
                 "Попробуйте снова или вернитесь в главное меню с помощью команды /cancel"
        )


@assist_router.message(Assist_settings.category_sett1)
async def category_sett1_catch_er(message: Message, state: FSMContext):
    await message.answer(
        text="Я не понял вашего ответа.\nПопробуйте снова или вернитесь в главное меню с помощью команды /cancel",
        reply_markup=category_sett1_keyboard()
    )



@assist_router.message(Assist_settings.category_sett2)
async def category_sett1_catch_er(message: Message, state: FSMContext):
    await message.answer(
        text="Я не понял вашего ответа.\nПопробуйте снова или вернитесь в главное меню с помощью команды /cancel",
        reply_markup=category_sett2_keyboard()
    )
