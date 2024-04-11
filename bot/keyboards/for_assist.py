from aiogram import types
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def category_keyboard() -> ReplyKeyboardBuilder:
    kb=ReplyKeyboardBuilder()
    kb.button(text="Наушники")
    kb.button(text="Мышь")
    kb.button(text="Клавиатура")
    kb.button(text="Монитор")
    kb.adjust(2,2)
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True
        )

def category_sett1_keyboard() -> ReplyKeyboardBuilder:
    kb=ReplyKeyboardBuilder()
    kb.button(text="Вкладыши")
    kb.button(text="Внутриканальный")
    kb.button(text="Мониторные")
    kb.button(text="Накладные")
    kb.adjust(2,2)
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True
        )
def category_sett2_keyboard() -> ReplyKeyboardBuilder:
    kb=ReplyKeyboardBuilder()
    kb.button(text="Беспроводные")
    kb.button(text="Проводные")
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True
        )
