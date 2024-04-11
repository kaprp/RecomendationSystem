from aiogram import types
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def y_or_n() -> ReplyKeyboardBuilder:
    kb = ReplyKeyboardBuilder()
    kb.row(
        types.KeyboardButton(text="Да"),
        types.KeyboardButton(text="Нет")
    )
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True,
        )
