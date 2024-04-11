from aiogram import types
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def get_main_menu() -> ReplyKeyboardBuilder:
    kb = ReplyKeyboardBuilder()
    kb.row(
        types.KeyboardButton(text="Вернуться в главное меню")
    )
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder=""
        )