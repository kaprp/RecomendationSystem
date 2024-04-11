from aiogram import types
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def get_setting_keyboard() -> ReplyKeyboardBuilder:
    kb = ReplyKeyboardBuilder()
    kb.row(
        types.KeyboardButton(text="Вернуться в главное меню"),
        types.KeyboardButton(text="Изменить настройки")
    )
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Выберите действие"
        )

def use_link_keyboard() -> ReplyKeyboardBuilder:
    kb = ReplyKeyboardBuilder()
    kb.row(
        types.KeyboardButton(text="Прикреплять"),
        types.KeyboardButton(text="Не прикреплять")
    )
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True
        )

def markets_link_keyboard() -> ReplyKeyboardBuilder:
    kb=ReplyKeyboardBuilder()
    kb.button(text="DNS")
    kb.button(text="Эльдорадо")
    kb.button(text="МВидео")
    kb.button(text="Ситилинк")
    kb.adjust(2,2)
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True
        )

def viewed_marketplaces_keyboard() -> ReplyKeyboardBuilder:
    kb=ReplyKeyboardBuilder()
    kb.button(text="DNS")
    kb.button(text="Эльдорадо")
    kb.button(text="МВидео")
    kb.button(text="Ситилинк")
    kb.adjust(2,2)
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True
        )

def add() -> ReplyKeyboardBuilder:
    kb = ReplyKeyboardBuilder()
    kb.row(
        types.KeyboardButton(text="Добавить"),
        types.KeyboardButton(text="Не добавлять")
    )
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True
        )