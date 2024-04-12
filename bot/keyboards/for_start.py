from aiogram.utils.keyboard import ReplyKeyboardBuilder


def get_start_keyboard() -> ReplyKeyboardBuilder:
    kb=ReplyKeyboardBuilder()
    kb.button(text="Рекомендации по наушникам")
    kb.button(text="Список доступных команд")
    kb.button(text="Узнать обо мне")
    kb.adjust(2,1)
    return kb.as_markup(
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Что вас интересует?"
        )
    