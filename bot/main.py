import asyncio
import logging
from aiogram import Bot, Dispatcher
from handlers import start_private, start_not_private,settings,assist



# Запуск процесса поллинга новых апдейтов
async def main():
    # Включаем логирование, чтобы не пропустить важные сообщения
    logging.basicConfig(level=logging.INFO)
    # Объект бота
    bot = Bot(token="6727926334:AAE2adCb4mijRkha7ACkMpYZAzHDUFOjYDI")
    # Диспетчер
    dp = Dispatcher()
    # Подключаем попорядку роутеры
    dp.include_routers(assist.assist_router,settings.settings_router, start_private.router, start_not_private.router2)

    # Пропускаем все накопленные входящие
    await bot.delete_webhook(drop_pending_updates=True)
    # Запускаем бота 
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
