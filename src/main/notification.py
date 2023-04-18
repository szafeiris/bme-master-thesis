import requests
from .configurator import configurator as conf, log

def send_to_telegram(message):
    apiURL = f'https://api.telegram.org/bot{conf.TELEGRAM_TOKEN}/sendMessage'
    try:
        if conf.TELEGRAM_SEND:
            response = requests.post(apiURL, json={'chat_id': conf.TELEGRAM_CHAT_ID, 'text': message}, )
    except Exception as e:
        log.error(str(e))
        log.exception(e)
    