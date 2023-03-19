import requests
from .configurator import configurator as conf, log

def send_to_telegram(message):
    apiURL = f'https://api.telegram.org/bot{conf.TELEGRAM_TOKEN}/sendMessage'
    try:
        response = requests.post(apiURL, json={'chat_id': conf.TELEGRAM_CHAT_ID, 'text': message})
        # log.debug(response.text)
    except Exception as e:
        print(e)
    