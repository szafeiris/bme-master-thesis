import requests
from .configurator import configurator as conf

def send_to_telegram(message):
    apiURL = f'https://api.telegram.org/bot{conf.TELEGRAM_TOKEN}/sendMessage'
    try:
        response = requests.post(apiURL, json={'chat_id': conf.TELEGRAM_CHAT_ID, 'text': message})
        # print(response.text)
    except Exception as e:
        print(e)
    