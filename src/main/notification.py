from .configuration import configuration as conf, log
import requests

def send_to_telegram(message):
    apiURL = f'https://api.telegram.org/bot{conf.TELEGRAM_TOKEN}/sendMessage'
    try:
        if conf.TELEGRAM_SEND:
            response = requests.post(apiURL, json={'chat_id': conf.TELEGRAM_CHAT_ID, 'text': message}, )
            log.debug("Response from telegram API: \n", response.raw)
        else:
            log.debug("No request send to telegram API.")
            
    except Exception as e:
        log.error(str(e))
        log.exception(e)
    