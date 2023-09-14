from .configuration import configuration as conf, log
import requests
import json

def send_to_telegram(message):
    apiURL = f'https://api.telegram.org/bot{conf.TELEGRAM_TOKEN}/sendMessage'
    try:
        if conf.TELEGRAM_SEND:
            response = requests.post(apiURL, json={'chat_id': conf.TELEGRAM_CHAT_ID, 'text': message}, )
            if response.status_code == 200:
                log.debug(f"Response from telegram API: \n{json.dumps(response.json(), indent=2)}\n")
            else:
                response = json.dumps(response.json(), indent=2) if response else '-'
                log.warning(f"Response from telegram API: \n{response}\n")
        else:
            log.debug("No request send to telegram API.")
            
    except Exception as e:
        log.error("Could not send telegram notification.")
        log.exception(e)
    