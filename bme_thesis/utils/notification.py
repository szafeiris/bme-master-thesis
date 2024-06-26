from bme_thesis.utils.settings import bmeThesisSettings
from bme_thesis.logger import getLogger

import requests
import json
        
       
def sendNotification(message: str):
    log = getLogger(__name__)
    apiURL = f'https://api.telegram.org/bot{bmeThesisSettings.telegram_token}/sendMessage'
    try:
        if bmeThesisSettings.telegram_send:
            response = requests.post(apiURL, json={'chat_id': bmeThesisSettings.telegram_chat_id, 'text': message}, )
            if response.status_code == 200:
                log.debug(f"Response from telegram API: \n{json.dumps(response.json(), indent=2)}\n")
            else:
                response = json.dumps(response.json(), indent=2) if response else '-'
                log.warning(f"Response from telegram API: \n{response}\n")
        else:
            log.warning("No request send to telegram API.")
    except Exception as e:
        log.error("Could not send telegram notification.")
        log.exception(e)
    