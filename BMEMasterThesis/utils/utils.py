from datetime import datetime as dt
import time

def getTime() -> float:
    return time.time()

def printTime(time: float) -> str:
    return dt.strftime(dt.utcfromtimestamp(time), '%d/%m/%Y %H:%M:%S')