import datetime

class TimeUtil:
    def now() -> datetime.datetime:
        return datetime.datetime.now()
    
    def format(time: datetime.datetime | datetime.timedelta = now()) -> str:
        if isinstance(time, datetime.timedelta):
            return str(time)
        
        return time.strftime('%d/%m/%Y %H:%M:%S')
    
    def elapsed(start: datetime.datetime, end: datetime.datetime = now()) -> str:
        return format(end - start)
    
    