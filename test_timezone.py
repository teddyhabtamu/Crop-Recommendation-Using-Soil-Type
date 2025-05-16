from datetime import datetime
import pytz
print(datetime.now(pytz.timezone('Africa/Nairobi')).strftime('%Y-%m-%d %I:%M %p %Z'))