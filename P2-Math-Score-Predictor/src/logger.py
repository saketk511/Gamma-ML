import logging  # Import the logging module to handle logging of events and errors
import os  # Import the os module to interact with the operating system, specifically paths
from datetime import datetime  # Import datetime to work with dates and times

# Create a log file name based on the current date and time (format: month_day_year_hour_minute_second)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory path for logs in the current working directory
logs_dir = os.path.join(os.getcwd(), "logs")

# Create the directory if it doesn't exist (including any necessary parent directories)
os.makedirs(logs_dir, exist_ok=True)

# Complete path to the log file by combining logs directory and the log file name
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Specify the file to which logs will be written
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Set the format of the log messages
    level=logging.INFO,  # Set the logging level to INFO, so only messages at this level and above are logged
)

