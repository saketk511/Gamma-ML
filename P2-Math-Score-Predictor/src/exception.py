import sys  # Import the sys module to access system-specific parameters and functions, exception in this case
from src.logger import logging  # Import a logging module for logging error details

def error_message_detail(error, error_detail: sys):
    '''
    This function captures and formats detailed error information, including:
    - The file name where the error occurred
    - The line number of the error
    - The error message itself
    '''
    # Extract exception traceback details (exc_tb) using sys.exc_info()
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format the error message with file name, line number, and the error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message  # Return the formatted error message
    

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        '''
        Initialize the CustomException class, which inherits from Python's built-in Exception class.
        - error_message: The original error message
        - error_detail: The sys module used to extract detailed error information
        '''
        super().__init__(error_message)  # Call the base class constructor with the error message
        # Store the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        '''
        When the exception is printed or logged, this method returns the detailed error message.
        '''
        return self.error_message  # Return the detailed error message as a string

    


        