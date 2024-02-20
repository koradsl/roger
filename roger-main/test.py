# import logging
#
# from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryCallState, before_sleep_log
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
#
# def callback(retry_state: RetryCallState):
#     print(type(retry_state.outcome._exception))
#
#     print(retry_state)
#
#
# def test_before():
#     print("hello before")
#
#
# @retry(
#     wait=wait_random_exponential(multiplier=1, max=5),
#     stop=stop_after_attempt(2),
#     retry_error_callback=callback,
#     before_sleep=before_sleep_log(logger, log_level=logging.DEBUG),
# )
# def zero_division():
#     1 / 0
#     return True
#
#
# try:
#     print(zero_division())
# except Exception as e:
#     print(e)

import logging

from tenacity import retry, wait_fixed, stop_after_attempt, before_sleep_log

# Configure logging
logging.basicConfig(level=logging.INFO)


# Function to be executed
@retry(
    wait=wait_fixed(1), stop=stop_after_attempt(3), before_sleep=before_sleep_log(logging.getLogger(), logging.INFO)
)
def my_function():
    5 / 0
    # Your code here
    pass


my_function()
