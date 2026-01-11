import os
import logging
from datetime import datetime

log_file = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

logs_path = os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

logfile_path = os.path.join(logs_path,log_file)

logging.basicConfig(
    filename=logfile_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s %(message)s",
    level = logging.INFO 
)
