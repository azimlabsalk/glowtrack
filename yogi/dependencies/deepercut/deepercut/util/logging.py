import logging


is_setup = False

def setup_logging():
    global is_setup
    if not is_setup:
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename='log.txt', filemode='w',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO, format=FORMAT)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
        is_setup = True

