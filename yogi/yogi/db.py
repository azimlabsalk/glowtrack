from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from yogi import config


engine = create_engine(config.db_url)
Session = sessionmaker(bind=engine)

session = Session()
