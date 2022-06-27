from celery import Celery

from yogi.config import celery_broker_url


app = Celery('yogi.celery.tasks',
             broker=celery_broker_url,
             include=['yogi.celery.tasks'])


if __name__ == '__main__':
    app.start()

