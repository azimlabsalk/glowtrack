from functools import wraps


def readlines(filepath):
    with open(filepath, 'r') as f:
        return f.read().splitlines()


def equals_one(value, eps=0.001):
    return abs(1.0 - value) < eps


def contains_duplicates(lst):
    n_unique = len(set(lst))
    return n_unique != len(lst)


def handle(exception, handler):
    """Exception handling decorator."""
    def wrapper(func):
        @wraps(func)
        def newfunc(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except exception as e:
                handler(e)
        return newfunc
    return wrapper


def img_to_blob(img):
    from io import BytesIO
    from PIL import Image

    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return buff.getvalue()
