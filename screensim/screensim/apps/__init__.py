"""Importing this package registers every app with the core registry."""
from . import (bank_app, mail_app, messages_app, phone_app, photos_app,
               settings_app, shop_app)  # noqa: F401
