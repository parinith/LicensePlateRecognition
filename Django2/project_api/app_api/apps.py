from django.apps import AppConfig
from django.conf import settings
from .mlmodels import fp1
import os

class AppApiConfig(AppConfig):
    name = 'app_api'
    test = fp1
