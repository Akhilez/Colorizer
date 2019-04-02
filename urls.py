from django.conf.urls import url

from colorizer import views

urlpatterns = [
    url(r'^$', views.index, name="index"),
]
