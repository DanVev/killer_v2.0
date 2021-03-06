from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^profile/$', views.profile, name='profile'),
    # url(r'^login/$', views.login_view, name='login'),
    url(r'^$', views.login_view, name='login'),
    url(r'^logout/$', views.logout_view, name='logout'),
    url(r'^rules/$', views.rules, name='rules'),
    url(r'^nominations/$', views.nominations, name='nominations'),
    url(r'^statistics/$', views.statistics, name='statistics'),
    url(r'^registration/$', views.registration, name='registration'),
    url(r'^profile/createGame/$', views.createGame, name='createGame'),
    url(r'^manage/(\w+)/$', views.manageGame, name='manageGame'),
    url(r'^startRegistration/$', views.startRegistration, name='startRegistration'),
    url(r'^registerOnGame/(\w+)/$', views.registerOnGame, name='registerOnGame'),
]
