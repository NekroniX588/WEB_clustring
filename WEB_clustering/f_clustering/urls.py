from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('login', views.SpeechLoginView.as_view(), name='login'),
    path('register', views.SpeechRegisterView.as_view(), name='register'),
    path('logout', views.SpeechLogout.as_view(), name='logout'),
    path('projects', views.SpeechProjectsView.as_view(), name='projects'),
    path('add_projects', views.SpeechProjectsCreate.as_view(), name='add_projects'),
    path('start_page/<int:pk>', views.project_start, name='start_project'),
    path('delete_project/<int:pk>', views.delete_project, name='delete_project')
]
