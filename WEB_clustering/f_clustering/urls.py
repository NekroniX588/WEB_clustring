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
    path('delete_project/<int:pk>', views.delete_project, name='delete_project'),

    path('start_page/<int:pk>', views.project_start, name='start_project'),
    path('statistic/<int:pk>', views.statistic, name='statistic'),
    path('show_data/<int:pk>', views.table, name='table'),

    path('const_page/<int:pk>', views.const_start, name='const_start'),
    path('const_reload/<int:pk>', views.const_reload, name='const_reload'),
    path('calculate_a/<int:pk>/<int:type_optimization>', views.calculate_a, name='calculate_a'),
    
    path('clustering_page/<int:pk>', views.clustering_start, name='clustering_start'),
]
