from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('block_ru/<int:pk>', views.block_ru, name='block_ru'),
    path('block_en/<int:pk>', views.block_en, name='block_en'),
    path('login', views.SpeechLoginView.as_view(), name='login'),
    path('register', views.SpeechRegisterView.as_view(), name='register'),
    path('logout', views.SpeechLogout.as_view(), name='logout'),

    path('projects', views.SpeechProjectsView.as_view(), name='projects'),
    path('add_projects', views.SpeechProjectsCreate.as_view(), name='add_projects'),
    path('delete_project/<int:pk>', views.delete_project, name='delete_project'),

    path('start_page/<int:pk>', views.project_start, name='start_project'),
    path('statistic/<int:pk>', views.statistic, name='statistic'),
    path('show_data/<int:pk>', views.table, name='table'),
    path('clear_log/<int:pk>', views.clear_log, name='clear_log'),
    path('split_data/<int:pk>', views.split_data, name='split_data'),

    path('const_page/<int:pk>', views.const_start, name='const_start'),
    path('const_reload/<int:pk>', views.const_reload, name='const_reload'),
    path('calculate_norms/<int:pk>', views.calculate_norms, name='calculate_norms'),
    path('calculate_pca_norms/<int:pk>', views.calculate_pca_norms, name='calculate_pca_norms'),
    path('calculate_a/<int:pk>/<int:type_optimization>', views.calculate_a, name='calculate_a'),
    path('distance_statistic/<int:pk>', views.distance_statistic, name='distance_statistic'),
    path('get_profile/<int:pk>', views.get_profile, name='get_profile'),
    
    path('f_statistic/<int:pk>', views.f_statistic, name='f_statistic'),
    path('clustering_page/<int:pk>', views.clustering_start, name='clustering_start'),
    path('compute_clustering/<int:pk>/<int:type_c>', views.compute_clustering, name='compute_clustering'),
    path('del_clustering/<int:pk>/<int:type_c>', views.del_clustering, name='del_clustering'),

    path('classification_page/<int:pk>', views.classification_start, name='classification_start'),
    path('classification/<int:pk>', views.classification, name='classification'),

    path('download_data/<int:pk>', views.download_data, name='download_data'),
    path('download_settings/<int:pk>', views.download_settings, name='download_settings'),
    path('download_clusters/<int:pk>', views.download_clusters, name='download_clusters'),
    path('download_subclusters/<int:pk>', views.download_subclusters, name='download_subclusters'),
]
