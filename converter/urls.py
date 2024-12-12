from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('encode/', views.encode, name='encode'),
    path('decode/', views.decode, name='decode'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
