from django.conf.urls import url
from segmentation import settings
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from viewer import views

urlpatterns = [
    url(r'^upload/', views.upload, name='upload'),
    url(r'^upload_without_gt/', views.upload_without_gt, name='upload_without_gt'),

    url(r'^imagelist/', views.image_list, name='imagelist'),
    url(r'^imagedetail/(?P<image_pk>\d+)/$', views.image_detail, name='imagedetail'),

]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)