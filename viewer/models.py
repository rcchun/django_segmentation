from django.db import models

# Create your models here.
from rest_framework import exceptions
from viewer.utils import filename
from django_mysql.models import JSONField
from modules.seg.main import Segmentation
from modules.seg.main_hed import Segmentation_HED
import ast
import json
from viewer.tasks import NumpyEncoder

# Create your models here.

def image_path(instance, filename):
    return 'image/u/' + instance.user.username + '/' + randstr(4) + '.' +filename.split('.')[-1]

class ImageModel(models.Model):
    image = models.ImageField(blank=True, upload_to=filename.default, null=True)
    token = models.AutoField(primary_key=True)
    segmentation_area = models.IntegerField(default=200)
    severity_threshold = models.FloatField(default=0.5)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    results = JSONField(null=True)
    results_hed = JSONField(null=True)

    def save(self, *args, **kwargs):
        super(ImageModel, self).save(*args, **kwargs)
        analyzer = Segmentation()
        analyzer_hed = Segmentation_HED()
        task_get = analyzer.inference_by_path(self.image.path, self.token)
        task_get_hed = analyzer_hed.inference_by_path(self.image.path, self.token)

        self.results = str(task_get)
        self.results_hed = str(task_get_hed)
        super(ImageModel, self).save()

