from rest_framework import serializers
from viewer.models import *

class ImageSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = ImageModel
        fields = ('image', 'token', 'uploaded_date', 'updated_date', 'results')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'results')
