from __future__ import unicode_literals

from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseForbidden
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageGTUploadForm, ImageUploadForm
from .models import *
import json, os, base64
from ast import literal_eval


@csrf_exempt
def upload(request):
    if request.method == "POST" :
        form = ImageGTUploadForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()

            return redirect('/imagelist/')
        else :
            return redirect('/imagelist/')
    else :
        template_name = "upload.html"
        return render(request, template_name)

@csrf_exempt
def upload_without_gt(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('/imagelist/')
        else:
            return redirect('/imagelist/')

def image_list(request):
    image_models = ImageModel.objects.all().order_by('-pk')

    segmentation_area = []
    severity_threshold = []

    for image_model in image_models:
        segmentation_area.append(image_model.segmentation_area)
        severity_threshold.append(image_model.severity_threshold)

    return render(request, 'imagelist.html', {
        'images' : image_models,
        'segmentation_area': segmentation_area,
        'severity_thresholds': severity_threshold,
    })


def image_detail(request, image_pk) :
    image = ImageModel.objects.filter(pk=image_pk)
    results = image[0].results
    results_hed = image[0].results_hed

    return render(request, 'imagedetail.html', {
                    'image': image[0],
                    'image_pk': image_pk,
                    'results' : results,
                    'results_hed':results_hed
                })
