# Generated by Django 3.0.2 on 2020-04-27 07:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0007_imagemodel_results'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imagemodel',
            name='results',
        ),
    ]