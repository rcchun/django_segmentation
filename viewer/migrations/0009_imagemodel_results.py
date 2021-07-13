# Generated by Django 3.0.2 on 2020-04-27 08:29

from django.db import migrations
import django_mysql.models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0008_remove_imagemodel_results'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='results',
            field=django_mysql.models.JSONField(default=dict, null=True),
        ),
    ]
