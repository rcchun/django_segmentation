# Generated by Django 3.0.2 on 2020-04-28 04:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0010_auto_20200428_0946'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='segresultmodel',
            name='results',
        ),
        migrations.AlterField(
            model_name='segresultmodel',
            name='seg_area',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='segresultmodel',
            name='seg_mask',
            field=models.IntegerField(default=0),
        ),
    ]
