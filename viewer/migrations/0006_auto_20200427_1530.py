# Generated by Django 3.0.2 on 2020-04-27 06:30

from django.db import migrations, models
import django.db.models.deletion
import django_mysql.models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0005_delete_segresultmodel'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imagemodel',
            name='results',
        ),
        migrations.CreateModel(
            name='SegResultModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('results', django_mysql.models.JSONField(default=dict, null=True)),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='seg_result', to='viewer.ImageModel')),
            ],
        ),
    ]
