# Generated by Django 3.1.7 on 2021-09-22 08:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('f_clustering', '0002_auto_20210911_2308'),
    ]

    operations = [
        migrations.AlterField(
            model_name='projects',
            name='name',
            field=models.CharField(max_length=200, unique=True, verbose_name='Название'),
        ),
    ]