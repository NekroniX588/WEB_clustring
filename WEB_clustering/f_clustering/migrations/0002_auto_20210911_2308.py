# Generated by Django 3.1.7 on 2021-09-11 20:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('f_clustering', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='projects',
            name='date',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
