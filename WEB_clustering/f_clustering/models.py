from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
# Create your models here.

class Projects(models.Model):
	author = models.ForeignKey(User, on_delete = models.CASCADE, verbose_name='Автор', blank=True, null=True)
	date = models.DateTimeField(auto_now=True)
	name = models.CharField(max_length=200, verbose_name='Название')
	stage = models.IntegerField(default=0)
	status = models.BooleanField(verbose_name='Готово', default=False)
	attach = models.FileField(storage=FileSystemStorage(location='df'))
