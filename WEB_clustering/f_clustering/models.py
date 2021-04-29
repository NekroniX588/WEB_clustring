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



# class Consts(models.Model):
# 	percent_for_norms = models.FloatField()
# 	percent_of_zeros = models.FloatField()
# 	round_const = models.FloatField()
# 	const = models.FloatField()
# 	percent_X = models.FloatField()
# 	down_steps = models.FloatField()
# 	up_steps = models.FloatField()
# 	max_depth = models.FloatField()
# 	power_koef = models.FloatField()
# 	percent_Y = models.FloatField()
# 	threshold = models.FloatField()
# 	Y_step = models.FloatField()

# class Conturs(models.Model):
# 	min_points = models.FloatField()
# 	contour_points = models.FloatField()
# 	min_diff = models.FloatField()
# 	min_diff_1 = models.FloatField()
# 	min_diff_2 = models.FloatField()
# 	num_of_lenghts = models.FloatField()

# class IsolatedCluster(models.Model):
# 	constU1 = models.FloatField()
# 	min_len = models.FloatField()
# 	min_dif_1  = models.FloatField()
# 	min_dif_2  = models.FloatField()
# 	divider = models.FloatField()
# 	max_div_num = models.FloatField()
# 	merge_threshold = models.FloatField()