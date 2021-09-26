from django.contrib import admin
from .models import Projects

class ProjectsAdmin(admin.ModelAdmin):
    pass

admin.site.register(Projects, ProjectsAdmin)

# Register your models here.
