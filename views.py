from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from colorizer.libs import test
import os
from django.conf import settings
from .forms import ImageForm
from colorizer.tf.check_model import check_model


def index(request):
    context = {}
    check_model()
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            image.name = 'gray.jpg'
            fullname = os.path.join(settings.BASE_DIR, 'colorizer/static/colorizer/' + image.name)
            if os.path.exists(fullname):
                os.remove(fullname)
            fs = FileSystemStorage()
            fs.save(fullname, image)
            test.colorize()
            context['success'] = 'success'
    else:
        form = ImageForm()

    context['form'] = form
    return render(request, 'colorizer/index.html', context)
