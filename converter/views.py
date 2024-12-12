from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os

def home(request):
    return render(request, 'converter/home.html')

from django.core.files.storage import FileSystemStorage

def encode(request):
    if request.method == 'POST':
        # Kullanıcıdan alınan resim
        uploaded_image = request.FILES['image']
        
        # Resmi kaydet
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_image_path = fs.path(filename)

        # Resmi 512x512 boyutuna yeniden boyutlandır
        resized_image_path = fs.path(f"resized_{uploaded_image.name}")
        with Image.open(uploaded_image_path) as img:
            img = img.convert('RGB')  # Her ihtimale karşı RGB formatına dönüştür
            img = img.resize((512, 512))  # Boyutlandırma
            img.save(resized_image_path)

        # Resized resmin URL'sini al
        resized_image_url = fs.url(f"resized_{uploaded_image.name}")
        
        # Kullanıcıdan alınan parametreler
        param1 = request.POST.get('param1')
        param2 = request.POST.get('param2')
        param3 = request.POST.get('param3')
        param4 = request.POST.get('param4')
        param5 = request.POST.get('param5')
        param6 = request.POST.get('param6')
        
        # İşlenmiş resim ve parametreleri frontend'e gönder
        return render(request, 'converter/encode.html', {
            'result_image': resized_image_url,  # 512x512 boyutundaki resim
            'param1': param1,
            'param2': param2,
            'param3': param3,
            'param4': param4,
            'param5': param5,
            'param6': param6,
        })
    
    return render(request, 'converter/encode.html')



def decode(request):
    if request.method == 'POST':
        # Kullanıcıdan alınan resim
        uploaded_image = request.FILES['image']
        
        # Kullanıcıdan alınan parametreler
        param1 = request.POST.get('param1')
        param2 = request.POST.get('param2')
        param3 = request.POST.get('param3')
        param4 = request.POST.get('param4')
        param5 = request.POST.get('param5')
        param6 = request.POST.get('param6')
        
        # Şimdilik, sadece yüklenen resmi işlenmiş resim olarak geri gönderiyoruz
        # İşlenmiş resim oluşturma (ileride burayı doldurabilirsiniz):
        # processed_image_path = process_image(uploaded_image, param1, param2, param3, param4, param5, param6)

        # Yüklenen resmi frontend'e geri gönderiyoruz
        return render(request, 'converter/decode.html', {
            'result_image': uploaded_image,
            'param1': param1,
            'param2': param2,
            'param3': param3,
            'param4': param4,
            'param5': param5,
            'param6': param6,
        })
    return render(request, 'converter/code.html')
