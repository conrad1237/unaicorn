from django.shortcuts import render
from . import fake_model
from . import ml_predict

def home(request):
    data={}
    data['transacoes'] = ['t1','t2','t3']
    return render(request, 'index.html',data) #este data é apenas um protocolo para nossa página funcionar

def result(request):
    pclass = int(request.GET["pclass"])
    sex = int(request.GET["sex"])
    age = int(request.GET["age"])
    sibsp = int(request.GET["sibsp"])
    parch = int(request.GET["parch"])
    fare = int(request.GET["fare"])
    embarked = int(request.GET["embarked"])
    title = int(request.GET["title"])

    prediction = ml_predict.prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,title)
    return render(request, 'result.html',{'prediction':prediction})