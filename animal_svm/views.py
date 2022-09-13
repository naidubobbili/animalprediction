from django.shortcuts import render
from django.http import HttpResponse
from django_user_agents.utils import get_user_agent
import pandas as pd
from sklearn import svm



df=pd.read_csv('animal_svm/dataset/ownSVMAnimaldataSet2.csv')
X,y = df.iloc[:,0:-1].to_numpy(),df.iloc[:,-1].to_numpy()
svc = svm.SVC(kernel='linear', C=1.0).fit(X, y)



# Create your views here.
def home(request):
    user=get_user_agent(request)
    if user.is_mobile:
        return render(request,'animal_svm/mobile_home.html')
    else:
        return render(request,'animal_svm/home.html')


def animalresult(request):
    weight=int(request.GET.get('Weight'))
    height=int(request.GET.get('Height'))
    result=svc.predict([[weight,height]])[0]
    
    user=get_user_agent(request)
    if user.is_mobile:
        return render(request,'animal_svm/mobile_animalresult.html',{'svmresult':result})
    else:
        return render(request,'animal_svm/animalresult.html',{'svmresult':result})
