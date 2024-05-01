from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from userapp.models import *
from adminapp.models import *
import urllib.request
import urllib.parse
import pandas as pd
from sklearn.ensemble  import AdaBoostClassifier
from sklearn.svm  import SVC
from django.core.paginator import Paginator
from xgboost import XGBClassifier
from userapp.models import UserHearingDetectionModels



# Create your views here.
def admin_index(request):
    messages.success(request,'login successfull')
    all_users_count =  UserHearingDetectionModels.objects.all().count()
    pending_users_count = UserHearingDetectionModels.objects.filter(user_status = 'pending').count()
    rejected_users_count = UserHearingDetectionModels.objects.filter(user_status = 'Rejected').count()
    accepted_users_count = UserHearingDetectionModels.objects.filter(user_status = 'Accepted').count()
    datasets_count = Upload_dataset_model.objects.all().count()
    no_of_predicts = Predict_details.objects.all().count()
    return render(request, 'admin/index.html',{'a' : pending_users_count, 'b' : all_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e' : datasets_count, 'f' : no_of_predicts})


def admin_pending(request):
    users = UserHearingDetectionModels.objects.filter(user_status ='pending')
    context = {'u':users}
    return render(request, "admin/pending.html", context)

def Admin_Reject_Btn(request, x):
        user = UserHearingDetectionModels.objects.get(user_id = x)
        user.user_status = 'Rejected'
        messages.success(request,'Status Changed successfull')

        user.save()
        messages.warning(request, 'Rejected') 
      
        return redirect('pending')

def Admin_accept_Btn(req, x):
        user = UserHearingDetectionModels.objects.get(user_id = x) 
        user.user_status = 'Accepted' 
        messages.success(req,'Status Changed successfull')
 
        user.save()
        messages.success(req, 'Accepted') 
        return redirect('pending')

def admin_manage(request):
    a = UserHearingDetectionModels.objects.all()
    paginator = Paginator(a, 5) 
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request, "admin/manage.html", {'all':post})

def admin_upload(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = str((file.size)/1024) +' kb'
        Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(request, 'Your dataset was uploaded..')
    return render(request, "admin/upload-data.html")

# Admin delete dataset button
def delete_dataset(request, id):
    dataset = Upload_dataset_model.objects.get(user_id = id).delete()
    messages.warning(request, 'Dataset was deleted..!')
    return redirect('view')

def admin_view(request):
    dataset = Upload_dataset_model.objects.all()
    paginator = Paginator(dataset, 5)
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request, "admin/view-data.html", {'data' : dataset, 'user' : post})

def view_view(request):
    # df=pd.read_csv('heart.csv')
    data = Upload_dataset_model.objects.last()
    print(data,type(data),'sssss')
    file = str(data.Dataset)
    df = pd.read_csv(f'./media/{file}')
    table = df.to_html(table_id='data_table')
    return render(request,'admin/view-view.html', {'t':table})

def admin_ada_algo(request):
    return render(request, "admin/ada-algo.html")


# ADABoost_btn
def ADABoost_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)

    X = df.drop('test_result', axis = 1)
    y = df['test_result']

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)

    from sklearn.ensemble import AdaBoostClassifier

    ADB = AdaBoostClassifier()
    ADB.fit(X_train, y_train)

    # prediction
    train_prediction= ADB.predict(X_train)
    test_prediction= ADB.predict(X_test)
    print('*'*20)
    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "ADA Boost Algorithm"
    ADA_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = ADA_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/ada-algo.html',{'i': data})

def admin_logistic_algo(request):
    return render(request, "admin/logistic-algo.html")

def logistic_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    X = df.drop('test_result', axis = 1)
    y = df['test_result']

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)

    from sklearn.linear_model import LogisticRegression
    ANN = LogisticRegression()
    ANN.fit(X_train, y_train)

    # prediction
    train_prediction= ANN.predict(X_train)
    test_prediction= ANN.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Logistic Regression Algorithm"
    Logistic.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = Logistic.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    req.session['accuracy']=accuracy
    return render(req, 'admin/logistic-algo.html',{'i': data})

def admin_decission_algo(request):
    return render(request, "admin/decission-algo.html")

def Decisiontree_btn(req):
    dataset = Upload_dataset_model.objects.last()
    df=pd.read_csv(dataset.Dataset.path)
   

    X = df.drop('test_result', axis = 1)
    y = df['test_result']


    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    
   #  XGBoost
    from sklearn.tree import DecisionTreeClassifier
    DEC = DecisionTreeClassifier()
    DEC.fit(X_train, y_train)

    # prediction
    train_prediction= DEC.predict(X_train)
    test_prediction= DEC.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    name = "Decision Tree Algorithm"
    DECISSION_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = DECISSION_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    req.session['des_accuracy']=accuracy

    return render(req, 'admin/decission-algo.html',{'i':data})

def admin_knn_algo(request):
    return render(request, "admin/knn-algo.html")

def KNN_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)

    X = df.drop('test_result', axis = 1)
    y = df['test_result']


    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test= scaler.transform(X_test)
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)

    # prediction
    train_prediction= KNN.predict(X_train)
    test_prediction= KNN.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "KNN Algorithm"
    KNN_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = KNN_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/knn-algo.html',{'i': data})

def admin_svm_algo(request):
    return render(request, "admin/svm-algo.html")

def SVM_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
 
    X = df.drop('test_result', axis = 1)
    y = df['test_result']


    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)


    SXM = SVC()
    SXM.fit(X_train, y_train)

    # prediction
    train_prediction= SXM.predict(X_train)
    test_prediction= SXM.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "SVM Algorithm"
    SXM_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = SXM_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/svm-algo.html',{'i':data})

def admin_RandomForest_algo(request):
    return render(request, "admin/RandomForest-algo.html")

def randomforest_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    
    X = df.drop('test_result', axis = 1)
    y = df['test_result']

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)
    from sklearn.ensemble import RandomForestClassifier
    RDF = RandomForestClassifier()
    RDF.fit(X_train, y_train)

    # prediction
    train_prediction= RDF.predict(X_train)
    test_prediction= RDF.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
    recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
    f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
    # Accuracy_train(accuracy_score(prediction_train,y_train))
    name = "Random Forest"
    RandomForest.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = RandomForest.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    req.session['ran_accuracy']=accuracy

    return render(req, 'admin/RandomForest-algo.html',{'i': data})

def admin_gradient_boosting_Classifier(request):
    return render (request, 'admin/gradient-boosting-Classifier.html')

def gradient_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    

    X = df.drop('test_result', axis = 1)
    y = df['test_result']


    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=22,test_size=0.2)
    from sklearn.ensemble import GradientBoostingClassifier
    GB = GradientBoostingClassifier()
    GB.fit(X_train, y_train)
    # prediction
    train_prediction= GB.predict(X_train)
    test_prediction= GB.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction)*100, 2)
    recall = round(recall_score(y_test,test_prediction)*100, 2)
    f1 = round(f1_score(y_test,test_prediction)*100, 2)
    name = "Gradient Boost Algorithm"
    GradientBoosting.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = GradientBoosting.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/gradient-boosting-Classifier.html',{'i': data})

# Admin XGBOOST Algorithm
def admin_xg_algo(req):
    return render(req, 'admin/xg-algo.html')

# XGBOOST_btn
def XGBOOST_btn(req):
    dataset = Upload_dataset_model.objects.last()
    # print(dataset.Dataset)
    df=pd.read_csv(dataset.Dataset.path)
    

    X = df.drop('test_result', axis = 1)
    y = df['test_result']


    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)
    XGB = XGBClassifier()
    XGB.fit(X_train, y_train)
    # prediction
    train_prediction= XGB.predict(X_train)
    test_prediction= XGB.predict(X_test)
    print('*'*20)

    # evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
    precession = round(precision_score(y_test,test_prediction)*100, 2)
    recall = round(recall_score(y_test,test_prediction)*100, 2)
    f1 = round(f1_score(y_test,test_prediction)*100, 2)
    name = "XG Boost Algorithm"
    XG_ALGO.objects.create(Accuracy=accuracy,Precession=precession,F1_Score=f1,Recall=recall,Name=name)
    data = XG_ALGO.objects.last()
    messages.success(req, 'Algorithm executed Successfully')
    return render(req, 'admin/xg-algo.html',{'i': data})

def admin_comparison_graph(request):
    accuracy = request.session.get('accuracy')
    ran_accuracy = request.session.get('ran_accuracy')
    des_accuracy = request.session.get('des_accuracy')


    details = XG_ALGO.objects.last()
    a = details.Accuracy
    deatails1 = ADA_ALGO.objects.last()
    b = deatails1.Accuracy
    details2 = KNN_ALGO.objects.last()
    c = details2.Accuracy
    deatails3 = SXM_ALGO.objects.last()
    d = deatails3.Accuracy
    details4 = des_accuracy
    # e = details4.Accuracy
    details6 = accuracy
    # g = details6.Accuracy
    details7 = ran_accuracy
    # h = details7.Accuracy
    print( details4, details6, details7,"kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
    details9 = GradientBoosting.objects.last()
    z = details9.Accuracy
    return render(request, 'admin/comparison-graph.html', {'xg':a,'ada':b,'knn':c,'sxm':d,'dt':details4,'log':details6, 'ran':details7, 'gst': z})

def Change_Status(req, id):
    # user_id = req.session['User_Id']
    user = UserHearingDetectionModels.objects.get(user_id = id)
    if user.user_status == 'Accepted':
        user.user_status = 'Rejected'   
        user.save()
        messages.success(req, 'Status Succefully Changed ') 
        return redirect('manage')
    else:
        user.user_status = 'Accepted'
        user.save()
        messages.success(req, 'Status Succefully Changed  ')
        return redirect('manage')
    
def Delete_User(req, id):
    UserHearingDetectionModels.objects.get(user_id = id).delete()
    messages.info(req, 'Deleted  ') 
    return redirect('manage')