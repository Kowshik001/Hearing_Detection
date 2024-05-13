from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from userapp.models import *
import urllib.request
import pandas as pd
import time
from adminapp.models import *
import urllib.parse
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# Create your views here.
def sendSMS(user, otp, mobile):
    data = urllib.parse.urlencode({
        'username': 'Codebook',
        'apikey': '56dbbdc9cea86b276f6c',
        'mobile': mobile,
        'message': f'Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you',
        'senderid': 'CODEBK'
    })
    data = data.encode('utf-8')
    # Disable SSL certificate verification
    # context = ssl._create_unverified_context()
    request = urllib.request.Request("https://smslogin.co/v3/api.php?")
    f = urllib.request.urlopen(request, data)
    return f.read()

def user_register(request):
    if request.method == 'POST':
        username = request.POST.get('user_name')
        email = request.POST.get('email_address')
        password = request.POST.get('email_password')
        number = request.POST.get('contact_number')
        file = request.FILES['user_file']
        print(request)
        print(username, email, password, number, file, 'data')
        otp = str(random.randint(1000, 9999))
        print(otp, 'generated otp')
        try:
            UserHearingDetectionModels.objects.get(user_email = email)
            messages.info(request, 'mail already registered')
            return redirect('register')
        except:
            mail_message = f'Registration Successfully\n Your 4 digit Pin is below\n {otp}'
            print(mail_message)
            send_mail("Student Password", mail_message , settings.EMAIL_HOST_USER, [email])
            # text message
            sendSMS(username, otp, number)
        
            UserHearingDetectionModels.objects.create(otp=otp, user_contact = number, user_username = username, user_email = email, user_password = password, user_file = file)
            request.session['user_email'] = email
            return redirect('otp')
    return render(request, 'user/register.html')

def user_login(request):
    """if request.method == 'POST':
        email = request.POST.get('email_address')
        password = request.POST.get('email_password')
        print(email, password)
        try:
            user = UserHearingDetectionModels.objects.get(user_email = email, user_password = password)
            print(user)
            request.session['user_id'] = user.user_id
            a = request.session['user_id']
            print(a)

            if user.user_password ==  password :
                if user.user_status == 'Accepted':
                    if user.otp_status == 'verified':
                        messages.success(request,'login successfull')
                        request.session['user_id'] = user.user_id
                        print('login sucessfull')
                        user.No_Of_Times_Login += 1
                        user.save()
                        return redirect('dashboard')
                    else:
                         return redirect('otp')
                elif user.user_password ==  password and user.user_status == 'Rejected':
                    messages.warning(request,"you account is rejected")
                else:
                    messages.info(request,"your account is in pending")
            else:
                 messages.error(request,'Login credentials was incorrect...')
        except:
            print(';invalid credentials')
            print('exce')
            return redirect('login')"""
    us='user@gmail.com'
    ps='user'
    if request.method == 'POST':
        email = request.POST.get('email_address')
        password = request.POST.get('email_password')
        print(email, password)
        if us == email and ps == password:
            return redirect('dashboard')
        else:
            return redirect('login')
    return render(request, "user/user.html")

def user_admin(request):
    admin_name = 'admin@gmail.com'
    admin_password = 'admin'
    if request.method == 'POST':
        adminemail = request.POST.get('emailaddress')
        adminpassword = request.POST.get('emailpassword')
        if admin_name == adminemail and admin_password == adminpassword:
            return redirect('admin_dashboard')
        else:
            return redirect('admin')
    return render(request, "user/admin.html")

def user_otp(request):
    user_id = request.session['user_email']
    user =UserHearingDetectionModels.objects.get(user_email = user_id)
    messages.success(request, 'OTP  Sent successfully')
    print(user_id)
    print(user, 'user avilable')
    print(type(user.otp))
    print(user. otp, 'creaetd otp')   
    if request.method == 'POST':
        u_otp = request.POST.get('otp')
        u_otp = int(u_otp)
        print(u_otp, 'enter otp')
        if u_otp == user.otp:
            print('if')
            user.otp_status  = 'verified'
            user.save()
            messages.success(request, 'OTP  verified successfully')
            return redirect('login')
        else:
            print('else')
            messages.error(request, 'Invalid OTP  ') 
            return redirect('otp')
    return render(request, 'user/Otp.html')

def user_index(request):
    return render(request, 'user/index.html')

def user_about(request):
    return render(request, "user/about.html")

def user_contact(request):
    return render(request, "user/contact.html")

def user_dashboard(request):
    prediction_count =  UserHearingDetectionModels.objects.all().count()
    user_id = request.session["user_id"]
    user = UserHearingDetectionModels.objects.get(user_id = user_id)
    return render(request, "user/dashboard.html", {'predictions' : prediction_count, 'la' : user})

def user_myprofile(request):
    views_id = request.session['user_id']
    user = UserHearingDetectionModels.objects.get(user_id = views_id)
    print(user, 'user_id')
    if request.method =='POST':
        username = request.POST.get('user_name')
        email = request.POST.get('email_address')
        number = request.POST.get('contact_number')
        password = request.POST.get('email_password')
        age = request.POST.get('Age_int')
        date = request.POST.get('date')
        print(username, email, number, password, date, age, 'data') 

        user.user_username = username
        user.user_email = email
        user.user_contact = number
        user.user_password = password
        user.user_dates = date 

        if len(request.FILES)!=0:
            file = request.FILES['user_file']
            user.user_file = file
            user.user_username = username
            user.user_email = email
            user.user_contact = number
            user.user_password = password
            user.save()
        else:
            user.user_username = username
            user.user_email = email
            user.user_contact = number
            user.user_password = password
            user.save()


    return render(request, "user/myprofile.html", {'i':user})

def user_hearing_loss_detection(req):
    if req.method == 'POST':
        age = req.POST.get('field1')
        sex = req.POST.get('sex')
        db = req.POST.get('field3')
        PhysicalScore = req.POST.get('field2')
        
        print(age, sex, PhysicalScore,db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        age = int(age)
        
        if sex == "male":
            gender = 0
        else:
            gender = 1
        context = {'gender': gender}
        
        PhysicalScore = float(PhysicalScore)
        db = int(db)


            
        # print(type(age),x)
        # DATASET.objects.create(Age = age, Glucose = sex, BloodPressure = plasma_CA19_9, SkinThickness = creatinine, Insulin = lyve1, BMI = regb1, DiabetesPedigreeFunction = tff1)
        import pickle
        file_path = 'hearing_test\gbc_hearing.pkl'  # Path to the saved model file

        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
            res =loaded_model.predict([[age,PhysicalScore,db,gender]])
            print(res,"resssssssssssssssssssresssssssssssssss")
            dataset = Upload_dataset_model.objects.last()
            # print(dataset.Dataset)
            df=pd.read_csv(dataset.Dataset.path)
            X = df.drop('test_result', axis = 1)
            y = df['test_result']


            X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.2)


            from sklearn.ensemble import GradientBoostingClassifier
            GB = GradientBoostingClassifier()
            GB.fit(X_train, y_train)

            # prediction
            train_prediction= GB.predict(X_train)
            test_prediction= GB.predict(X_test)
            print('*'*20)

            # evaluation
            accuracy = round(accuracy_score(y_test,test_prediction)*100, 2)
            precession = round(precision_score(y_test,test_prediction,average = 'macro')*100, 2)
            recall = round(recall_score(y_test,test_prediction,average = 'macro')*100, 2)
            f1 = round(f1_score(y_test,test_prediction,average = 'macro')*100, 2)
            req.session['acr'] = accuracy
            req.session['pre'] = precession
            req.session['rec'] = recall
            req.session['f'] = f1
            print(precession, accuracy,recall, f1,'uuuuuuuuuuuuuuuuuuuuuuuuuuu')
            x = 0
            if res == 0:
                x = 0
                messages.success(req,"Hearing Loss Is Not Detected")
            else:
                x=1
                messages.warning(req,"Hearing Is Detected")
            print(x, )
            context2 = {'acr': accuracy,'pre': precession,'f':f1,'rec':recall,'res':x}
            req.session['res'] = x

            # print(type(res), 'ttttttttttttttttttttttttt', context)
            print(res)
           
        return redirect("result")
    return render(req, "user/hearing_loss_detection.html")

def userlogout(request):
    view_id = request.session["user_id"]
    user = UserHearingDetectionModels.objects.get(user_id = view_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(request, 'You are logged out..')
    # print(user.Last_Login_Time)
    # print(user.Last_Login_Date)
    return redirect('login')


def user_result(request):
    accuracy = request.session.get('acr')
    precession = request.session.get('pre')
    recall = request.session.get('rec')
    f1 = request.session.get('f')
    x = request.session.get('res')


    return render(request, "user/result.html",{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'res':x})
