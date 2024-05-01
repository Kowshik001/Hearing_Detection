"""
URL configuration for hearing_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from userapp import views as user_views
from adminapp import views as admin_view
# URLS
urlpatterns = [
    # User_Urls
    path('admin/', admin.site.urls),
    path('register/', user_views.user_register, name = 'register'),
    path('login/', user_views.user_login, name = 'login'),
    path('alogin/', user_views.user_admin, name = 'admin'),
    path('otp/', user_views.user_otp, name = 'otp'),
    path('', user_views.user_index, name = "index"),
    path('user/dashboard/', user_views.user_dashboard, name = 'dashboard'),
    path('about/', user_views.user_about, name = "about"),
    path('contact/', user_views.user_contact, name = "contact"),
    path('myprofile/', user_views.user_myprofile, name = 'myprofile'),
    path('hearing_loss_detection/', user_views.user_hearing_loss_detection, name = "detection"),
    path('result/', user_views.user_result, name ="result"),
    path('userlogout/', user_views.userlogout, name = 'userlogout'),


    #URLS_admin
    path('dashboard/', admin_view.admin_index, name = "admin_dashboard"),
    path('pending/', admin_view.admin_pending, name = "pending"),
    path('manage/', admin_view.admin_manage, name = 'manage'),
    path('upload/', admin_view.admin_upload, name = 'upload'),
    path('view/', admin_view.admin_view, name = "view"),
    path('ada/', admin_view.admin_ada_algo, name = "ada-algo"),
    path('logistic/', admin_view.admin_logistic_algo, name = "logistic-algo"),
    path('decission/', admin_view.admin_decission_algo, name = "decission-algo"),
    path('knn/', admin_view.admin_knn_algo, name = "knn-algo"),
    path('svc/', admin_view.admin_svm_algo, name = 'svm-algo'),
    path('RandomForest/', admin_view.admin_RandomForest_algo, name = 'RandomForest-algo'),
    path('Gradient/', admin_view.admin_gradient_boosting_Classifier, name='Gradient'),
    path('xg/', admin_view.admin_xg_algo, name = "xg-algo"),
    path('XGBOOST_btn', admin_view.XGBOOST_btn, name='XGBOOST_btn'),
    path('ADABoost_btn', admin_view.ADABoost_btn, name='ADABoost_btn'),
    path('KNN_btn', admin_view.KNN_btn, name='KNN_btn'),
    path('SVM_btn', admin_view.SVM_btn, name='SVM_btn'),
    path('Decisiontree_btn', admin_view.Decisiontree_btn, name='Decisiontree_btn'),
    path('logistic_btn', admin_view.logistic_btn, name='logistic_btn'),
    path('randomforest_btn', admin_view.randomforest_btn, name='randomforest_btn'),
    path('comparison/', admin_view.admin_comparison_graph, name = "comparison-graph"),
    path('adminrejectbtn/<int:x>', admin_view.Admin_Reject_Btn, name='adminreject'),
    path('adminacceptbtn/<int:x>', admin_view.Admin_accept_Btn, name='adminaccept'),
    path('GRADIENT_btn', admin_view.gradient_btn, name='gradient_btn'),
    path('admin-change-status/<int:id>',admin_view.Change_Status, name ='change_status'),
    path('admin-delete/<int:id>',admin_view.Delete_User, name ='delete_user'), 
    path('delete-dataset/<int:id>', admin_view.delete_dataset, name = 'delete_dataset'),
    path('view_view/', admin_view.view_view, name='view_view'),

]   + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
