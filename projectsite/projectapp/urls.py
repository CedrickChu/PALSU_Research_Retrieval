from django.urls import path
from . import views

urlpatterns = [
    path("", views.first),
    path("dashboard/", views.dashboard, name='dashboard'),
    path("index/", views.search_view, name='search'),
    path("search/", views.navbar_search, name='navbar-search'),
    path("logout", views.logout_view, name='logout'),
    path('thesis/<str:thesis_id>/', views.thesis_detail, name='thesis_detail'),
    path('profile/', views.profile_view, name='profile'),
    path('update_profile/', views.update_profile, name='update_profile'),
    path('research-papers/', views.research_paper_list, name='research-paper-list'),
    path('research-papers/edit/<str:paper_id>/', views.edit_research_paper, name='edit_research_paper'),
    path('upload/', views.upload_pdf_view, name='upload_paper'),
]
