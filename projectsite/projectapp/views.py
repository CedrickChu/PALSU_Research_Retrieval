from django.shortcuts import render, redirect
from django.contrib.auth import logout
from typing import List
import os
from gensim.models import KeyedVectors
import pymongo
from pinecone import ServerlessSpec, Pinecone
from django.http import Http404
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
import nltk
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize, sent_tokenize
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
from django import template
from bokeh.io import output_file, save
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, from_networkx
from bokeh.palettes import Spectral6
from bokeh.embed import components
from .forms import UpdateProfileForm, ResearchPaperForm
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from docx import Document
from django.conf import settings
import io
from PIL import Image
import pytesseract
import fitz 
from datetime import datetime
nltk.download('wordnet')
import gensim.models as gm
from gensim.models import Word2Vec

# configure client
api_key = os.environ.get('PINECONE_API_KEY')
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
pc = Pinecone(api_key=api_key)

index_name = "semanticmap"
index = pc.Index(index_name)

client = pymongo.MongoClient("{{mongo_client_id}}")
db = client.IRSRP
collection = db.paper


  
def get_doc_vector(model, tokens):
    valid_tokens = [word for word in tokens if word in model.wv]

    if not valid_tokens:
        return np.zeros(model.vector_size)

    query_vector = np.mean([model.wv[word] for word in valid_tokens], axis=0)
    return query_vector
  
load_model = gm.Word2Vec.load('../models/5model_cbow_400_5/word2vec_model.gensim')


def is_superuser(user):
    return user.is_superuser

def preprocess_text(text: str) -> str:
    """Cleans and preprocesses the input text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

def first(request):
    return render(request, "first.html")

def logout_view(request):
    logout(request)
    return redirect('/')


@login_required(login_url="/accounts/login/")
def dashboard(request):
    context = {'segment': 'dashboard'}
    html_template = loader.get_template('home/dashboard.html')
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/accounts/login/")
def pages(request):
    context = {}
    try:
        load_template = request.path.split('/')[-1]
        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
    
    
@login_required(login_url="/accounts/login/")
def search_view(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        qemb = preprocess_text(query)
        if not query:
            context = {'results': [], 'query': query, 'error': 'Query cannot be empty.'}
            return render(request, 'search_results.html', context)
        try:
            query_embedding = get_doc_vector(load_model, qemb)
        except Exception as e:
            context = {'results': [], 'query': query, 'error': f'Error generating embedding: {e}'}
            return render(request, 'search_results.html', context)
        try:
            results = index.query(
                vector=query_embedding,
                top_k=100,
                include_metadata=True,
            )
        except Exception as e:
            context = {'results': [], 'query': query, 'error': f'Error querying Pinecone: {e}'}
            return render(request, 'search_results.html', context)
        
        context = {'results': [], 'query': query}
        if 'matches' in results:
            for match in results['matches']:
                metadata = match.get('metadata', {})
                score = match.get('score', 0)
                if score > 0.2:
                    context['results'].append({
                        'thesis_name': metadata.get('title', 'No title'),
                        'filename': metadata.get('filename', '#'),
                        'year': metadata.get('year', 'Unknown'),
                        'authors': metadata.get('authors', 'Unknown'),
                        'faculty': metadata.get('faculty', 'Unknown'),
                        'score': score,
                    })
        return render(request, 'paper/search_results.html', context)

    # Default GET request behavior
    return render(request, 'index.html')

@login_required(login_url="/accounts/login/")
def navbar_search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        qemb = preprocess_text(query)
        if not query:
            context = {'results': [], 'query': query, 'error': 'Query cannot be empty.'}
            return render(request, 'search_results.html', context)
        try:
            query_embedding = get_doc_vector(load_model, qemb)
        except Exception as e:
            context = {'results': [], 'query': query, 'error': f'Error generating embedding: {e}'}
            return render(request, 'search_results.html', context)
        try:
            results = index.query(
                vector=query_embedding,
                top_k=100,
                include_metadata=True,
            )
        except Exception as e:
            context = {'results': [], 'query': query, 'error': f'Error querying Pinecone: {e}'}
            return render(request, 'search_results.html', context)
        
        context = {'results': [], 'query': query}
        if 'matches' in results:
            for match in results['matches']:
                metadata = match.get('metadata', {})
                score = match.get('score', 0)
                if score > 0.2:
                    context['results'].append({
                        'thesis_name': metadata.get('title', 'No title'),
                        'filename': metadata.get('filename', '#'),
                        'year': metadata.get('year', 'Unknown'),
                        'authors': metadata.get('authors', 'Unknown'),
                        'faculty': metadata.get('faculty', 'Unknown'),
                        'score': score,
                    })
        return render(request, 'paper/search_results.html', context)
    
    return render(request, 'includes/navigation.html')

@login_required(login_url="/accounts/login/")
def thesis_detail(request, thesis_id):
    try:
        result = index.fetch(ids=[thesis_id])
        if not result['vectors']:
            raise Http404("Thesis not found")
        thesis_metadata = result['vectors'][thesis_id]['metadata']
        context = {
            'thesis': thesis_metadata,
        }
        return render(request, 'paper/detail_view.html', context)
    except Exception as e:
        raise Http404(f"Error fetching thesis details: {str(e)}")
    
def profile_view(request):
    if request.user.is_authenticated:
        user = request.user 
        return render(request, 'home/profile.html', {'user': user})
    
@login_required(login_url="/accounts/login/")
def update_profile(request):
    user = request.user  
    if request.method == 'POST':
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        if User.objects.filter(email=email).exclude(id=user.id).exists():
            messages.error(request, "This email address is already in use.")
            return redirect('update_profile')

        # Update the user fields
        user.email = email
        user.first_name = first_name
        user.last_name = last_name
        user.save()  

        messages.success(request, "Your profile has been updated.")
        return redirect('profile') 

    return render(request, 'home/update_profile.html') 

@login_required(login_url="/accounts/login/")
def research_paper_list(request):
    """Retrieve and display a list of research papers stored in Pinecone."""
    # Query Pinecone for all the papers stored (adjust based on your needs)
    results = index.query(
        vector = [0] * 768,
        top_k=100,  
        include_metadata=True
    )
    papers = [
        {
            'id': match['id'],
            'authors': match['metadata'].get('authors', 'Unknown'), 
            'faculty': match['metadata'].get('faculty', 'Unknown'), 
            'title': match['metadata'].get('title', 'No Title'),
            'year': match['metadata'].get('year', 'Unknown'),
            'fuzzy_ratio': 100
        }
        for match in results.get('matches', [])
    ]

    context = {'papers': papers}
    return render(request, 'paper/research_paper_list.html', context)

@login_required(login_url="/accounts/login/")
@user_passes_test(is_superuser)
def edit_research_paper(request, paper_id):
    """View to edit an existing research paper stored in Pinecone."""
    # Retrieve the research paper's metadata from Pinecone
    result = index.fetch(ids=[paper_id])
    paper_data = result['vectors'][paper_id].get('metadata')
    if not paper_data:
        return HttpResponse("Research paper not found.", status=404)

    if request.method == 'POST':
        form = ResearchPaperForm(request.POST, initial=paper_data)
        if form.is_valid():
            updated_data = {
                'title': form.cleaned_data['title'],
                'authors': form.cleaned_data['authors'],
                'faculty': form.cleaned_data['faculty'],
                'year': form.cleaned_data['year'],
            }
            index.upsert([(paper_id, result['vectors'][paper_id]['values'], updated_data)])
            return redirect('research-paper-list')
    else:
        form = ResearchPaperForm(initial=paper_data)
    return render(request, 'paper/update_paper.html', {'form': form, 'paper_id': paper_id})


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF and Tesseract OCR for scanned PDFs.
    """
    text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Try to extract text directly from the PDF (text-based)
        page_text = page.get_text()
        
        if page_text.strip():
            text += page_text
        else:
            # If no text is found, the page is likely scanned; perform OCR
            
            # Convert the page to a PNG image
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # Perform OCR on the image using Tesseract
            ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')

            # If OCR detected text, append it
            if ocr_text.strip():
                text += ocr_text

    return text  # Indentation fix: return statement was inside else block

def extract_text_from_word(word_path: str) -> str:
    """Extract text from a Word document."""
    doc = Document(word_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text



@login_required(login_url="/accounts/login/")
@user_passes_test(is_superuser)
def upload_pdf_view(request):
    """Handles the PDF upload and text extraction."""
    if request.method == 'POST' and request.FILES.get('pdf'):
        uploaded_file = request.FILES['pdf']
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)
    
        text = ""
        if file_extension == "pdf":
            text = extract_text_from_pdf(file_path)
        elif file_extension in ["docx", "doc"]:
            text = extract_text_from_word(file_path)
        else:
            return HttpResponse("Unsupported file format.", status=400)

        embedding = get_doc_vector(load_model, text)

        # Extract the year from the form data
        year = request.POST.get('year', 'N/A')
        try:
            # If the year is a full date, extract just the year
            if len(year) > 4:
                year = str(datetime.strptime(year, "%Y-%m-%d").year)
            # If it's a year-only string, use it directly
            elif len(year) == 4:
                year = year
            else:
                year = 'N/A'  
        except ValueError:
            year = 'N/A'  

        authors = request.POST.get('authors', 'Unknown').strip()
        authors_list = [author.strip() for author in authors.split(',') if author.strip()]

        metadata = {
            'title': request.POST.get('title', 'Untitled'),
            'authors': authors_list,
            'filename': fs.url(filename),
            'year': year,
            'faculty': request.POST.get('faculty', 'N/A'),
        }
        try:
            index.upsert([{
                "id": filename,
                "values": embedding,
                "metadata": metadata
            }])
            print(f"Thesis uploaded successfully with ID: {filename}")
        except Exception as e:
            return HttpResponse(f"Error while uploading paper: {str(e)}", status=500)

        return redirect('thesis_detail', thesis_id=filename)

    return render(request, 'paper/upload_paper.html')