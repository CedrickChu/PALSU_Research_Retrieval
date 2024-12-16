from django import forms
from django.contrib.auth.models import User

class UpdateProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exclude(id=self.instance.id).exists():
            raise forms.ValidationError("This email address is already in use.")
        return email

from django import forms

class QueryForm(forms.Form):
    query = forms.CharField(label='Enter your query', max_length=200, widget=forms.TextInput(attrs={'placeholder': 'Type your query here...'}))

class ResearchPaperForm(forms.Form):
    title = forms.CharField(
        label='Title', 
        max_length=255, 
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    authors = forms.CharField(
        label='Authors', 
        max_length=255, 
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    faculty = forms.CharField(
        label='Faculty', 
        max_length=100, 
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    year = forms.CharField(
        label='Year',
        required=False, 
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    filename = forms.CharField(
        label='File', 
        required=False, 
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )

    def clean_authors(self):
        authors = self.cleaned_data['authors']
        return [author.strip() for author in authors.split(',') if author.strip()]