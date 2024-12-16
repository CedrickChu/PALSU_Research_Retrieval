from allauth.account.adapter import DefaultAccountAdapter
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.exceptions import ImmediateHttpResponse
from django.http import HttpResponse

class CustomAccountAdapter(DefaultAccountAdapter):
    def clean_email(self, email):
        """Validate that the email ends with @psu.palawan.edu.ph."""
        email = super().clean_email(email)
        domain = "@psu.palawan.edu.ph"
        if not email.endswith(domain):
            raise ValidationError(
                _("Make sure to use your PSU Organization Account"),
                params={"domain": domain},
            )
        return email


class MySocialAccountAdapter(DefaultSocialAccountAdapter):
    def pre_social_login(self, request, sociallogin):
        if sociallogin.account.provider == 'google':
            email_verified = sociallogin.account.extra_data.get('email_verified', False)
            if not email_verified:
                raise ValidationError("Google account email is not verified.")