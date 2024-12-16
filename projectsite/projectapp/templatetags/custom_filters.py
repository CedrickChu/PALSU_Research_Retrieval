# custom_filters.py
from django import template
from datetime import datetime
from dateutil.parser import parse as parse_date

register = template.Library()

@register.filter(name='custom_date_format')
def custom_date_format(value):
    try:
        parsed_date = parse_date(value)
        return parsed_date.strftime('%d %B %Y')
    except (TypeError, ValueError):
        return ''
