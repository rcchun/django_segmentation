from django import template

register = template.Library()

@register.inclusion_tag('nav_bar.html', takes_context=True)
def nav_bar(context) :
    return

@register.filter
def get_at_index(object_list, index):
    return object_list[index]