from django import template
from typing import Any, Optional

register = template.Library()


@register.filter
def get_item(dictionary: dict, key: Any) -> Optional[Any]:
    if dictionary and key in dictionary:
        return dictionary.get(key)
    return None
