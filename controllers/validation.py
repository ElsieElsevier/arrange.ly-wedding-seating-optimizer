"""Shared helpers for rejecting bad request data with a 400 response."""
import math

from flask import request
from werkzeug.exceptions import BadRequest


def get_json_object():
    """Return the request body, which must be a JSON object."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest('Request body must be a JSON object sent with Content-Type: application/json.')
    return data


def get_required_list(data, key):
    """Return the list stored at data[key]."""
    value = data.get(key)
    if not isinstance(value, list):
        raise BadRequest(f"'{key}' is required and must be a list.")
    return value


def is_number(value):
    """Return True for finite JSON numbers. Python counts booleans as ints, so exclude them."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
