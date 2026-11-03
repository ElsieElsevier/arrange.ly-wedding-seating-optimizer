"""Arrangely: wedding seating from guest descriptions."""
import math
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, HTTPException

from embeddings import EmbeddingUnavailable, embed_texts
from seating import SeatingError, plan_seating

MAX_GUESTS = 500
MAX_NAME_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 2000
MAX_SEATS_PER_TABLE = 50
REQUEST_FIELDS = {'guests', 'seats_per_table', 'num_tables'}
GUEST_FIELDS = {'name', 'description', 'partner', 'avoid'}


def _whole_number(value, label, largest):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= largest:
        raise BadRequest(f"{label} must be a whole number from 1 to {largest}.")
    return value


def _text(value, label, largest):
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"{label} is required and must be text.")
    value = value.strip()
    if len(value) > largest:
        raise BadRequest(f"{label} must be {largest} characters or fewer.")
    return value


def _parse_guests(raw_guests):
    if not isinstance(raw_guests, list) or not 2 <= len(raw_guests) <= MAX_GUESTS:
        raise BadRequest(f"Add at least 2 guests (up to {MAX_GUESTS}).")

    guests = []
    for position, raw in enumerate(raw_guests, start=1):
        if not isinstance(raw, dict):
            raise BadRequest(f"Guest {position} must be an object.")
        unknown = sorted(set(raw) - GUEST_FIELDS)
        if unknown:
            raise BadRequest(f"Guest {position} has unknown field(s): {', '.join(unknown)}.")

        partner = raw.get('partner') or None
        avoid = raw.get('avoid') or []
        if not isinstance(avoid, list):
            raise BadRequest(f"Guest {position}'s 'avoid' must be a list of names.")
        guests.append({
            'name': _text(raw.get('name'), f"Guest {position}'s name", MAX_NAME_LENGTH),
            'description': _text(raw.get('description'), f"Guest {position}'s description", MAX_DESCRIPTION_LENGTH),
            'partner': _text(partner, f"Guest {position}'s partner", MAX_NAME_LENGTH) if partner else None,
            'avoid': [_text(name, f"Guest {position}'s avoid name", MAX_NAME_LENGTH) for name in avoid],
        })

    _check_names(guests)
    return guests


def _check_names(guests):
    names = {guest['name'] for guest in guests}
    if len({guest['name'].casefold() for guest in guests}) < len(guests):
        raise BadRequest('Every guest needs a different name.')

    for guest in guests:
        for other in filter(None, [guest['partner'], *guest['avoid']]):
            if other not in names:
                raise BadRequest(f"{guest['name']} refers to '{other}', who isn't on the guest list.")
            if other == guest['name']:
                raise BadRequest(f"{guest['name']} can't be linked to themselves.")


def _parse_request(body):
    if not isinstance(body, dict):
        raise BadRequest('Request body must be a JSON object sent with Content-Type: application/json.')
    unknown = sorted(set(body) - REQUEST_FIELDS)
    if unknown:
        raise BadRequest(f"Unknown field(s): {', '.join(unknown)}.")

    guests = _parse_guests(body.get('guests'))
    seats = _whole_number(body.get('seats_per_table'), "Seats per table", MAX_SEATS_PER_TABLE)

    tables = body.get('num_tables')
    if tables is None:
        tables = math.ceil(len(guests) / seats)
    else:
        _whole_number(tables, "Number of tables", len(guests))

    if tables * seats < len(guests):
        raise BadRequest(f"Not enough seats: {len(guests)} guests but only {tables * seats} seats.")
    return guests, seats, tables


def get_run_options(env=os.environ):
    """Local, non-debug defaults. Anything more exposed must be requested explicitly."""
    return {
        'host': env.get('HOST', '127.0.0.1'),
        'port': int(env.get('PORT', 5000)),
        'debug': env.get('FLASK_DEBUG') == '1',
    }


def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

    @app.get('/')
    def index():
        return app.send_static_file('index.html')

    @app.post('/api/seating')
    def create_seating():
        guests, seats, tables = _parse_request(request.get_json(silent=True))
        embeddings = embed_texts([guest['description'] for guest in guests])
        return jsonify(plan_seating(guests, seats, tables, embeddings))

    @app.errorhandler(SeatingError)
    def handle_seating_error(error):
        return jsonify({'message': str(error)}), 400

    @app.errorhandler(EmbeddingUnavailable)
    def handle_embedding_error(error):
        app.logger.error('Language model unavailable', exc_info=error)
        message = 'The language model could not be loaded. The first run needs internet access to download it.'
        return jsonify({'message': message}), 503

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        return jsonify({'message': error.description}), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        # Keep the details in the server log rather than sending them to the client.
        app.logger.error('Unhandled error', exc_info=error)
        return jsonify({'message': 'Internal server error'}), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(**get_run_options())
