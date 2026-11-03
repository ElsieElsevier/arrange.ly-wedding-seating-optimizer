from flask import Blueprint, jsonify
from werkzeug.exceptions import BadRequest
import numpy as np

from controllers.validation import get_json_object, get_required_list

# Create a Blueprint for vectorization routes
vectorization_routes = Blueprint('vectorization', __name__, url_prefix='/api')

@vectorization_routes.route('/vectorize', methods=['POST'])
def vectorize_text():
    """Convert text inputs to vector embeddings."""
    texts = get_required_list(get_json_object(), 'texts')
    if not all(isinstance(text, str) for text in texts):
        raise BadRequest("Every item in 'texts' must be a string.")

    # This is a placeholder. In a real implementation, you would use a
    # language model or embedding service to convert texts to vectors.
    # For demo purposes, we'll just return random vectors
    embeddings = [np.random.rand(128).tolist() for _ in texts]

    return jsonify({
        'success': True,
        'embeddings': embeddings
    })


@vectorization_routes.route('/analyze_survey', methods=['POST'])
def analyze_survey():
    """Process survey responses for relationship data."""
    responses = get_required_list(get_json_object(), 'responses')

    # This is a placeholder for survey analysis logic
    # In a real implementation, you might use NLP to extract relationships

    # For demo, just return dummy analysis
    result = {
        'relationships_found': len(responses),
        'relationships': [
            {'guest_id': f"G{i+1:03d}",
             'related_to': f"G{(i+5)%20+1:03d}",
             'strength': 0.8,
             'type': 'friend'}
            for i in range(min(len(responses), 10))
        ]
    }

    return jsonify(result)
