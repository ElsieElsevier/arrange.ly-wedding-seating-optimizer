from flask import Flask
from flask_cors import CORS
import os

# Import controllers (route handlers)
from controllers.vectorization import vectorization_routes
from controllers.graph import graph_routes
from controllers.optimization import optimization_routes
from controllers.analytics import analytics_routes

# Import storage
from models.storage import init_storage

# Create and configure Flask app
def create_app():
    app = Flask(__name__)
    
    # Configure app from config.py if it exists
    if os.path.exists('config.py'):
        app.config.from_pyfile('config.py')
    
    # Enable CORS for all routes
    CORS(app)
    
    # Initialize storage
    init_storage()
    
    # Register routes from controllers
    app.register_blueprint(vectorization_routes)
    app.register_blueprint(graph_routes)
    app.register_blueprint(optimization_routes)
    app.register_blueprint(analytics_routes)
    
    # Add a simple root route
    @app.route('/')
    def index():
        return {
            'status': 'online',
            'api_name': 'Arrangely API',
            'version': '1.0.0',
            'documentation': '/api/docs'
        }
    
    # Add a documentation route
    @app.route('/api/docs')
    def docs():
        return {
            'endpoints': [
                {'path': '/api/vectorize', 'method': 'POST', 'description': 'Convert text inputs to vector embeddings'},
                {'path': '/api/analyze_survey', 'method': 'POST', 'description': 'Process survey responses for relationship data'},
                {'path': '/api/events/<event_id>/build_graph', 'method': 'POST', 'description': 'Build a relationship graph from guest data'},
                {'path': '/api/events/<event_id>/graph_visualization', 'method': 'GET', 'description': 'Get a visualization of the relationship graph'},
                {'path': '/api/events/<event_id>/update_graph', 'method': 'POST', 'description': 'Update the relationship graph with manual changes'},
                {'path': '/api/events/<event_id>/optimize', 'method': 'POST', 'description': 'Run the seating optimization algorithm'},
                {'path': '/api/events/<event_id>/optimize/<job_id>/status', 'method': 'GET', 'description': 'Get the status of an optimization job'},
                {'path': '/api/events/<event_id>/optimize/<job_id>/results', 'method': 'GET', 'description': 'Get the results of a completed optimization job'},
                {'path': '/api/events/<event_id>/score_arrangement', 'method': 'POST', 'description': 'Score an existing arrangement'},
                {'path': '/api/events/<event_id>/relationship_analytics', 'method': 'GET', 'description': 'Generate relationship network analysis'},
                {'path': '/api/events/<event_id>/arrangements/<arrangement_id>/metrics', 'method': 'GET', 'description': 'Generate seating quality metrics'},
                {'path': '/api/events/<event_id>/compare_arrangements', 'method': 'POST', 'description': 'Compare multiple arrangements'},
                {'path': '/api/events/<event_id>/recommendations', 'method': 'GET', 'description': 'Generate arrangement recommendations'},
            ]
        }
    
    return app

# Create the application instance
app = create_app()

# Run the app if executed directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=True)