from flask import Blueprint, jsonify
from werkzeug.exceptions import BadRequest, NotFound
import matplotlib
# Render without a GUI. The default desktop backend crashes the server when it
# is used from a web request thread.
matplotlib.use('Agg')
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import io
import base64
import pandas as pd

from controllers.validation import get_json_object, get_required_list, is_number
from models.storage import get_event_data, save_graph, get_graph

# Create a Blueprint for graph routes
graph_routes = Blueprint('graph', __name__, url_prefix='/api')

EDGE_CHANGES = {'add_edge', 'remove_edge', 'update_edge'}
NODE_CHANGES = {'add_node', 'remove_node', 'update_node'}


def is_identifier(value):
    return isinstance(value, str) and value != ''


def validate_changes(G, changes):
    """Check every change first, so one bad change can't leave the graph half updated."""
    nodes = set(G.nodes)
    for position, change in enumerate(changes):
        where = f"changes[{position}]"
        if not isinstance(change, dict):
            raise BadRequest(f"{where} must be an object.")

        change_type = change.get('type')
        if change_type not in EDGE_CHANGES | NODE_CHANGES:
            allowed = ', '.join(sorted(EDGE_CHANGES | NODE_CHANGES))
            raise BadRequest(f"{where}.type must be one of: {allowed}.")

        source = change.get('source')
        if not is_identifier(source):
            raise BadRequest(f"{where}.source must be a non-empty string.")

        if change_type in EDGE_CHANGES and not is_identifier(change.get('target')):
            raise BadRequest(f"{where}.target must be a non-empty string.")

        attributes = change.get('attributes', {})
        if not isinstance(attributes, dict):
            raise BadRequest(f"{where}.attributes must be an object.")

        # The weight is drawn as the line width, so it can't be negative.
        if 'weight' in attributes and not (is_number(attributes['weight']) and attributes['weight'] >= 0):
            raise BadRequest(f"{where}.attributes.weight must be a non-negative number.")

        if change_type == 'add_edge':
            for node in (source, change['target']):
                if node not in nodes:
                    raise BadRequest(f"{where} refers to unknown guest '{node}'.")
        elif change_type == 'add_node':
            nodes.add(source)
        elif change_type == 'remove_node':
            nodes.discard(source)


@graph_routes.route('/events/<event_id>/build_graph', methods=['POST'])
def build_relationship_graph(event_id):
    """Build a relationship graph from guest data."""
    # Get event data or create if not exists
    ev_data = get_event_data(event_id)

    # For demo, create a dummy graph if no guest data
    if isinstance(ev_data['guests'], pd.DataFrame) and ev_data['guests'].empty:
        # Generate 20 dummy guests
        guests = []
        for i in range(20):
            guest = {
                'GuestID': f"G{i+1:03d}",
                'FullName': f"Guest {i+1}",
                'RSVP': True,
                'Relationship': 'Friend',
                'AgeRange': '31-45',
                'ComfortWithStrangers': bool(i % 2),
                'LikesConversations': bool((i+1) % 2),
                'DancingLikelihood': bool((i+2) % 2),
                'SignificantOther': f"G{(i+10)%20+1:03d}" if i % 3 == 0 else None
            }
            guests.append(guest)

        ev_data['guests'] = pd.DataFrame(guests)

    # Create a basic graph
    G = nx.Graph()

    # Add nodes (guests)
    for _, guest in ev_data['guests'].iterrows():
        G.add_node(guest['GuestID'], name=guest['FullName'])

    # Add edges (relationships)
    for _, guest in ev_data['guests'].iterrows():
        # Add edge to significant other if exists
        if pd.notna(guest['SignificantOther']):
            G.add_edge(guest['GuestID'], guest['SignificantOther'], type='couple', weight=1.0)

        # Add some random relationships for demo
        for _ in range(2):
            other_guest = ev_data['guests'].sample(1).iloc[0]
            if other_guest['GuestID'] != guest['GuestID']:
                G.add_edge(guest['GuestID'], other_guest['GuestID'],
                          type='friend', weight=np.random.random())

    # Store graph
    save_graph(event_id, G)

    # Return success
    return jsonify({
        'success': True,
        'graph_id': event_id,
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges()
    })


@graph_routes.route('/events/<event_id>/graph_visualization', methods=['GET'])
def get_graph_visualization(event_id):
    """Return a visualization of the relationship graph."""
    # Check if graph exists
    G = get_graph(event_id)
    if G is None:
        raise NotFound('Graph not found. Please build graph first.')

    # Each request gets its own Figure, because pyplot's shared "current figure"
    # isn't safe when requests run in parallel.
    figure = Figure(figsize=(12, 12))
    axes = figure.add_subplot()
    pos = nx.spring_layout(G, k=0.15, iterations=100)

    # Get edge weights
    edge_weights = [G.edges[e].get('weight', 0.5) for e in G.edges()]

    # Get edge colors based on type
    edge_colors = ['red' if G.edges[e].get('type') == 'couple' else 'blue' for e in G.edges()]

    # Draw the graph
    nx.draw(G, pos, ax=axes, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, width=edge_weights,
            edge_color=edge_colors)

    # Save to buffer
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png')
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Return the visualization
    return jsonify({
        'success': True,
        'image': f"data:image/png;base64,{image_base64}",
        'graph_data': {
            'nodes': [{'id': n, 'name': G.nodes[n].get('name', '')} for n in G.nodes()],
            'edges': [{'source': u, 'target': v,
                      'weight': G.edges[u, v].get('weight', 0.5),
                      'type': G.edges[u, v].get('type', 'unknown')}
                     for u, v in G.edges()]
        }
    })


@graph_routes.route('/events/<event_id>/update_graph', methods=['POST'])
def update_relationship_graph(event_id):
    """Update the relationship graph with manual changes."""
    changes = get_required_list(get_json_object(), 'changes')

    # Check if graph exists
    G = get_graph(event_id)
    if G is None:
        raise NotFound('Graph not found. Please build graph first.')

    validate_changes(G, changes)

    # Apply changes
    for change in changes:
        change_type = change.get('type')
        source = change.get('source')
        target = change.get('target')
        attributes = change.get('attributes', {})

        if change_type == 'add_edge':
            G.add_edge(source, target)
            G.edges[source, target].update(attributes)
        elif change_type == 'remove_edge':
            if G.has_edge(source, target):
                G.remove_edge(source, target)
        elif change_type == 'update_edge':
            if G.has_edge(source, target):
                G.edges[source, target].update(attributes)
        elif change_type == 'add_node':
            G.add_node(source)
            G.nodes[source].update(attributes)
        elif change_type == 'remove_node':
            if G.has_node(source):
                G.remove_node(source)
        elif change_type == 'update_node':
            if G.has_node(source):
                G.nodes[source].update(attributes)

    # Save updated graph
    save_graph(event_id, G)

    # Return success
    return jsonify({
        'success': True,
        'graph_id': event_id,
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges()
    })
