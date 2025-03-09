from flask import Blueprint, request, jsonify
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import traceback
import pandas as pd

from models.storage import get_event_data, save_graph, get_graph

# Create a Blueprint for graph routes
graph_routes = Blueprint('graph', __name__, url_prefix='/api')

@graph_routes.route('/events/<event_id>/build_graph', methods=['POST'])
def build_relationship_graph(event_id):
    """Build a relationship graph from guest data."""
    try:
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
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@graph_routes.route('/events/<event_id>/graph_visualization', methods=['GET'])
def get_graph_visualization(event_id):
    """Return a visualization of the relationship graph."""
    try:
        # Check if graph exists
        G = get_graph(event_id)
        if G is None:
            return jsonify({
                'success': False,
                'message': 'Graph not found. Please build graph first.'
            }), 404
        
        # Create visualization
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.15, iterations=100)
        
        # Get edge weights
        edge_weights = [G.edges[e].get('weight', 0.5) for e in G.edges()]
        
        # Get edge colors based on type
        edge_colors = ['red' if G.edges[e].get('type') == 'couple' else 'blue' for e in G.edges()]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, width=edge_weights, 
                edge_color=edge_colors, connectionstyle='arc3,rad=0.1')
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
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
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@graph_routes.route('/events/<event_id>/update_graph', methods=['POST'])
def update_relationship_graph(event_id):
    """Update the relationship graph with manual changes."""
    try:
        data = request.json
        changes = data.get('changes', [])
        
        # Check if graph exists
        G = get_graph(event_id)
        if G is None:
            return jsonify({
                'success': False,
                'message': 'Graph not found. Please build graph first.'
            }), 404
        
        # Apply changes
        for change in changes:
            change_type = change.get('type')
            source = change.get('source')
            target = change.get('target')
            
            if change_type == 'add_edge':
                G.add_edge(source, target, **change.get('attributes', {}))
            elif change_type == 'remove_edge':
                if G.has_edge(source, target):
                    G.remove_edge(source, target)
            elif change_type == 'update_edge':
                if G.has_edge(source, target):
                    for key, value in change.get('attributes', {}).items():
                        G.edges[source, target][key] = value
            elif change_type == 'add_node':
                G.add_node(source, **change.get('attributes', {}))
            elif change_type == 'remove_node':
                if G.has_node(source):
                    G.remove_node(source)
            elif change_type == 'update_node':
                if G.has_node(source):
                    for key, value in change.get('attributes', {}).items():
                        G.nodes[source][key] = value
        
        # Save updated graph
        save_graph(event_id, G)
        
        # Return success
        return jsonify({
            'success': True,
            'graph_id': event_id,
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500