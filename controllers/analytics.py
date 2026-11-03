from flask import Blueprint, jsonify
from werkzeug.exceptions import BadRequest, NotFound
import networkx as nx

from controllers.validation import get_json_object, get_required_list
from models.storage import (
    get_event_data, 
    get_graph, 
    get_arrangement,
    get_all_arrangements
)

# Create a Blueprint for analytics routes
analytics_routes = Blueprint('analytics', __name__, url_prefix='/api')

@analytics_routes.route('/events/<event_id>/relationship_analytics', methods=['GET'])
def get_relationship_analytics(event_id):
    """Generate relationship network analysis."""
    # Check if graph exists
    G = get_graph(event_id)
    if G is None:
        raise NotFound('Graph not found. Please build graph first.')
    
    # Calculate basic network metrics
    analytics = {
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges(),
        'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'density': nx.density(G),
    }
    
    try:
        analytics['clustering_coefficient'] = nx.average_clustering(G)
        analytics['connected_components'] = nx.number_connected_components(G)
    except:
        analytics['clustering_coefficient'] = 0
        analytics['connected_components'] = 0
    
    # Calculate centrality metrics
    try:
        degree_centrality = nx.degree_centrality(G)
        analytics['most_connected_guests'] = sorted(
            degree_centrality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
    except:
        analytics['most_connected_guests'] = []
    
    # Identify important groups
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        analytics['community_sizes'] = [len(c) for c in communities]
        analytics['community_count'] = len(communities)
    except:
        analytics['community_sizes'] = []
        analytics['community_count'] = 0
    
    return jsonify({
        'success': True,
        'analytics': analytics
    })


@analytics_routes.route('/events/<event_id>/arrangements/<arrangement_id>/metrics', methods=['GET'])
def get_seating_metrics(event_id, arrangement_id):
    """Generate detailed metrics for a specific seating arrangement."""
    # Get arrangement
    arrangement = get_arrangement(event_id, arrangement_id)
    if arrangement is None:
        raise NotFound('Arrangement not found')
    
    # Get base metrics
    metrics = arrangement.get('metrics', {})
    
    # Calculate additional metrics
    tables = arrangement.get('tables', {})
    guest_assignments = arrangement.get('guest_assignments', [])
    
    # Table utilization
    table_counts = {}
    for assignment in guest_assignments:
        table = assignment.get('TableAssignment')
        if table:
            table_counts[table] = table_counts.get(table, 0) + 1
    
    # Calculate table statistics
    if tables and table_counts:
        capacities = [table.get('capacity', 0) for table in tables.values()]
        occupancies = [table_counts.get(table, 0) for table in tables]
        
        metrics['table_utilization'] = sum(occupancies) / sum(capacities) if sum(capacities) > 0 else 0
        metrics['min_table_occupancy'] = min(table_counts.values()) if table_counts else 0
        metrics['max_table_occupancy'] = max(table_counts.values()) if table_counts else 0
        metrics['empty_tables'] = sum(1 for table in tables if table not in table_counts)
    
    # Return detailed metrics
    return jsonify({
        'success': True,
        'metrics': metrics,
        'table_occupancy': {table: count for table, count in table_counts.items()}
    })


@analytics_routes.route('/events/<event_id>/compare_arrangements', methods=['POST'])
def compare_arrangements(event_id):
    """Compare multiple seating arrangements."""
    arrangement_ids = get_required_list(get_json_object(), 'arrangement_ids')
    if not all(isinstance(arr_id, str) for arr_id in arrangement_ids):
        raise BadRequest("Every item in 'arrangement_ids' must be a string.")
    
    # Get arrangements
    arrangements = {}
    for arr_id in arrangement_ids:
        arr = get_arrangement(event_id, arr_id)
        if arr:
            arrangements[arr_id] = arr
    
    if not arrangements:
        raise NotFound('None of the specified arrangements were found')
    
    # Compare metrics
    comparison = {
        'arrangements': [],
        'metric_comparison': {}
    }
    
    for arr_id, arr in arrangements.items():
        # Add basic info
        comparison['arrangements'].append({
            'id': arr_id,
            'created_at': arr.get('created_at'),
            'metrics': arr.get('metrics', {})
        })
        
        # Add metrics to comparison
        metrics = arr.get('metrics', {})
        for metric, value in metrics.items():
            if metric not in comparison['metric_comparison']:
                comparison['metric_comparison'][metric] = {}
            comparison['metric_comparison'][metric][arr_id] = value
    
    # Find best arrangement overall
    best_arrangement = None
    best_score = -float('inf')
    
    for arr_id, arr in arrangements.items():
        metrics = arr.get('metrics', {})
        
        # Calculate a simple score (higher is better)
        score = (
            metrics.get('silhouette_score', 0) * 2 +
            metrics.get('couple_success_rate', 0) * 3 +
            (1 - min(1, metrics.get('table_balance', 0) / 3)) +  # Lower std dev is better
            metrics.get('assignment_rate', 0) * 4
        )
        
        if score > best_score:
            best_score = score
            best_arrangement = arr_id
    
    comparison['best_overall'] = best_arrangement
    
    return jsonify({
        'success': True,
        'comparison': comparison
    })


@analytics_routes.route('/events/<event_id>/recommendations', methods=['GET'])
def get_recommendations(event_id):
    """Generate arrangement recommendations."""
    # Get all arrangements
    arrangements = get_all_arrangements(event_id)
    
    # Generate recommendations
    recommendations = []
    
    if arrangements:
        # Find arrangement with best metrics
        best_arrangement = None
        best_score = -float('inf')
        
        for arr_id, arr in arrangements.items():
            metrics = arr.get('metrics', {})
            
            # Calculate a simple score (higher is better)
            score = (
                metrics.get('silhouette_score', 0) * 2 +
                metrics.get('couple_success_rate', 0) * 3 +
                (1 - min(1, metrics.get('table_balance', 0) / 3)) +  # Lower std dev is better
                metrics.get('assignment_rate', 0) * 4
            )
            
            if score > best_score:
                best_score = score
                best_arrangement = {
                    'id': arr_id,
                    'data': arr
                }
        
        if best_arrangement:
            recommendations.append({
                'type': 'optimal_arrangement',
                'arrangement_id': best_arrangement['id'],
                'metrics': best_arrangement['data'].get('metrics', {}),
                'description': 'This arrangement provides the best overall balance of guest satisfaction and table utilization.'
            })
    
    # Add a recommendation to try different parameters
    recommendations.append({
        'type': 'parameter_adjustment',
        'description': 'Try increasing the number of tables for more flexibility in seating arrangements.',
        'suggested_params': {
            'num_tables': 15,
            'capacity_per_table': 8
        }
    })
    
    return jsonify({
        'success': True,
        'recommendations': recommendations
    })
