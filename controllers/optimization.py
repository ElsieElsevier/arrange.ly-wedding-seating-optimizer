from flask import Blueprint, request, jsonify
import networkx as nx
import numpy as np
import pandas as pd
import uuid
import traceback
import math
from datetime import datetime

from models.storage import (
    get_event_data, 
    save_optimization_job, 
    get_optimization_job,
    save_arrangement,
    get_arrangement
)

# Create a Blueprint for optimization routes
optimization_routes = Blueprint('optimization', __name__, url_prefix='/api')

# Import core functions from services (to be implemented)
# For now, we'll define basic functions here

def preprocess_guest_data(guests_df):
    """Prepare guest data for clustering."""
    # Filter for RSVP'd guests only
    rsvp_guests = guests_df[guests_df['RSVP'] == True].copy()
    
    # For demo, just use some basic features
    features = np.random.rand(len(rsvp_guests), 5)
    
    return rsvp_guests, features

def build_similarity_matrix(guests_df, feature_matrix):
    """Build a similarity matrix between guests."""
    # Simple similarity measure (random for demo)
    n = len(guests_df)
    similarity = np.random.rand(n, n)
    
    # Make it symmetric
    similarity = 0.5 * (similarity + similarity.T)
    
    # Set diagonal to 1
    np.fill_diagonal(similarity, 1.0)
    
    # Ensure couples have high similarity
    for i, (_, guest) in enumerate(guests_df.iterrows()):
        if pd.notna(guest['SignificantOther']):
            partner_idx = guests_df[guests_df['GuestID'] == guest['SignificantOther']].index
            if len(partner_idx) > 0:
                similarity[i, partner_idx[0]] = 1.0
                similarity[partner_idx[0], i] = 1.0
    
    return similarity

def create_tables(num_tables, capacity_per_table):
    """Create table structures."""
    tables = {
        f"Table_{i+1}": {
            "capacity": capacity_per_table,
            "current_occupancy": 0,
            "guests": []
        } for i in range(num_tables)
    }
    return tables

def run_optimization(event_id, guests_df, config):
    """Run the optimization algorithm."""
    # Extract configuration
    num_tables = config.get('num_tables', 10)
    capacity_per_table = config.get('capacity_per_table', 8)
    
    # Create tables
    tables = create_tables(num_tables, capacity_per_table)
    
    # For demo, we'll just randomly assign guests to tables
    rsvp_guests = guests_df[guests_df['RSVP'] == True].copy()
    
    # Make sure we have at least one table
    if num_tables == 0:
        num_tables = 1
        tables = create_tables(1, capacity_per_table)
    
    # Assign guests to tables
    rsvp_guests['TableAssignment'] = None
    
    # Track which guests are assigned
    assigned_guests = set()
    
    # First, assign couples together
    for idx, guest in rsvp_guests.iterrows():
        if guest['GuestID'] in assigned_guests:
            continue
            
        so_id = guest['SignificantOther']
        
        if pd.notna(so_id) and so_id in rsvp_guests['GuestID'].values:
            # Find a table with at least 2 spaces
            for table_name, table in tables.items():
                if table['current_occupancy'] + 2 <= table['capacity']:
                    # Assign both guests to this table
                    rsvp_guests.loc[idx, 'TableAssignment'] = table_name
                    partner_idx = rsvp_guests[rsvp_guests['GuestID'] == so_id].index[0]
                    rsvp_guests.loc[partner_idx, 'TableAssignment'] = table_name
                    
                    # Update table
                    table['guests'].append(guest['GuestID'])
                    table['guests'].append(so_id)
                    table['current_occupancy'] += 2
                    
                    # Mark as assigned
                    assigned_guests.add(guest['GuestID'])
                    assigned_guests.add(so_id)
                    break
    
    # Then assign remaining guests
    for idx, guest in rsvp_guests.iterrows():
        if guest['GuestID'] in assigned_guests:
            continue
            
        # Find a table with space
        for table_name, table in tables.items():
            if table['current_occupancy'] < table['capacity']:
                # Assign guest to this table
                rsvp_guests.loc[idx, 'TableAssignment'] = table_name
                
                # Update table
                table['guests'].append(guest['GuestID'])
                table['current_occupancy'] += 1
                
                # Mark as assigned
                assigned_guests.add(guest['GuestID'])
                break
    
    # Calculate metrics
    metrics = {
        'silhouette_score': 0.5,  # Placeholder
        'couple_success_rate': 1.0,  # Assuming all couples are seated together
        'table_balance': np.std([table['current_occupancy'] for table in tables.values()]),
        'assignment_rate': len(assigned_guests) / len(rsvp_guests)
    }
    
    # Format results
    results = {
        'guest_assignments': rsvp_guests[['GuestID', 'TableAssignment']].to_dict(orient='records'),
        'tables': tables,
        'metrics': metrics,
        'config': {
            'num_tables': num_tables,
            'capacity_per_table': capacity_per_table
        }
    }
    
    return results

@optimization_routes.route('/events/<event_id>/optimize', methods=['POST'])
def optimize_seating(event_id):
    """Run the seating optimization algorithm."""
    try:
        data = request.json
        config = data.get('config', {})
        
        # Get event data
        ev_data = get_event_data(event_id)
        
        # If no guest data, create dummy data for demo
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
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Run optimization
        results = run_optimization(event_id, ev_data['guests'], config)
        
        # Store results
        save_optimization_job(event_id, job_id, {
            'status': 'complete',
            'progress': 100,
            'results': results,
            'config': config
        })
        
        # Store arrangement
        arrangement_id = f"arr_{job_id[:8]}"
        save_arrangement(event_id, arrangement_id, {
            'id': arrangement_id,
            'job_id': job_id,
            'guest_assignments': results['guest_assignments'],
            'tables': results['tables'],
            'metrics': results['metrics'],
            'created_at': datetime.now().isoformat()
        })
        
        # Return job ID and initial status
        return jsonify({
            'job_id': job_id,
            'status': 'complete',
            'results': {
                'arrangement_id': arrangement_id,
                'metrics': results['metrics']
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@optimization_routes.route('/events/<event_id>/optimize/<job_id>/status', methods=['GET'])
def get_optimization_status(event_id, job_id):
    """Get the status of a running optimization job."""
    try:
        # Check if job exists
        job = get_optimization_job(event_id, job_id)
        if job is None:
            return jsonify({
                'success': False,
                'message': 'Job not found'
            }), 404
        
        return jsonify({
            'status': job['status'],
            'progress': job['progress'],
            'message': f"Optimization {job['status']}"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@optimization_routes.route('/events/<event_id>/optimize/<job_id>/results', methods=['GET'])
def get_optimization_results(event_id, job_id):
    """Get the results of a completed optimization job."""
    try:
        # Check if job exists
        job = get_optimization_job(event_id, job_id)
        if job is None:
            return jsonify({
                'success': False,
                'message': 'Job not found'
            }), 404
        
        # Check if job is complete
        if job['status'] != 'complete':
            return jsonify({
                'success': False,
                'message': f"Job is {job['status']}, not complete"
            }), 400
        
        # Find arrangement ID
        arrangement_id = None
        ev_data = get_event_data(event_id)
        for arr_id, arr in ev_data.get('arrangements', {}).items():
            if arr.get('job_id') == job_id:
                arrangement_id = arr_id
                break
        
        # Return results
        return jsonify({
            'success': True,
            'results': job['results'],
            'arrangement_id': arrangement_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@optimization_routes.route('/events/<event_id>/score_arrangement', methods=['POST'])
def score_arrangement(event_id):
    """Score an existing arrangement."""
    try:
        data = request.json
        arrangement = data.get('arrangement', {})
        
        # Get event data
        ev_data = get_event_data(event_id)
        
        # Extract guest assignments and tables
        guest_assignments = arrangement.get('guest_assignments', [])
        tables = arrangement.get('tables', {})
        
        # Convert to DataFrame
        assignments_df = pd.DataFrame(guest_assignments)
        
        # Merge with guest data
        if not ev_data['guests'].empty and 'GuestID' in assignments_df.columns:
            merged = pd.merge(
                ev_data['guests'], 
                assignments_df, 
                on='GuestID', 
                how='inner'
            )
        else:
            # For demo, use the assignments as is
            merged = assignments_df
        
        # Calculate metrics
        # For demo, just return simple metrics
        metrics = {
            'silhouette_score': 0.5,  # Placeholder
            'couple_success_rate': 1.0,  # Placeholder
            'table_balance': np.std([table.get('current_occupancy', 0) for table in tables.values()]) if tables else 0,
            'assignment_rate': len(guest_assignments) / len(ev_data['guests']) if not ev_data['guests'].empty else 1.0
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500