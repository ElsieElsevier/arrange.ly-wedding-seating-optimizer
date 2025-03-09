import networkx as nx
import pandas as pd

# In-memory storage for demonstration purposes
# In a production environment, you would use a database
optimization_jobs = {}
event_data = {}
graphs = {}
arrangements = {}

def init_storage():
    """Initialize the storage."""
    # This would connect to databases in a production setting
    # For now, just ensure our in-memory dictionaries are initialized
    global optimization_jobs, event_data, graphs, arrangements
    optimization_jobs = {}
    event_data = {}
    graphs = {}
    arrangements = {}
    print("Storage initialized")

def get_event_data(event_id):
    """Retrieve event data for a specific event."""
    if event_id not in event_data:
        # For demo, create dummy data if not exists
        event_data[event_id] = {
            'guests': pd.DataFrame(),
            'arrangements': {}
        }
    return event_data[event_id]

def save_graph(event_id, graph):
    """Save a graph for an event."""
    graphs[event_id] = graph
    return True

def get_graph(event_id):
    """Retrieve a graph for an event."""
    return graphs.get(event_id, None)

def save_optimization_job(event_id, job_id, job_data):
    """Save optimization job data."""
    if event_id not in optimization_jobs:
        optimization_jobs[event_id] = {}
    
    optimization_jobs[event_id][job_id] = job_data
    return True

def get_optimization_job(event_id, job_id):
    """Retrieve optimization job data."""
    if event_id not in optimization_jobs:
        return None
    return optimization_jobs[event_id].get(job_id, None)

def save_arrangement(event_id, arrangement_id, arrangement_data):
    """Save arrangement data."""
    ev_data = get_event_data(event_id)
    if 'arrangements' not in ev_data:
        ev_data['arrangements'] = {}
    
    ev_data['arrangements'][arrangement_id] = arrangement_data
    return True

def get_arrangement(event_id, arrangement_id):
    """Retrieve arrangement data."""
    ev_data = get_event_data(event_id)
    if 'arrangements' not in ev_data:
        return None
    return ev_data['arrangements'].get(arrangement_id, None)

def get_all_arrangements(event_id):
    """Retrieve all arrangements for an event."""
    ev_data = get_event_data(event_id)
    return ev_data.get('arrangements', {})