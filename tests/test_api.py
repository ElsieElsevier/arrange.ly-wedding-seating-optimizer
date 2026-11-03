"""API tests.

Seating quality isn't tested here because the clustering isn't written yet, and
some metrics are still hardcoded. These tests cover the API's behavior: valid
requests work, bad input gets a clear 400, and the server stays up.
"""
import base64
import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import pytest

from app import create_app, get_run_options

EVENT = '/api/events/demo'


def strict_json(response):
    """Parse a response, failing on NaN and Infinity, which aren't valid JSON."""
    def reject(constant):
        raise AssertionError(f'Response contains invalid JSON value {constant}')
    return json.loads(response.get_data(as_text=True), parse_constant=reject)


@pytest.fixture
def app():
    return create_app()  # also resets the in-memory storage


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def built(client):
    assert client.post(f'{EVENT}/build_graph').status_code == 200
    return client


@pytest.fixture
def optimized(client):
    response = client.post(f'{EVENT}/optimize', json={'config': {'num_tables': 5, 'capacity_per_table': 8}})
    assert response.status_code == 200
    return client, strict_json(response)


# ---------------------------------------------------------------- server setup

def test_root_and_docs(client):
    assert client.get('/').get_json()['status'] == 'online'
    assert len(client.get('/api/docs').get_json()['endpoints']) == 13


def test_server_defaults_are_local_and_not_debug():
    assert get_run_options({}) == {'host': '127.0.0.1', 'port': 5000, 'debug': False}


def test_server_options_can_be_set_explicitly():
    options = get_run_options({'HOST': '0.0.0.0', 'PORT': '8080', 'FLASK_DEBUG': '1'})
    assert options == {'host': '0.0.0.0', 'port': 8080, 'debug': True}


def test_app_debug_is_off_by_default(monkeypatch):
    monkeypatch.delenv('FLASK_DEBUG', raising=False)
    assert create_app().debug is False


def test_unknown_route_returns_json(client):
    response = client.get('/api/nope')
    assert response.status_code == 404
    assert response.get_json()['success'] is False


def test_unexpected_error_hides_details(client, monkeypatch):
    def explode(*args):
        raise RuntimeError('secret internal detail')
    monkeypatch.setattr('controllers.vectorization.np.random.rand', explode)

    response = client.post('/api/vectorize', json={'texts': ['hello']})

    assert response.status_code == 500
    assert response.get_json() == {'success': False, 'message': 'Internal server error'}


# ------------------------------------------------------------------- happy path

def test_vectorize(client):
    embeddings = client.post('/api/vectorize', json={'texts': ['a', 'b']}).get_json()['embeddings']
    assert [len(vector) for vector in embeddings] == [128, 128]


def test_analyze_survey_accepts_question_and_answer_objects(client):
    responses = [{'question': 'Who would you like to sit with?', 'answer': 'Jane'}]
    response = client.post('/api/analyze_survey', json={'event_id': 'demo', 'responses': responses})
    assert response.status_code == 200
    assert response.get_json()['relationships_found'] == 1


def test_build_graph(client):
    body = client.post(f'{EVENT}/build_graph').get_json()
    assert body['success'] is True
    assert body['node_count'] == 20


def test_visualization_returns_png(built):
    body = built.get(f'{EVENT}/graph_visualization').get_json()
    encoded = body['image'].removeprefix('data:image/png;base64,')
    assert base64.b64decode(encoded).startswith(b'\x89PNG')
    assert len(body['graph_data']['nodes']) == 20


def test_matplotlib_does_not_use_a_gui_backend():
    # The desktop backend killed the whole server when used from request threads.
    assert matplotlib.get_backend().lower() == 'agg'


def test_visualization_survives_parallel_requests(app, built):
    def fetch(_):
        return app.test_client().get(f'{EVENT}/graph_visualization')

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(fetch, range(8)))

    assert [response.status_code for response in responses] == [200] * 8


def test_empty_graph_can_be_drawn_and_analyzed(built):
    nodes = built.get(f'{EVENT}/graph_visualization').get_json()['graph_data']['nodes']
    changes = [{'type': 'remove_node', 'source': node['id']} for node in nodes]
    assert built.post(f'{EVENT}/update_graph', json={'changes': changes}).get_json()['node_count'] == 0

    assert built.get(f'{EVENT}/graph_visualization').status_code == 200
    assert strict_json(built.get(f'{EVENT}/relationship_analytics'))['analytics']['node_count'] == 0


def test_update_graph_applies_every_change_type(built):
    changes = [
        {'type': 'add_node', 'source': 'NEW', 'attributes': {'name': 'New Guest'}},
        {'type': 'update_node', 'source': 'NEW', 'attributes': {'name': 'Renamed'}},
        {'type': 'add_edge', 'source': 'NEW', 'target': 'G001', 'attributes': {'weight': 0.5, 'type': 'friend'}},
        {'type': 'update_edge', 'source': 'NEW', 'target': 'G001', 'attributes': {'weight': 0.9}},
    ]
    assert built.post(f'{EVENT}/update_graph', json={'changes': changes}).status_code == 200

    data = built.get(f'{EVENT}/graph_visualization').get_json()['graph_data']
    assert {'id': 'NEW', 'name': 'Renamed'} in data['nodes']
    assert any({edge['source'], edge['target']} == {'NEW', 'G001'} and edge['weight'] == 0.9 for edge in data['edges'])

    changes = [
        {'type': 'remove_edge', 'source': 'NEW', 'target': 'G001'},
        {'type': 'remove_node', 'source': 'NEW'},
    ]
    assert built.post(f'{EVENT}/update_graph', json={'changes': changes}).get_json()['node_count'] == 20


def test_optimize_uses_defaults_for_empty_config(client):
    response = client.post(f'{EVENT}/optimize', json={})
    assert response.status_code == 200
    job_id = response.get_json()['job_id']
    config = client.get(f'{EVENT}/optimize/{job_id}/results').get_json()['results']['config']
    assert config == {'num_tables': 10, 'capacity_per_table': 8}


def test_optimize_respects_config_and_seats_everyone(optimized):
    client, started = optimized
    results = strict_json(client.get(f"{EVENT}/optimize/{started['job_id']}/results"))['results']

    assert list(results['tables']) == [f'Table_{number}' for number in range(1, 6)]
    assert all(table['current_occupancy'] <= 8 for table in results['tables'].values())
    assert sum(table['current_occupancy'] for table in results['tables'].values()) == 20
    assert all(guest['TableAssignment'] for guest in results['guest_assignments'])


def test_job_status(optimized):
    client, started = optimized
    assert client.get(f"{EVENT}/optimize/{started['job_id']}/status").get_json()['status'] == 'complete'


def test_exactly_enough_seats_is_allowed(client):
    config = {'num_tables': 2, 'capacity_per_table': 10}
    assert client.post(f'{EVENT}/optimize', json={'config': config}).status_code == 200


def test_arrangement_endpoints(optimized):
    client, started = optimized
    arrangement_id = started['results']['arrangement_id']

    metrics = strict_json(client.get(f'{EVENT}/arrangements/{arrangement_id}/metrics'))['metrics']
    assert metrics['table_utilization'] == 0.5

    comparison = strict_json(client.post(f'{EVENT}/compare_arrangements', json={'arrangement_ids': [arrangement_id]}))
    assert comparison['comparison']['best_overall'] == arrangement_id

    recommendations = strict_json(client.get(f'{EVENT}/recommendations'))['recommendations']
    assert recommendations[0]['arrangement_id'] == arrangement_id


def test_score_arrangement(optimized):
    client, started = optimized
    results = client.get(f"{EVENT}/optimize/{started['job_id']}/results").get_json()['results']
    arrangement = {'guest_assignments': results['guest_assignments'], 'tables': results['tables']}

    body = strict_json(client.post(f'{EVENT}/score_arrangement', json={'arrangement': arrangement}))

    assert body['metrics']['assignment_rate'] == 1.0


def test_relationship_analytics(built):
    analytics = strict_json(built.get(f'{EVENT}/relationship_analytics'))['analytics']
    assert analytics['node_count'] == 20


# -------------------------------------------------------------------- bad input

BAD_JSON_BODIES = [
    '/api/vectorize',
    '/api/analyze_survey',
    f'{EVENT}/update_graph',
    f'{EVENT}/optimize',
    f'{EVENT}/score_arrangement',
    f'{EVENT}/compare_arrangements',
]


@pytest.mark.parametrize('path', BAD_JSON_BODIES)
@pytest.mark.parametrize('request_options', [
    pytest.param({}, id='no-body'),
    pytest.param({'data': 'hello', 'content_type': 'text/plain'}, id='not-json-content-type'),
    pytest.param({'data': '{broken', 'content_type': 'application/json'}, id='malformed-json'),
    pytest.param({'json': [1, 2]}, id='json-but-not-an-object'),
])
def test_bad_request_bodies_return_400(built, path, request_options):
    response = built.post(path, **request_options)
    assert response.status_code == 400
    assert response.get_json()['success'] is False


@pytest.mark.parametrize('path, body', [
    ('/api/vectorize', {}),
    ('/api/vectorize', {'texts': 'not a list'}),
    ('/api/vectorize', {'texts': [1]}),
    ('/api/analyze_survey', {'responses': 'not a list'}),
    (f'{EVENT}/update_graph', {}),
    (f'{EVENT}/update_graph', {'changes': 'not a list'}),
    (f'{EVENT}/update_graph', {'changes': ['not an object']}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'explode', 'source': 'G001'}]}),
    (f'{EVENT}/update_graph', {'changes': [{'source': 'G001', 'target': 'G002', 'relationship': 'friend'}]}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'add_node'}]}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'add_node', 'source': ['list']}]}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'add_edge', 'source': 'G001'}]}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'add_edge', 'source': 'G001', 'target': 'NOPE'}]}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'add_node', 'source': 'X', 'attributes': 'bad'}]}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'add_edge', 'source': 'G001', 'target': 'G002', 'attributes': {'weight': 'heavy'}}]}),
    (f'{EVENT}/update_graph', {'changes': [{'type': 'add_edge', 'source': 'G001', 'target': 'G002', 'attributes': {'weight': -1}}]}),
    (f'{EVENT}/score_arrangement', {}),
    (f'{EVENT}/score_arrangement', {'arrangement': 'bad'}),
    (f'{EVENT}/score_arrangement', {'arrangement': {'guest_assignments': 'bad'}}),
    (f'{EVENT}/score_arrangement', {'arrangement': {'tables': [{'id': 'table1'}]}}),
    (f'{EVENT}/score_arrangement', {'arrangement': {'tables': {'t1': 'bad'}}}),
    (f'{EVENT}/score_arrangement', {'arrangement': {'tables': {'t1': {'current_occupancy': 'many'}}}}),
    (f'{EVENT}/compare_arrangements', {}),
    (f'{EVENT}/compare_arrangements', {'arrangement_ids': 'bad'}),
    (f'{EVENT}/compare_arrangements', {'arrangement_ids': [1]}),
])
def test_invalid_fields_return_400(built, path, body):
    response = built.post(path, json=body)
    assert response.status_code == 400
    assert response.get_json()['success'] is False


@pytest.mark.parametrize('config', [
    'not an object',
    {'num_tables': 0},
    {'num_tables': -1},
    {'num_tables': 2.5},
    {'num_tables': '5'},
    {'num_tables': True},
    {'num_tables': 1001},
    {'capacity_per_table': 0},
    {'capacity_per_table': None},
])
def test_invalid_optimize_config_returns_400(client, config):
    assert client.post(f'{EVENT}/optimize', json={'config': config}).status_code == 400


def test_unknown_config_setting_is_rejected_by_name(client):
    # The old test script sent this name, and the API silently ignored it.
    response = client.post(f'{EVENT}/optimize', json={'config': {'seats_per_table': 2}})
    assert response.status_code == 400
    assert 'seats_per_table' in response.get_json()['message']


def test_too_few_seats_is_an_error(client):
    response = client.post(f'{EVENT}/optimize', json={'config': {'num_tables': 1, 'capacity_per_table': 3}})
    assert response.status_code == 400
    assert response.get_json()['message'] == 'Not enough seats: 20 guests but only 3 seats.'


def test_rejected_update_changes_nothing(built):
    changes = [
        {'type': 'remove_node', 'source': 'G001'},
        {'type': 'explode', 'source': 'G002'},
    ]
    assert built.post(f'{EVENT}/update_graph', json={'changes': changes}).status_code == 400

    nodes = built.get(f'{EVENT}/graph_visualization').get_json()['graph_data']['nodes']
    assert 'G001' in [node['id'] for node in nodes]


# -------------------------------------------------------------------- not found

@pytest.mark.parametrize('method, path, body', [
    ('get', f'{EVENT}/graph_visualization', None),
    ('post', f'{EVENT}/update_graph', {'changes': []}),
    ('get', f'{EVENT}/relationship_analytics', None),
    ('get', f'{EVENT}/optimize/missing/status', None),
    ('get', f'{EVENT}/optimize/missing/results', None),
    ('get', f'{EVENT}/arrangements/missing/metrics', None),
    ('post', f'{EVENT}/compare_arrangements', {'arrangement_ids': ['missing']}),
])
def test_missing_resources_return_404(client, method, path, body):
    response = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    assert response.status_code == 404
    assert response.get_json()['success'] is False
