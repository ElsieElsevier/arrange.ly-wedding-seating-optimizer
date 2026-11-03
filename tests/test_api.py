"""Tests for the seating API. Most use a small fake model so they run instantly."""
import hashlib
import json

import numpy as np
import pytest

import app as app_module
import embeddings
from app import create_app, get_run_options
from embeddings import EmbeddingUnavailable
from seating import SeatingError, plan_seating

TOPICS = ['doctor', 'hiker', 'coder', 'elder']


def fake_embed(texts):
    """One axis per topic word, plus small repeatable noise for each text."""
    rows = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        noise = np.random.default_rng(seed).normal(scale=0.05, size=len(TOPICS) + 8)
        for axis, topic in enumerate(TOPICS):
            if topic in text:
                noise[axis] += 1.0
        rows.append(noise)
    return np.array(rows)


def guest(name, description=None, **links):
    return {'name': name, 'description': description or f'{name} details', **links}


def topic_guests(per_topic=4, topics=TOPICS):
    return [
        guest(f'{topic}{number}', f'{topic} person number {number}')
        for topic in topics for number in range(per_topic)
    ]


def seated_at(result):
    return {seat['name']: table['table'] for table in result['tables'] for seat in table['guests']}


def strict_json(response):
    """Parse a response, failing on NaN and Infinity, which aren't valid JSON."""
    def reject(constant):
        raise AssertionError(f'Response contains invalid JSON value {constant}')
    return json.loads(response.get_data(as_text=True), parse_constant=reject)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, 'embed_texts', fake_embed)
    return create_app().test_client()


def post(client, guests, seats=4, **options):
    return client.post('/api/seating', json={'guests': guests, 'seats_per_table': seats, **options})


# ------------------------------------------------------------------ server setup

def test_page_is_served(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Arrangely' in response.data


def test_server_defaults_are_local_and_not_debug(monkeypatch):
    monkeypatch.delenv('FLASK_DEBUG', raising=False)
    assert get_run_options({}) == {'host': '127.0.0.1', 'port': 5000, 'debug': False}
    assert create_app().debug is False


def test_unexpected_error_hides_details(client, monkeypatch):
    def explode(*args):
        raise RuntimeError('secret internal detail')
    monkeypatch.setattr(app_module, 'plan_seating', explode)
    response = post(client, topic_guests())
    assert response.status_code == 500
    assert response.get_json() == {'message': 'Internal server error'}


def test_model_failure_gives_a_helpful_error(client, monkeypatch):
    def unavailable(texts):
        raise EmbeddingUnavailable('no network')
    monkeypatch.setattr(app_module, 'embed_texts', unavailable)
    response = post(client, topic_guests())
    assert response.status_code == 503
    assert 'internet' in response.get_json()['message']


# ------------------------------------------------------------------ seating logic

def test_similar_guests_are_seated_together(client):
    result = strict_json(post(client, topic_guests()))
    tables = seated_at(result)
    for topic in TOPICS:
        assert len({tables[f'{topic}{number}'] for number in range(4)}) == 1
    assert result['summary']['average_tablemate_similarity'] > result['summary']['random_seating_similarity']
    assert result['summary']['silhouette_score'] > 0.5


def test_number_of_tables_is_worked_out_when_omitted(client):
    result = strict_json(post(client, topic_guests(), seats=5))
    assert result['summary']['tables_used'] == 4


def test_same_input_gives_same_plan(client):
    assert post(client, topic_guests()).get_json() == post(client, topic_guests()).get_json()


def test_partners_stay_together_even_when_very_different(client):
    guests = topic_guests()
    guests[0]['partner'] = 'coder1'
    tables = seated_at(strict_json(post(client, guests)))
    assert tables['doctor0'] == tables['coder1']


def test_partner_link_only_needs_to_be_set_on_one_guest(client):
    guests = topic_guests()
    guests[1]['partner'] = 'elder2'
    tables = seated_at(strict_json(post(client, guests)))
    assert tables['doctor1'] == tables['elder2']


def test_avoid_separates_even_near_identical_guests(client):
    guests = topic_guests()
    guests[0]['avoid'] = ['doctor1']
    result = strict_json(post(client, guests))
    tables = seated_at(result)
    assert tables['doctor0'] != tables['doctor1']
    assert result['summary']['avoid_conflicts'] == 0
    assert result['summary']['avoid_requests'] == 1


def test_impossible_avoid_is_reported_rather_than_hidden(client):
    guests = [guest('A', avoid=['B']), guest('B')]
    result = strict_json(post(client, guests, seats=2, num_tables=1))
    assert result['summary']['avoid_conflicts'] == 1


def test_identical_descriptions_still_work(client):
    guests = [guest(f'G{number}', 'exactly the same words') for number in range(12)]
    result = strict_json(post(client, guests))
    assert sorted(len(table['guests']) for table in result['tables']) == [4, 4, 4]


def test_each_guest_alone_at_a_table(client):
    result = strict_json(post(client, [guest('A'), guest('B')], seats=1))
    assert result['summary']['tables_used'] == 2
    assert result['summary']['average_tablemate_similarity'] is None
    assert result['tables'][0]['guests'][0]['closest_tablemate'] is None


def test_more_tables_than_couples_leaves_extras_unused(client):
    guests = [guest('A', partner='B'), guest('B'), guest('C', partner='D'), guest('D')]
    result = strict_json(post(client, guests, seats=2, num_tables=4))
    tables = seated_at(result)
    assert tables['A'] == tables['B'] and tables['C'] == tables['D']


def test_one_table_seats_everyone(client):
    result = strict_json(post(client, topic_guests(per_topic=1), seats=8, num_tables=1))
    assert result['summary']['tables_used'] == 1
    assert result['summary']['silhouette_score'] is None


def test_couple_stranded_by_clustering_is_still_seated():
    # Tables of 3 with singles forming their own cluster would strand the couples
    # unless the planner falls back to packing them first.
    guests = [guest('A1', partner='A2'), guest('A2'), guest('B1', partner='B2'), guest('B2'),
              guest('S1'), guest('S2')]
    vectors = fake_embed(['doctor', 'doctor', 'hiker', 'hiker', 'coder', 'coder'])
    tables = seated_at(plan_seating(guests, 3, 2, vectors))
    assert tables['A1'] == tables['A2'] and tables['B1'] == tables['B2']


def test_couples_that_cannot_fit_raise_a_clear_error():
    guests = [guest(name, partner=partner) for name, partner in
              [('A', 'B'), ('B', None), ('C', 'D'), ('D', None), ('E', 'F'), ('F', None)]]
    with pytest.raises(SeatingError, match='do not fit'):
        plan_seating(guests, 3, 2, fake_embed([g['description'] for g in guests]))


@pytest.mark.parametrize('seed', range(40))
def test_random_scenarios_keep_every_rule(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(6, 60))
    seats = int(rng.integers(3, 11))
    names = [f'G{number}' for number in range(count)]
    guests = [guest(name) for name in names]

    # Random couples, using each guest at most once.
    order = rng.permutation(count).tolist()
    couple_count = int(rng.integers(0, count // 3 + 1))
    couples = [(order.pop(), order.pop()) for _ in range(couple_count)]
    for a, b in couples:
        guests[a]['partner'] = names[b]

    # Random keep-apart requests between guests who aren't a couple.
    single_pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
    avoids = single_pairs[:int(rng.integers(0, 4))]
    for a, b in avoids:
        guests[a]['avoid'] = [names[b]]

    tables_needed = int(np.ceil(count / seats)) + int(rng.integers(1, 3))
    vectors = rng.normal(size=(count, 16))
    result = plan_seating(guests, seats, tables_needed, vectors)
    tables = seated_at(result)

    assert sorted(tables) == sorted(names)
    assert all(len(table['guests']) <= seats for table in result['tables'])
    assert all(tables[names[a]] == tables[names[b]] for a, b in couples)
    assert all(tables[names[a]] != tables[names[b]] for a, b in avoids)
    assert result['summary']['avoid_conflicts'] == 0


@pytest.mark.parametrize('seed', range(6))
def test_nobody_is_stranded_at_a_nearly_empty_table(seed):
    # Some unevenness is intended, so that guests can stay with their own circle.
    vectors = np.random.default_rng(seed).normal(size=(60, 16))
    guests = [guest(f'G{number}') for number in range(60)]
    sizes = [len(table['guests']) for table in plan_seating(guests, 10, 8, vectors)['tables']]
    assert len(sizes) == 8
    assert min(sizes) >= 4
    assert max(sizes) <= 10


# ---------------------------------------------------------------------- bad input

@pytest.mark.parametrize('options', [
    pytest.param({}, id='no-body'),
    pytest.param({'data': 'hello', 'content_type': 'text/plain'}, id='not-json'),
    pytest.param({'data': '{broken', 'content_type': 'application/json'}, id='malformed'),
    pytest.param({'json': [1, 2]}, id='not-an-object'),
])
def test_bad_bodies_return_400(client, options):
    assert client.post('/api/seating', **options).status_code == 400


GOOD = [guest('A'), guest('B')]


@pytest.mark.parametrize('body', [
    pytest.param({'seats_per_table': 4}, id='no-guests'),
    pytest.param({'guests': GOOD}, id='no-seats'),
    pytest.param({'guests': 'A,B', 'seats_per_table': 4}, id='guests-not-a-list'),
    pytest.param({'guests': [guest('A')], 'seats_per_table': 4}, id='one-guest'),
    pytest.param({'guests': [guest(str(n)) for n in range(501)], 'seats_per_table': 50}, id='too-many-guests'),
    pytest.param({'guests': GOOD, 'seats_per_table': 0}, id='zero-seats'),
    pytest.param({'guests': GOOD, 'seats_per_table': 51}, id='too-many-seats'),
    pytest.param({'guests': GOOD, 'seats_per_table': 2.5}, id='fractional-seats'),
    pytest.param({'guests': GOOD, 'seats_per_table': True}, id='boolean-seats'),
    pytest.param({'guests': GOOD, 'seats_per_table': '4'}, id='text-seats'),
    pytest.param({'guests': GOOD, 'seats_per_table': 4, 'num_tables': 0}, id='zero-tables'),
    pytest.param({'guests': GOOD, 'seats_per_table': 4, 'num_tables': 3}, id='more-tables-than-guests'),
    pytest.param({'guests': GOOD, 'seats_per_table': 4, 'extra': 1}, id='unknown-field'),
    pytest.param({'guests': ['A', 'B'], 'seats_per_table': 4}, id='guest-not-object'),
    pytest.param({'guests': [{'name': 'A'}, guest('B')], 'seats_per_table': 4}, id='no-description'),
    pytest.param({'guests': [guest('A', '   '), guest('B')], 'seats_per_table': 4}, id='blank-description'),
    pytest.param({'guests': [guest('A', 'x' * 2001), guest('B')], 'seats_per_table': 4}, id='long-description'),
    pytest.param({'guests': [{'description': 'x'}, guest('B')], 'seats_per_table': 4}, id='no-name'),
    pytest.param({'guests': [guest('A'), guest('a')], 'seats_per_table': 4}, id='duplicate-names'),
    pytest.param({'guests': [guest('A', age=9), guest('B')], 'seats_per_table': 4}, id='unknown-guest-field'),
    pytest.param({'guests': [guest('A', partner='Z'), guest('B')], 'seats_per_table': 4}, id='unknown-partner'),
    pytest.param({'guests': [guest('A', partner='A'), guest('B')], 'seats_per_table': 4}, id='own-partner'),
    pytest.param({'guests': [guest('A', partner=7), guest('B')], 'seats_per_table': 4}, id='partner-not-text'),
    pytest.param({'guests': [guest('A', avoid='B'), guest('B')], 'seats_per_table': 4}, id='avoid-not-a-list'),
    pytest.param({'guests': [guest('A', avoid=['Z']), guest('B')], 'seats_per_table': 4}, id='unknown-avoid'),
    pytest.param({'guests': [guest('A', avoid=['A']), guest('B')], 'seats_per_table': 4}, id='avoid-self'),
    pytest.param({'guests': [guest('A', avoid=[7]), guest('B')], 'seats_per_table': 4}, id='avoid-not-text'),
])
def test_invalid_requests_return_400(client, body):
    response = client.post('/api/seating', json=body)
    assert response.status_code == 400
    assert response.get_json()['message']


def test_too_few_seats_message(client):
    response = post(client, topic_guests(per_topic=1), seats=1, num_tables=2)
    assert response.status_code == 400
    assert response.get_json()['message'] == 'Not enough seats: 4 guests but only 2 seats.'


def test_couple_larger_than_table_message(client):
    response = post(client, [guest('A', partner='B'), guest('B')], seats=1)
    assert response.status_code == 400
    assert 'A, B must sit together' in response.get_json()['message']


def test_partner_who_must_also_be_avoided_is_rejected(client):
    response = post(client, [guest('A', partner='B', avoid=['B']), guest('B')])
    assert response.status_code == 400
    assert 'partners' in response.get_json()['message']


# --------------------------------------------------------------- real language model

@pytest.mark.model
def test_real_model_keeps_best_friend_with_her_circle_when_circles_are_uneven():
    # The page's example. Evening out the tables too strongly once seated the bride's
    # best friend with the teenagers, because her circle was the largest.
    texts = [
        "Bride's college roommate, now a lawyer in Chicago. Loves wine and karaoke.",
        "Maya's husband, a startup founder who is outgoing and loves to dance.",
        "Was in the bride's sorority. Very outgoing and always first on the dance floor.",
        'Studied abroad in Spain with the bride. Funny and talks to everybody.',
        "Bride's best friend from freshman year, a marketing director who loves to party.",
        "Groom's manager at the accounting firm, in his fifties and a big golf fan.",
        "Tom's wife, also in finance, enjoys golf and quiet conversation.",
        'Sits next to the groom at the office and talks about spreadsheets and fantasy football.',
        'Colleague of the groom in the finance department, and a new parent.',
        "Bride's grandmother, 84 and hard of hearing. Would like to be away from the speakers.",
        "Bride's grandfather, a retired farmer who tells long stories about the old days.",
        "Groom's great-aunt who uses a walker and goes to bed early.",
        "Groom's elderly uncle, a retired pastor who doesn't drink.",
        "Groom's teenage nephew, who mostly wants to play on his phone.",
        "Bride's 16-year-old cousin, shy and into anime and drawing.",
        "Flower girl's older brother, age 13, who loves Minecraft.",
    ]
    guests = [guest(f'G{number}', text) for number, text in enumerate(texts)]
    guests[0]['partner'] = 'G1'
    guests[5]['partner'] = 'G6'
    guests[9]['partner'] = 'G10'
    guests[4]['avoid'] = ['G5']

    tables = seated_at(plan_seating(guests, 6, 4, embeddings.embed_texts(texts)))

    def table_set(numbers):
        return {tables[f'G{number}'] for number in numbers}

    assert len(table_set(range(5))) == 1      # friends, including the best friend
    assert len(table_set(range(5, 9))) == 1   # coworkers
    assert len(table_set(range(9, 13))) == 1  # elders
    assert len(table_set(range(13, 16))) == 1 # teens
    assert len(set(tables.values())) == 4


@pytest.mark.model
def test_real_model_groups_by_meaning_not_shared_words():
    circles = {
        'medicine': ['Pediatric nurse on night shifts.', 'Orthopedic surgeon, always on call.',
                     'Paramedic with ambulance stories.', 'Radiologist who reads scans at the clinic.'],
        'outdoors': ['Backpacks in the Rockies every weekend.', 'Rock climber just back from Yosemite.',
                     'Trail runner training for an ultramarathon.', 'Kayaks rivers and camps under the stars.'],
        'elders': ['Grandmother of nine who bakes pies for church.', 'Retired teacher who knits and does crosswords.',
                   'Great-uncle who loves big band music.', 'Widowed grandfather who plays bingo.'],
    }
    guests = [guest(f'{circle}{number}', text)
              for circle, texts in circles.items() for number, text in enumerate(texts)]

    result = plan_seating(guests, 4, 3, embeddings.embed_texts([g['description'] for g in guests]))

    tables = seated_at(result)
    for circle in circles:
        assert len({tables[f'{circle}{number}'] for number in range(4)}) == 1
