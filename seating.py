"""Turn guest descriptions into table assignments using spectral clustering.

1. Couples and other linked guests become one unit, so they can't be separated.
2. Spectral clustering groups the units by how similar their descriptions are.
3. Clusters are adjusted to fit the tables, and "avoid" requests are enforced.
"""
import math

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# Large enough that no amount of similarity can outweigh an "avoid" request.
AVOID_PENALTY = 1000.0
# How strongly tables are pushed toward equal sizes. Too strong, and a guest gets
# seated with strangers just to even out a table. Too weak, and guests end up
# alone. Chosen by comparing values on sample guest lists.
BALANCE_WEIGHT = 0.15
# How many of a unit's nearest neighbors set its local similarity scale.
NEIGHBORS_FOR_SCALE = 5
MAX_REFINEMENT_PASSES = 50


class SeatingError(Exception):
    """The request can't be seated, for a reason the user can fix."""


def plan_seating(guests, seats_per_table, num_tables, embeddings):
    """Assign guests to tables.

    guests: list of dicts with 'name' and optional 'partner' and 'avoid' names.
    embeddings: array with one row per guest.
    """
    index_of = {guest['name']: index for index, guest in enumerate(guests)}
    similarity = _cosine_similarity(embeddings)
    units = _build_units(guests, index_of)
    avoid_pairs = _avoid_pairs(guests, index_of)
    _check_requests_are_possible(guests, units, avoid_pairs, seats_per_table)

    unit_of = {guest: unit for unit, members in enumerate(units) for guest in members}
    sizes = np.array([len(members) for members in units])
    weights = _unit_weights(similarity, units, avoid_pairs, unit_of)

    labels = _spectral_labels(similarity, units, avoid_pairs, unit_of, num_tables)
    assignment = _fit_to_tables(labels, weights, sizes, num_tables, seats_per_table)
    assignment = _refine(assignment, weights, sizes, num_tables, seats_per_table)

    guest_tables = np.empty(len(guests), dtype=int)
    for unit, members in enumerate(units):
        guest_tables[members] = assignment[unit]

    return _build_result(guests, guest_tables, similarity, avoid_pairs, seats_per_table)


def _cosine_similarity(embeddings):
    vectors = np.asarray(embeddings, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(lengths == 0, 1, lengths)
    return np.clip(vectors @ vectors.T, -1.0, 1.0)


def _build_units(guests, index_of):
    """Group guests linked by 'partner' so each group stays at one table."""
    parents = list(range(len(guests)))

    def find(guest):
        while parents[guest] != guest:
            parents[guest] = parents[parents[guest]]
            guest = parents[guest]
        return guest

    for index, guest in enumerate(guests):
        if guest.get('partner'):
            parents[find(index)] = find(index_of[guest['partner']])

    units = {}
    for index in range(len(guests)):
        units.setdefault(find(index), []).append(index)
    return list(units.values())


def _avoid_pairs(guests, index_of):
    return {
        tuple(sorted((index, index_of[name])))
        for index, guest in enumerate(guests)
        for name in guest.get('avoid', [])
    }


def _check_requests_are_possible(guests, units, avoid_pairs, seats_per_table):
    for members in units:
        names = ', '.join(guests[member]['name'] for member in members)
        if len(members) > seats_per_table:
            raise SeatingError(
                f"{names} must sit together, but tables only have {seats_per_table} seat(s)."
            )
        if any(a in members and b in members for a, b in avoid_pairs):
            raise SeatingError(f"{names} are linked as partners, but an avoid request separates them.")


def _unit_weights(similarity, units, avoid_pairs, unit_of):
    """Score each pair of units for sharing a table. Above-average pairs score positive."""
    above_average = similarity - _average_pair_similarity(similarity)
    weights = np.zeros((len(units), len(units)))
    for a, members_a in enumerate(units):
        for b, members_b in enumerate(units):
            if a != b:
                weights[a, b] = above_average[np.ix_(members_a, members_b)].sum()
    for guest_a, guest_b in avoid_pairs:
        a, b = unit_of[guest_a], unit_of[guest_b]
        weights[a, b] = weights[b, a] = -AVOID_PENALTY
    return weights


def _average_pair_similarity(similarity):
    count = len(similarity)
    if count < 2:
        return 0.0
    return float((similarity.sum() - np.trace(similarity)) / (count * (count - 1)))


def _spectral_labels(similarity, units, avoid_pairs, unit_of, num_tables):
    count = len(units)
    if num_tables == 1:
        return np.zeros(count, dtype=int)
    if count <= num_tables:
        return np.arange(count)

    affinity = _affinity_matrix(similarity, units)
    for guest_a, guest_b in avoid_pairs:
        a, b = unit_of[guest_a], unit_of[guest_b]
        affinity[a, b] = affinity[b, a] = 0.0

    return SpectralClustering(
        n_clusters=num_tables,
        affinity='precomputed',
        assign_labels='kmeans',
        n_init=20,
        random_state=0,
    ).fit_predict(affinity)


def _affinity_matrix(similarity, units):
    """Convert similarity into a graph, scaling distances to each unit's neighborhood.

    Text similarities bunch into a narrow range, and a local scale spreads them out
    so that spectral clustering can see the groups. See Zelnik-Manor & Perona (2004).
    """
    count = len(units)
    distance = np.zeros((count, count))
    for a, members_a in enumerate(units):
        for b, members_b in enumerate(units):
            if a != b:
                distance[a, b] = 1.0 - similarity[np.ix_(members_a, members_b)].mean()

    neighbor = min(NEIGHBORS_FOR_SCALE, count - 1)
    scale = np.maximum(np.sort(distance, axis=1)[:, neighbor], 1e-6)
    affinity = np.exp(-(distance ** 2) / np.outer(scale, scale))
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _fit_to_tables(labels, weights, sizes, num_tables, seats_per_table):
    """Move units out of overfull clusters into the tables where they fit best."""
    assignment = np.asarray(labels).copy()
    loads = np.bincount(assignment, weights=sizes, minlength=num_tables).astype(int)
    waiting = []

    for table in range(num_tables):
        while loads[table] > seats_per_table:
            members = np.flatnonzero(assignment == table)
            fit = weights[np.ix_(members, members)].sum(axis=1)
            weakest = members[np.argmin(fit)]
            assignment[weakest] = -1
            loads[table] -= sizes[weakest]
            waiting.append(int(weakest))

    for unit in sorted(waiting, key=lambda candidate: -sizes[candidate]):
        open_tables = np.flatnonzero(loads + sizes[unit] <= seats_per_table)
        if len(open_tables) == 0:
            return _pack_evenly(sizes, num_tables, seats_per_table)
        fit = [weights[unit, assignment == table].sum() for table in open_tables]
        table = open_tables[int(np.argmax(fit))]
        assignment[unit] = table
        loads[table] += sizes[unit]

    return assignment


def _pack_evenly(sizes, num_tables, seats_per_table):
    """Fallback when a couple is stranded: seat the largest units first at the emptiest table."""
    assignment = np.full(len(sizes), -1)
    loads = np.zeros(num_tables, dtype=int)
    for unit in np.argsort(-sizes, kind='stable'):
        table = int(np.argmin(loads))
        if loads[table] + sizes[unit] > seats_per_table:
            raise SeatingError('These couples and groups do not fit. Try adding a table or more seats.')
        assignment[unit] = table
        loads[table] += sizes[unit]
    return assignment


def _refine(assignment, weights, sizes, num_tables, seats_per_table):
    """Improve the seating with single moves and swaps until nothing helps.

    This evens out table sizes and resolves avoid requests. Every accepted change
    raises the total score, so the loop always finishes.
    """
    assignment = assignment.copy()
    count = len(sizes)
    target = sizes.sum() / num_tables
    loads = np.bincount(assignment, weights=sizes, minlength=num_tables).astype(float)
    table_scores = np.zeros((count, num_tables))
    np.add.at(table_scores.T, assignment, weights)
    everyone = np.arange(count)

    def balance_cost(values):
        return BALANCE_WEIGHT * (values - target) ** 2

    def move(unit, table):
        old_table = assignment[unit]
        table_scores[:, old_table] -= weights[unit]
        table_scores[:, table] += weights[unit]
        loads[old_table] -= sizes[unit]
        loads[table] += sizes[unit]
        assignment[unit] = table

    for _ in range(MAX_REFINEMENT_PASSES):
        improved = False
        for unit in range(count):
            current = assignment[unit]
            size = sizes[unit]

            # Option 1: move this unit to another table with room.
            move_gains = table_scores[unit] - table_scores[unit, current] - (
                balance_cost(loads + size) - balance_cost(loads)
                + balance_cost(loads[current] - size) - balance_cost(loads[current])
            )
            move_gains[current] = -np.inf
            move_gains[loads + size > seats_per_table] = -np.inf
            best_table = int(np.argmax(move_gains))

            # Option 2: swap this unit with a unit at another table.
            other_tables = assignment
            difference = sizes - size
            swap_gains = (
                table_scores[unit, other_tables] - table_scores[unit, current]
                + table_scores[everyone, current] - table_scores[everyone, other_tables]
                - 2 * weights[unit]
                - balance_cost(loads[current] + difference) + balance_cost(loads[current])
                - balance_cost(loads[other_tables] - difference) + balance_cost(loads[other_tables])
            )
            swap_gains[other_tables == current] = -np.inf
            swap_gains[loads[current] + difference > seats_per_table] = -np.inf
            swap_gains[loads[other_tables] - difference > seats_per_table] = -np.inf
            best_partner = int(np.argmax(swap_gains))

            if max(move_gains[best_table], swap_gains[best_partner]) <= 1e-9:
                continue

            improved = True
            if move_gains[best_table] >= swap_gains[best_partner]:
                move(unit, best_table)
            else:
                partner_table = int(assignment[best_partner])
                move(best_partner, current)
                move(unit, partner_table)

        if not improved:
            break

    return assignment


def _build_result(guests, guest_tables, similarity, avoid_pairs, seats_per_table):
    tables = []
    occupied = sorted(set(guest_tables.tolist()))
    for number, table in enumerate(occupied, start=1):
        members = np.flatnonzero(guest_tables == table).tolist()
        tables.append({
            'table': number,
            'average_similarity': _rounded_table_similarity(similarity, members),
            'guests': [_describe_seat(guests, guest, members, similarity) for guest in members],
        })

    conflicts = sum(guest_tables[a] == guest_tables[b] for a, b in avoid_pairs)
    return {
        'tables': tables,
        'summary': {
            'guest_count': len(guests),
            'tables_used': len(tables),
            'seats_per_table': seats_per_table,
            'average_tablemate_similarity': _average_tablemate_similarity(similarity, guest_tables),
            'random_seating_similarity': round(_average_pair_similarity(similarity), 3),
            'silhouette_score': _silhouette(similarity, guest_tables),
            'avoid_requests': len(avoid_pairs),
            'avoid_conflicts': int(conflicts),
        },
    }


def _describe_seat(guests, guest, members, similarity):
    tablemates = [member for member in members if member != guest]
    seat = {'name': guests[guest]['name'], 'closest_tablemate': None}
    if tablemates:
        closest = max(tablemates, key=lambda member: similarity[guest, member])
        seat['closest_tablemate'] = {
            'name': guests[closest]['name'],
            'similarity': round(float(similarity[guest, closest]), 3),
        }
    return seat


def _rounded_table_similarity(similarity, members):
    if len(members) < 2:
        return None
    return round(_average_pair_similarity(similarity[np.ix_(members, members)]), 3)


def _average_tablemate_similarity(similarity, guest_tables):
    same_table = guest_tables[:, None] == guest_tables[None, :]
    np.fill_diagonal(same_table, False)
    if not same_table.any():
        return None
    return round(float(similarity[same_table].mean()), 3)


def _silhouette(similarity, guest_tables):
    """How well separated the tables are, from -1 to 1. None if it can't be calculated."""
    table_count = len(set(guest_tables.tolist()))
    if not 2 <= table_count < len(guest_tables):
        return None
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    score = float(silhouette_score(distance, guest_tables, metric='precomputed'))
    return round(score, 3) if math.isfinite(score) else None
