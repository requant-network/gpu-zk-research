"""Public-input sumcheck baseline over BabyBear. Python standard library only."""
import hashlib
import random
import struct
import time

P = 2013265921


def fold(values, challenge):
    return [(left + challenge * (right - left)) % P
            for left, right in zip(values[::2], values[1::2])]


def commitment(a, b):
    payload = struct.pack('<I', len(a))
    for values in (a, b):
        payload += struct.pack(f'<{len(values)}I', *values)
    return hashlib.sha256(b'requant-sumcheck-v1' + payload).digest()


def challenge(transcript, coefficients):
    transcript = hashlib.sha256(transcript + struct.pack('<3I', *coefficients)).digest()
    return transcript, int.from_bytes(transcript, 'little') % P


def prove(a, b):
    transcript = commitment(a, b)
    rounds = []
    while len(a) > 1:
        c0 = c1 = c2 = 0
        for a0, a1, b0, b1 in zip(a[::2], a[1::2], b[::2], b[1::2]):
            da, db = (a1 - a0) % P, (b1 - b0) % P
            c0 += a0 * b0
            c1 += a0 * db + b0 * da
            c2 += da * db
        coefficients = [c0 % P, c1 % P, c2 % P]
        rounds.append(coefficients)
        transcript, r = challenge(transcript, coefficients)
        a, b = fold(a, r), fold(b, r)
    return rounds


def verify(a, b, claimed_sum, rounds):
    if not a or len(a) != len(b) or len(a) & (len(a) - 1):
        return False
    if len(rounds) != len(a).bit_length() - 1:
        return False
    transcript = commitment(a, b)
    for coefficients in rounds:
        if len(coefficients) != 3 or any(c < 0 or c >= P for c in coefficients):
            return False
        c0, c1, c2 = coefficients
        if (2 * c0 + c1 + c2) % P != claimed_sum:
            return False
        transcript, r = challenge(transcript, coefficients)
        claimed_sum = (c0 + r * (c1 + r * c2)) % P
        a, b = fold(a, r), fold(b, r)
    return claimed_sum == a[0] * b[0] % P


def experiment():
    rng = random.Random(20261109)
    cases = []
    rejected = 0
    for n in (2, 16, 256, 4096, 16384):
        for pattern in ('zero', 'max', 'random'):
            a = [0 if pattern == 'zero' else P - 1 if pattern == 'max'
                 else rng.randrange(P) for _ in range(n)]
            b = [rng.randrange(P) for _ in range(n)]
            total = sum(x * y for x, y in zip(a, b)) % P
            start = time.perf_counter()
            proof = prove(a, b)
            elapsed = time.perf_counter() - start
            assert verify(a, b, total, proof)
            altered = [row[:] for row in proof]
            altered[0][0] = (altered[0][0] + 1) % P
            assert not verify(a, b, total, altered)
            assert not verify(a, b, (total + 1) % P, proof)
            assert not verify(a, b, total, proof[:-1])
            rejected += 3
            cases.append({'n': n, 'pattern': pattern, 'prover_seconds': elapsed,
                          'rounds': len(proof), 'verified': True})
    return {'cases': cases, 'honest_proofs_verified': len(cases),
            'tampered_cases_rejected': rejected,
            'scope': 'Public-input algebraic baseline. Final MLE is recomputed; no hiding or succinct commitment.'}


def fixture():
    rng = random.Random(7)
    a = [rng.randrange(P) for _ in range(4096)]
    b = [rng.randrange(P) for _ in range(4096)]
    r = P - 7
    expected = [(x + r * (y - x)) % P for x, y in zip(a, b)]
    return r, P, a, b, expected
