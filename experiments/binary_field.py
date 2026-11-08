"""Independent GF(2^128) arithmetic in little-endian polynomial coordinates."""
import hashlib
import random
import struct

MODULUS = (1 << 128) | 0x87
MASK = (1 << 128) - 1


def multiply(a, b):
    product = 0
    while b:
        bit = b & -b
        product ^= a << (bit.bit_length() - 1)
        b ^= bit
    while product.bit_length() > 128:
        product ^= MODULUS << (product.bit_length() - 129)
    return product


def bind(a, b):
    h = hashlib.sha256(b'requant-binary-sumcheck-v1' + struct.pack('<I', len(a)))
    for values in (a, b):
        for x in values:
            h.update(x.to_bytes(16, 'little'))
    return h.digest()


def challenge(digest, coefficients):
    digest = hashlib.sha256(digest + b''.join(x.to_bytes(16, 'little') for x in coefficients)).digest()
    return digest, int.from_bytes(digest[:16], 'little')


def prove(a, b):
    digest = bind(a, b)
    rounds = []
    while len(a) > 1:
        c0 = c1 = c2 = 0
        for a0, a1, b0, b1 in zip(a[::2], a[1::2], b[::2], b[1::2]):
            da, db = a0 ^ a1, b0 ^ b1
            c0 ^= multiply(a0, b0)
            c1 ^= multiply(a0, db) ^ multiply(b0, da)
            c2 ^= multiply(da, db)
        rounds.append([c0, c1, c2])
        digest, r = challenge(digest, rounds[-1])
        a = [x ^ multiply(r, x ^ y) for x, y in zip(a[::2], a[1::2])]
        b = [x ^ multiply(r, x ^ y) for x, y in zip(b[::2], b[1::2])]
    return rounds


def verify(a, b, claim, rounds):
    if not a or len(a) != len(b) or len(a) & (len(a)-1) or len(rounds) != len(a).bit_length()-1:
        return False
    digest = bind(a, b)
    for cs in rounds:
        if len(cs) != 3 or any(x < 0 or x > MASK for x in cs):
            return False
        c0, c1, c2 = cs
        if c1 ^ c2 != claim:
            return False
        digest, r = challenge(digest, cs)
        claim = c0 ^ multiply(r, c1 ^ multiply(r, c2))
        a = [x ^ multiply(r, x ^ y) for x, y in zip(a[::2], a[1::2])]
        b = [x ^ multiply(r, x ^ y) for x, y in zip(b[::2], b[1::2])]
    return claim == multiply(a[0], b[0])


def self_check():
    rng = random.Random(8675309)
    # Rabin irreducibility test; 128 has only the prime divisor 2.
    power = 2
    for iteration in range(1, 129):
        power = multiply(power, power)
        if iteration == 64:
            a, b = MODULUS, power ^ 2
            while b:
                remainder = a
                while remainder and remainder.bit_length() >= b.bit_length():
                    remainder ^= b << (remainder.bit_length()-b.bit_length())
                a, b = b, remainder
            assert a == 1
    assert power == 2
    assert multiply(1 << 127, 2) == 0x87
    for _ in range(100):
        a, b, c = (rng.getrandbits(128) for _ in range(3))
        assert multiply(a, b) == multiply(b, a)
        assert multiply(a, b ^ c) == multiply(a, b) ^ multiply(a, c)
        assert multiply(a, 1) == a
    for n in (2, 16, 128):
        a, b = ([rng.getrandbits(128) for _ in range(n)] for _ in range(2))
        claim = 0
        for x, y in zip(a, b):
            claim ^= multiply(x, y)
        proof = prove(a, b)
        assert verify(a, b, claim, proof)
        assert not verify(a, b, claim ^ 1, proof)
        assert not verify(a, b, claim, proof[:-1])
        damaged = [x[:] for x in proof]
        damaged[0][0] ^= 1
        assert not verify(a, b, claim, damaged)
    print('Binary-field identities and honest/tampered sumchecks passed')


if __name__ == '__main__':
    assert __debug__
    self_check()
