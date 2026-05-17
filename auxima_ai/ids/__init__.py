"""Identifier primitives — lex-sortable + URL-safe ids for the sidecar.

:mod:`.ulid` implements ULID (Universally Unique Lexicographically
Sortable Identifier) per https://github.com/ulid/spec. ULIDs are used
for outbound webhook delivery rows, activity-log primary keys, and
any other surface where a 16-byte v4 UUID would lose insertion-order
locality (UUIDv4 randomness destroys b-tree clustering at scale).
"""
