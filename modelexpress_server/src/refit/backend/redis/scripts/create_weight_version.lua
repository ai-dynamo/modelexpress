-- Atomically reserve an idempotency key and create one immutable WeightVersion.
--
-- KEYS[1]: version hash
-- KEYS[2]: create-request idempotency key
-- KEYS[3]: expected source slots set
-- ARGV: uid, model_name, version_number, idempotency_key, payload_format,
--       base_version_id, expected_source_slots JSON, expected_source_slot_count,
--       s3_uri, initial_state, state, created_at_unix_ms
--
-- Returns:
--   CREATED              this invocation created the version
--   EXISTING:<id>        another invocation already owns the idempotency key
--   COLLISION            the randomly selected version ID already exists
--
-- Redis executes the script atomically, so two MX server replicas cannot create
-- different versions for the same idempotency key.

local existing = redis.call('GET', KEYS[2])
if existing then
  return 'EXISTING:' .. existing
end

if redis.call('EXISTS', KEYS[1]) == 1 then
  return 'COLLISION'
end

redis.call('HSET', KEYS[1],
  'uid', ARGV[1],
  'model_name', ARGV[2],
  'version_number', ARGV[3],
  'idempotency_key', ARGV[4],
  'payload_format', ARGV[5],
  'base_version_id', ARGV[6],
  'expected_source_slots', ARGV[7],
  'expected_source_slot_count', ARGV[8],
  's3_uri', ARGV[9],
  'initial_state', ARGV[10],
  'layout_signature', '',
  'state', ARGV[11],
  'created_at_unix_ms', ARGV[12])
for index = 13, #ARGV do
  redis.call('SADD', KEYS[3], ARGV[index])
end
redis.call('SET', KEYS[2], ARGV[1])

return 'CREATED'
