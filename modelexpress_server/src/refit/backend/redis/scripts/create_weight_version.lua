-- Atomically reserve an idempotency key and create one immutable WeightVersion.
--
-- KEYS[1]: version hash
-- KEYS[2]: create-request idempotency key
-- KEYS[3]: trainer mesh hash (worker-sharded versions only)
-- KEYS[4]: trainer mesh version-reference set
-- ARGV: uid, model_name, idempotency_key, payload_format,
--       base_version_id,
--       s3_uri, initial_state, state, created_at_unix_ms, trainer_mesh_id, version_number
--
-- Returns:
--   CREATED              this invocation created the version
--   EXISTING:<id>        another invocation already owns the idempotency key
--   COLLISION            the selected version ID already exists
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

local generation = '0'
if ARGV[10] ~= '' then
  if redis.call('EXISTS', KEYS[3]) == 0 then
    return 'MESH_NOT_FOUND'
  end
  if redis.call('HGET', KEYS[3], 'model_name') ~= ARGV[2] then
    return 'MESH_MODEL_MISMATCH'
  end
  generation = redis.call('HGET', KEYS[3], 'generation')
  if not generation or generation == '0' then
    return 'MESH_GENERATION_MISSING'
  end
end

redis.call('HSET', KEYS[1],
  'uid', ARGV[1],
  'model_name', ARGV[2],
  'idempotency_key', ARGV[3],
  'payload_format', ARGV[4],
  'base_version_id', ARGV[5],
  's3_uri', ARGV[6],
  'initial_state', ARGV[7],
  'layout_signature', '',
  'state', ARGV[8],
  'created_at_unix_ms', ARGV[9],
  'trainer_mesh_id', ARGV[10],
  'trainer_mesh_generation', generation,
  'version_number', ARGV[11])
if ARGV[10] ~= '' then
  redis.call('SADD', KEYS[4], ARGV[1])
end
redis.call('SET', KEYS[2], ARGV[1])

return 'CREATED'
