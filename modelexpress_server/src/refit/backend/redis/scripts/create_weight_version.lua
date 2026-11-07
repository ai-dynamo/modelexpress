-- Atomically reserve an idempotency key and create one immutable WeightVersion.
--
-- KEYS[1]: version hash
-- KEYS[2]: create-request idempotency key
-- KEYS[3]: trainer mesh hash (worker-sharded versions only)
-- KEYS[4]: trainer mesh version-reference set
-- ARGV: uid, model_name, idempotency_key, payload_format,
--       base_version_id,
--       object_storage_uri, initial_state, state, created_at_unix_ms, trainer_mesh_id,
--       version_number, object_storage_type, s3_uri, releasing_state,
--       max_trainer_readers
--
-- Returns:
--   CREATED              this invocation created the version
--   EXISTING:<id>        another invocation already owns the idempotency key
--   COLLISION            the selected version ID already exists
--
-- Redis executes the script atomically, so two MX server replicas cannot create
-- different versions for the same idempotency key. Creating a mesh-backed
-- version also deletes the mesh's retired versions, so a trainer that keeps
-- publishing does not grow the version list readers poll.

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
  'object_storage_uri', ARGV[6],
  'initial_state', ARGV[7],
  'layout_signature', '',
  'state', ARGV[8],
  'created_at_unix_ms', ARGV[9],
  'trainer_mesh_id', ARGV[10],
  'trainer_mesh_generation', generation,
  'version_number', ARGV[11],
  'object_storage_type', ARGV[12],
  's3_uri', ARGV[13],
  'max_trainer_readers', ARGV[15])
if ARGV[10] ~= '' then
  local clock = redis.call('TIME')
  local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
  for _, version_id in ipairs(redis.call('SMEMBERS', KEYS[4])) do
    if version_retired(version_id, ARGV[14], now) then
      delete_version(version_id)
      redis.call('SREM', KEYS[4], version_id)
    end
  end
  redis.call('SADD', KEYS[4], ARGV[1])
end
redis.call('SET', KEYS[2], ARGV[1])

return 'CREATED'
