-- Atomically publish one worker manifest and advance version readiness.
--
-- KEYS[1]: version hash
-- KEYS[2]: publishing worker registration hash
-- KEYS[3]: physical WeightVersionShard publications, keyed by the private
--          worker_id + logical_shard_id identity
-- KEYS[4]: trainer mesh hash
-- KEYS[5]: publication endpoint index, keyed like physical publications
-- ARGV: publication key, encoded shard, model_name, logical_shard_id, staging_state,
--       ready_state, worker_id, trainer_role, manifest_endpoint
--
-- Returns OK:<state> for a new or byte-identical repeated publication. Other
-- named results reject missing, incompatible, or conflicting inputs. The
-- publication, logical-shard coverage update, and final READY transition are one
-- Redis operation, so observers cannot see partial readiness.

if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'VERSION_NOT_FOUND'
end
if redis.call('EXISTS', KEYS[2]) == 0 then
  return 'WORKER_NOT_FOUND'
end

if redis.call('HGET', KEYS[2], 'model_name') ~= ARGV[3] then
  return 'MODEL_MISMATCH'
end

local state = redis.call('HGET', KEYS[1], 'state')
if state ~= ARGV[5] and state ~= ARGV[6] then
  return 'VERSION_NOT_WRITABLE'
end
local s3_uri = redis.call('HGET', KEYS[1], 's3_uri')
if s3_uri and s3_uri ~= '' then
  return 'VERSION_NOT_WRITABLE'
end
if redis.call('EXISTS', KEYS[4]) == 0 then
  return 'MESH_NOT_FOUND'
end
local generation = redis.call('HGET', KEYS[1], 'trainer_mesh_generation')
if not generation or generation == '0' then
  return 'MESH_GENERATION_MISSING'
end
if generation ~= redis.call('HGET', KEYS[4], 'generation') then
  return 'MESH_GENERATION_MISMATCH'
end
if tonumber(redis.call('HGET', KEYS[2], 'role')) ~= tonumber(ARGV[8]) then
  return 'WORKER_NOT_TRAINER'
end
local workers = cjson.decode(redis.call('HGET', KEYS[4], 'workers'))
local metadata = workers[ARGV[7]]
if not metadata or metadata.logical_shard_id ~= ARGV[4] then
  return 'WORKER_NOT_IN_MESH'
end
if metadata.metadata_endpoint ~= ARGV[9]
    or redis.call('HGET', KEYS[2], 'refit_endpoint') ~= ARGV[9] then
  return 'WORKER_ENDPOINT_MISMATCH'
end
local expected = #cjson.decode(redis.call('HGET', KEYS[4], 'logical_shards'))
local existing = redis.call('HGET', KEYS[3], ARGV[1])
if existing and existing ~= ARGV[2] then
  return 'SHARD_CONFLICT'
end

redis.call('HSET', KEYS[3], ARGV[1], ARGV[2])
redis.call('HSET', KEYS[5], ARGV[1], ARGV[9])
local present = {}
local covered = 0
for worker_id, metadata in pairs(workers) do
  local key = string.len(worker_id) .. ':' .. worker_id .. metadata.logical_shard_id
  local registration = 'mx:refit:worker:' .. worker_id
  if redis.call('HEXISTS', KEYS[3], key) == 1
      and redis.call('HGET', KEYS[5], key) == metadata.metadata_endpoint
      and redis.call('HGET', registration, 'refit_endpoint') == metadata.metadata_endpoint
      and redis.call('HGET', registration, 'model_name') == ARGV[3]
      and redis.call('HGET', registration, 'role') == ARGV[8]
      and not present[metadata.logical_shard_id] then
    present[metadata.logical_shard_id] = true
    covered = covered + 1
  end
end
if covered == expected then
  state = ARGV[6]
  redis.call('HSET', KEYS[1], 'state', state)
end

return 'OK:' .. state
