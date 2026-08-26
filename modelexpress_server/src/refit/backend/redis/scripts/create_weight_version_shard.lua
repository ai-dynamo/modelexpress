-- Atomically publish one worker manifest and advance version readiness.
--
-- KEYS[1]: version hash
-- KEYS[2]: publishing worker registration hash
-- KEYS[3]: physical WeightVersionShard publications, keyed by the private
--          worker_id + source_slot_id identity
-- KEYS[4]: covered version-scoped source slots set
-- KEYS[5]: expected version-scoped source slots set
-- ARGV: publication key, encoded shard, model_name, source_slot_id, staging_state,
--       ready_state
--
-- Returns OK:<state> for a new or byte-identical repeated publication. Other
-- named results reject missing, incompatible, or conflicting inputs. The
-- publication, source-slot coverage update, and final READY transition are one
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
if redis.call('SISMEMBER', KEYS[5], ARGV[4]) == 0 then
  return 'SOURCE_SLOT_NOT_REQUIRED'
end

local existing = redis.call('HGET', KEYS[3], ARGV[1])
if existing then
  if existing == ARGV[2] then
    return 'OK:' .. state
  end
  return 'SHARD_CONFLICT'
end

redis.call('HSET', KEYS[3], ARGV[1], ARGV[2])
redis.call('SADD', KEYS[4], ARGV[4])

local covered = redis.call('SCARD', KEYS[4])
local expected = tonumber(redis.call('HGET', KEYS[1], 'expected_source_slot_count'))
if covered == expected then
  state = ARGV[6]
  redis.call('HSET', KEYS[1], 'state', state)
end

return 'OK:' .. state
