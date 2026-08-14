-- Atomically acquire or renew the one lease owned by a worker for a version.
--
-- KEYS[1]: version hash
-- KEYS[2]: consumer worker registration hash
-- KEYS[3]: deterministic lease hash
-- KEYS[4]: active lease expiry sorted set for the version
-- ARGV: lease_id, version_id, worker_id, ttl_milliseconds, ready_state,
--       releasing_state, generator_role
--
-- A new lease is accepted only while the version is READY. The same owner may
-- renew an existing lease while the version is READY or RELEASING, allowing an
-- in-flight update to finish after logical release begins.
--
-- Returns OK:<expiry_ms> for a successful acquisition or renewal. Named
-- results reject missing, incompatible, or no-longer-leaseable inputs.

if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'VERSION_NOT_FOUND'
end
if redis.call('EXISTS', KEYS[2]) == 0 then
  return 'WORKER_NOT_FOUND'
end
if redis.call('HGET', KEYS[2], 'role') ~= ARGV[7] then
  return 'WORKER_NOT_GENERATOR'
end
if redis.call('HGET', KEYS[2], 'model_name') ~= redis.call('HGET', KEYS[1], 'model_name') then
  return 'MODEL_MISMATCH'
end

local state = redis.call('HGET', KEYS[1], 'state')
local existing = redis.call('EXISTS', KEYS[3]) == 1
if existing then
  local same_lease =
    redis.call('HGET', KEYS[3], 'lease_id') == ARGV[1]
    and redis.call('HGET', KEYS[3], 'version_id') == ARGV[2]
    and redis.call('HGET', KEYS[3], 'worker_id') == ARGV[3]
  if not same_lease then
    return 'LEASE_CONFLICT'
  end
  if state ~= ARGV[5] and state ~= ARGV[6] then
    return 'VERSION_NOT_LEASEABLE'
  end
elseif state ~= ARGV[5] then
  return 'VERSION_NOT_LEASEABLE'
end

local clock = redis.call('TIME')
local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
local expires_at = now + tonumber(ARGV[4])

redis.call('HSET', KEYS[3],
  'lease_id', ARGV[1],
  'version_id', ARGV[2],
  'worker_id', ARGV[3],
  'expires_at_unix_ms', expires_at)
redis.call('PEXPIRE', KEYS[3], ARGV[4])
redis.call('ZADD', KEYS[4], expires_at, ARGV[1])

return 'OK:' .. expires_at
