-- Atomically register or refresh one TTL-bound worker process.
--
-- KEYS[1]: expiring worker registration hash
-- ARGV: worker_id, role, model_name, ttl_milliseconds
--
-- Returns:
--   OK:<expiry_ms>       registration stored and TTL refreshed
--   CONFLICT             the worker ID has different immutable metadata

if redis.call('EXISTS', KEYS[1]) == 1 then
  local same_registration =
    redis.call('HGET', KEYS[1], 'role') == ARGV[2]
    and redis.call('HGET', KEYS[1], 'model_name') == ARGV[3]
  if not same_registration then
    return 'CONFLICT'
  end
end

-- Redis time, rather than an MX server clock, makes expiry consistent across replicas.
local clock = redis.call('TIME')
local expires_at = clock[1] * 1000 + math.floor(clock[2] / 1000) + tonumber(ARGV[4])

redis.call('HSET', KEYS[1],
  'worker_id', ARGV[1],
  'role', ARGV[2],
  'model_name', ARGV[3],
  'expires_at_unix_ms', expires_at)
redis.call('PEXPIRE', KEYS[1], ARGV[4])

return 'OK:' .. expires_at
