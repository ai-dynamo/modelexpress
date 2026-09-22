-- Atomically release one consumer lease.
--
-- KEYS[1]: lease hash
-- KEYS[2]: active lease expiry sorted set for the version
-- ARGV: lease_id, version_id, worker_id

if redis.call('EXISTS', KEYS[1]) == 0 then
  redis.call('ZREM', KEYS[2], ARGV[1])
  return 'NOT_FOUND'
end

local same_lease =
  redis.call('HGET', KEYS[1], 'lease_id') == ARGV[1]
  and redis.call('HGET', KEYS[1], 'version_id') == ARGV[2]
  and redis.call('HGET', KEYS[1], 'worker_id') == ARGV[3]
if not same_lease then
  return 'LEASE_CONFLICT'
end

redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return 'DELETED'
