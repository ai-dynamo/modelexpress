-- Atomically delete one source shard after logical version release.
--
-- KEYS[1]: version hash
-- KEYS[2]: source worker registration hash
-- KEYS[3]: physical shards hash
-- KEYS[4]: active lease expiry sorted set for the version
-- ARGV: publication key, encoded shard, releasing_state
--
-- Expired leases are removed using Redis time. Any remaining lease protects
-- every shard of the version, independent of which source was selected.

if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'VERSION_NOT_FOUND'
end
if redis.call('HGET', KEYS[1], 'state') ~= ARGV[3] then
  return 'VERSION_NOT_RELEASING'
end
if redis.call('EXISTS', KEYS[2]) == 0 then
  return 'WORKER_NOT_FOUND'
end
local current = redis.call('HGET', KEYS[3], ARGV[1])
if not current then
  return 'SHARD_NOT_FOUND'
end
if current ~= ARGV[2] then
  return 'SHARD_CONFLICT'
end

local clock = redis.call('TIME')
local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', now)
if redis.call('ZCARD', KEYS[4]) > 0 then
  return 'VERSION_LEASED'
end

redis.call('HDEL', KEYS[3], ARGV[1])
return 'DELETED'
