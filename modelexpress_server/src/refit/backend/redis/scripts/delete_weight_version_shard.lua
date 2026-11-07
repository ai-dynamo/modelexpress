-- Atomically delete one source shard when no active reader lease protects it.
--
-- KEYS[1]: version hash
-- KEYS[2]: source worker registration hash
-- KEYS[3]: physical shards hash
-- KEYS[4]: active lease expiry sorted set for the version
-- KEYS[5]: publication endpoint index
-- KEYS[6]: trainer reader lease expiry sorted set for the version
-- ARGV: publication key, encoded shard, version_id,
--       wait_for_trainer_readers_only ("1" or "0")
--
-- Expired leases are removed using Redis time. Any remaining lease protects
-- every shard of the version, independent of which source was selected, unless
-- the caller waits only for trainer readers: then only a lease still marked as
-- reading the trainer protects the shards.

if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'VERSION_NOT_FOUND'
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
local readers = KEYS[4]
if ARGV[4] == '1' then
  readers = KEYS[6]
  redis.call('ZREMRANGEBYSCORE', readers, '-inf', now)
  for _, lease_id in ipairs(redis.call('ZRANGE', readers, 0, -1)) do
    if redis.call('EXISTS', 'mx:refit:version:lease:' .. ARGV[3] .. ':' .. lease_id) == 0 then
      redis.call('ZREM', readers, lease_id)
    end
  end
end
if redis.call('ZCARD', readers) > 0 then
  return 'VERSION_LEASED'
end

redis.call('HDEL', KEYS[3], ARGV[1])
redis.call('HDEL', KEYS[5], ARGV[1])
return 'DELETED'
