-- Delete a mesh only after linked versions and their resources retire.
-- KEYS: mesh hash, version-reference set
-- ARGV: releasing state
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'NOT_FOUND'
end
local clock = redis.call('TIME')
local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
for _, version_id in ipairs(redis.call('SMEMBERS', KEYS[2])) do
  local version_key = 'mx:refit:version:metadata:' .. version_id
  local shards_key = 'mx:refit:version:shards:' .. version_id
  local leases_key = 'mx:refit:version:leases:' .. version_id
  redis.call('ZREMRANGEBYSCORE', leases_key, '-inf', now)
  if redis.call('HGET', version_key, 'state') ~= ARGV[1]
      or redis.call('HLEN', shards_key) > 0
      or redis.call('ZCARD', leases_key) > 0 then
    return 'IN_USE'
  end
end
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
return 'DELETED'
