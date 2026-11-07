-- Helpers that remove retired mesh-backed versions. Redis EVAL has no
-- `require`, so redis.rs prepends this chunk to each script that uses it.

local function version_retired(version_id, releasing_state, now)
  local version_key = 'mx:refit:version:metadata:' .. version_id
  local leases_key = 'mx:refit:version:leases:' .. version_id
  if redis.call('EXISTS', version_key) == 0 then
    return true
  end
  redis.call('ZREMRANGEBYSCORE', leases_key, '-inf', now)
  return redis.call('HGET', version_key, 'state') == releasing_state
    and redis.call('HLEN', 'mx:refit:version:shards:' .. version_id) == 0
    and redis.call('ZCARD', leases_key) == 0
end

local function delete_version(version_id)
  redis.call('DEL',
    'mx:refit:version:metadata:' .. version_id,
    'mx:refit:version:shards:' .. version_id,
    'mx:refit:version:leases:' .. version_id,
    'mx:refit:version:publication-endpoints:' .. version_id,
    'mx:refit:version:trainer-readers:' .. version_id)
end
