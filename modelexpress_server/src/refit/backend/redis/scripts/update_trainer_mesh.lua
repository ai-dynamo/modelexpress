-- Compare-and-swap membership after checking all selected workers.
-- KEYS[1]: mesh hash; KEYS[2]: linked version IDs; KEYS[3..]: worker registrations
-- ARGV: expected_generation, workers JSON, model_name, trainer_role
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'NOT_FOUND'
end
local old = cjson.decode(redis.call('HGET', KEYS[1], 'workers'))
local replacement = cjson.decode(ARGV[2])
local expected = {}
local seen = {}
local unchanged = true
for worker_id, metadata in pairs(old) do
  expected[metadata.logical_shard_id] = true
  if not replacement[worker_id]
      or replacement[worker_id].logical_shard_id ~= metadata.logical_shard_id
      or replacement[worker_id].metadata_endpoint ~= metadata.metadata_endpoint then
    unchanged = false
  end
end
for worker_id, metadata in pairs(replacement) do
  if not expected[metadata.logical_shard_id] then
    return 'COVERAGE_MISMATCH'
  end
  seen[metadata.logical_shard_id] = true
  if not old[worker_id] then
    unchanged = false
  end
end
for shard_id, _ in pairs(expected) do
  if not seen[shard_id] then
    return 'COVERAGE_MISMATCH'
  end
end
if tonumber(redis.call('HGET', KEYS[1], 'generation')) ~= tonumber(ARGV[1]) then
  return 'GENERATION_MISMATCH'
end
for index = 3, #KEYS do
  if redis.call('EXISTS', KEYS[index]) == 0 then
    return 'WORKER_NOT_FOUND'
  end
  if redis.call('HGET', KEYS[index], 'model_name') ~= ARGV[3]
      or tonumber(redis.call('HGET', KEYS[index], 'role')) ~= tonumber(ARGV[4]) then
    return 'WORKER_MISMATCH'
  end
  local worker_id = redis.call('HGET', KEYS[index], 'worker_id')
  if not replacement[worker_id]
      or redis.call('HGET', KEYS[index], 'refit_endpoint') ~= replacement[worker_id].metadata_endpoint then
    return 'WORKER_MISMATCH'
  end
end
if unchanged then
  return 'UNCHANGED'
end
local retired = {}
local changed = {}
for worker_id, metadata in pairs(replacement) do
  local previous = old[worker_id]
  if not previous or previous.logical_shard_id ~= metadata.logical_shard_id
      or previous.metadata_endpoint ~= metadata.metadata_endpoint then
    changed[worker_id] = metadata
  end
end
local removed = {}
for worker_id, metadata in pairs(old) do
  local current = replacement[worker_id]
  if not current or current.logical_shard_id ~= metadata.logical_shard_id then
    removed[worker_id] = metadata
  end
end
local clock = redis.call('TIME')
local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
for _, version_id in ipairs(redis.call('SMEMBERS', KEYS[2])) do
  local shards = 'mx:refit:version:shards:' .. version_id
  local endpoints = 'mx:refit:version:publication-endpoints:' .. version_id
  local leases = 'mx:refit:version:leases:' .. version_id
  redis.call('ZREMRANGEBYSCORE', leases, '-inf', now)
  if redis.call('ZCARD', leases) > 0 then
    return 'VERSION_LEASED'
  end
  for worker_id, metadata in pairs(removed) do
    local key = string.len(worker_id) .. ':' .. worker_id .. metadata.logical_shard_id
    if redis.call('HEXISTS', shards, key) == 1 then
      table.insert(retired, {shards, endpoints, key})
    end
  end
  for worker_id, metadata in pairs(changed) do
    local key = string.len(worker_id) .. ':' .. worker_id .. metadata.logical_shard_id
    if redis.call('HEXISTS', shards, key) == 1
        and redis.call('HGET', endpoints, key) ~= metadata.metadata_endpoint then
      table.insert(retired, {shards, endpoints, key})
    end
  end
end
for _, publication in ipairs(retired) do
  redis.call('HDEL', publication[1], publication[3])
  redis.call('HDEL', publication[2], publication[3])
end
redis.call('HSET', KEYS[1],
  'workers', ARGV[2],
  'generation', tonumber(ARGV[1]) + 1)
return 'UPDATED'
