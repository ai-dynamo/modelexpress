-- Delete a mesh and its versions once every version has retired: released,
-- unpublished by its sources and read by no live lease.
-- KEYS: mesh hash, version-reference set
-- ARGV: releasing state
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'NOT_FOUND'
end
local clock = redis.call('TIME')
local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
local version_ids = redis.call('SMEMBERS', KEYS[2])
for _, version_id in ipairs(version_ids) do
  if not version_retired(version_id, ARGV[1], now) then
    return 'IN_USE'
  end
end
for _, version_id in ipairs(version_ids) do
  delete_version(version_id)
end
redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
return 'DELETED'
