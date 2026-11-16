-- Atomically advance one WeightVersion lifecycle state.
--
-- KEYS[1]: version hash
-- ARGV: target_state, staging_state, ready_state, releasing_state

if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'VERSION_NOT_FOUND'
end

local current = redis.call('HGET', KEYS[1], 'state')
if current == ARGV[1] then
  return 'OK'
end

local uri = redis.call('HGET', KEYS[1], 'object_storage_uri')
if not uri then
  uri = redis.call('HGET', KEYS[1], 's3_uri')
end
local object_storage_ready = current == ARGV[2]
  and ARGV[1] == ARGV[3]
  and uri
  and uri ~= ''
local releasing = (current == ARGV[2] or current == ARGV[3])
  and ARGV[1] == ARGV[4]
if not object_storage_ready and not releasing then
  return 'INVALID_TRANSITION'
end

redis.call('HSET', KEYS[1], 'state', ARGV[1])
return 'OK'
