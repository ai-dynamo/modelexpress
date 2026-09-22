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

local s3_uri = redis.call('HGET', KEYS[1], 's3_uri')
local s3_ready = current == ARGV[2]
  and ARGV[1] == ARGV[3]
  and s3_uri
  and s3_uri ~= ''
local releasing = (current == ARGV[2] or current == ARGV[3])
  and ARGV[1] == ARGV[4]
if not s3_ready and not releasing then
  return 'INVALID_TRANSITION'
end

redis.call('HSET', KEYS[1], 'state', ARGV[1])
return 'OK'
