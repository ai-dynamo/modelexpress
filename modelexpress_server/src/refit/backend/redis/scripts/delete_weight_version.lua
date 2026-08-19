-- Atomically mark a READY WeightVersion as logically deleted.
--
-- KEYS[1]: version hash
-- ARGV[1]: RELEASING state value
-- ARGV[2]: READY state value

if redis.call('EXISTS', KEYS[1]) == 0 then
  return 'VERSION_NOT_FOUND'
end

local state = redis.call('HGET', KEYS[1], 'state')
if state == ARGV[1] then
  return 'OK'
end
if state ~= ARGV[2] then
  return 'VERSION_NOT_READY'
end

redis.call('HSET', KEYS[1], 'state', ARGV[1])
return 'OK'
