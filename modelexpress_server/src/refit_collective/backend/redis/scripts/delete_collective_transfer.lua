-- Delete only a terminal transfer and release its idempotency reservation if
-- this operation still owns it.
--
-- KEYS[1]: operation hash
-- KEYS[2]: reported worker set
-- KEYS[3]: create-request idempotency key
-- ARGV[1]: operation_id

local state = redis.call('HGET', KEYS[1], 'state')
if not state then
  return 'NOTFOUND'
end
if state ~= 'COMPLETE' and state ~= 'FAILED' and state ~= 'ABORTED' then
  return 'NOTTERMINAL'
end

redis.call('DEL', KEYS[1])
redis.call('DEL', KEYS[2])
if redis.call('GET', KEYS[3]) == ARGV[1] then
  redis.call('DEL', KEYS[3])
end
return 'DELETED'
