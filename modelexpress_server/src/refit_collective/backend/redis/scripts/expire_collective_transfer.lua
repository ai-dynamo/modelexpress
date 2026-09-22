-- Atomically abort one elapsed collective transfer and release its idempotency
-- reservation. Every lifecycle operation invokes this before it observes or
-- mutates an operation, so a dead rank cannot pin a nonterminal operation.
--
-- KEYS[1]: operation hash
-- KEYS[2]: reported worker set
-- KEYS[3]: create-request idempotency key
-- KEYS[4]: operation's group hash
-- ARGV: operation_id, group_id, deadline_message
--
-- Returns:
--   ACTIVE       operation is live or already terminal
--   ABORTED      PENDING/RUNNING operation expired and was fenced
--   NOTFOUND     operation no longer exists

local state = redis.call('HGET', KEYS[1], 'state')
if not state then
  return 'NOTFOUND'
end
if state == 'COMPLETE' or state == 'FAILED' or state == 'ABORTED' then
  return 'ACTIVE'
end

local deadline = tonumber(redis.call('HGET', KEYS[1], 'deadline_unix_ms'))
local clock = redis.call('TIME')
local now_ms = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
if deadline and deadline > now_ms then
  return 'ACTIVE'
end

-- A pre-deadline record is malformed. Fencing it is safer than permitting a
-- legacy/incomplete operation to remain nonterminal forever.
redis.call('HSET', KEYS[1],
  'state', 'ABORTED',
  'failure_message', ARGV[3])

local operation_group = redis.call('HGET', KEYS[1], 'group_id')
local operation_epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
local group_epoch = tonumber(redis.call('HGET', KEYS[4], 'epoch'))
if operation_group == ARGV[2] and operation_epoch and group_epoch
    and operation_epoch == group_epoch then
  redis.call('HSET', KEYS[4],
    'epoch', group_epoch + 1,
    'state', 'FORMING',
    'bootstrap_complete_epoch', 0,
    'active_operation_id', '',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '')

  local lanes = redis.call('HGET', KEYS[4], 'lanes')
  if lanes then
    for line in string.gmatch(lanes .. '\n', '([^\n]*)\n') do
      local lane_id = string.match(line, '^([^|]*)|')
      if lane_id then
        redis.call('DEL', KEYS[4] .. ':lane:' .. lane_id)
        redis.call('DEL', KEYS[4] .. ':fence:' .. lane_id)
      end
    end
  end
end

if redis.call('GET', KEYS[3]) == ARGV[1] then
  redis.call('DEL', KEYS[3])
end
return 'ABORTED'
