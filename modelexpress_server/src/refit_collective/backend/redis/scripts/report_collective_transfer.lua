-- Atomically record one participant's terminal result for a transfer.
--
-- KEYS[1]: operation hash
-- KEYS[2]: reported worker set
-- KEYS[3]: group hash
-- KEYS[4]: participants hash
-- ARGV: operation_id, group_id, epoch, worker_id, succeeded ('1'/'0'), message
--
-- Returns:
--   OK:<state>
--   NOTFOUND
--   WRONGGROUP
--   STALE:<current_epoch>   the report names a superseded membership
--   NOTADMITTED             the reporter is not an admitted generation
--
-- Fencing matters more here than it looks. A worker that died mid-collective,
-- restarted, and reported success would otherwise complete an operation whose
-- transfer never finished, and the orchestrator would advance to a version the
-- generators never installed.

local state = redis.call('HGET', KEYS[1], 'state')
if not state then
  return 'NOTFOUND'
end

if redis.call('HGET', KEYS[1], 'group_id') ~= ARGV[2] then
  return 'WRONGGROUP'
end

local current_epoch = redis.call('HGET', KEYS[3], 'epoch')
if not current_epoch or tonumber(current_epoch) ~= tonumber(ARGV[3]) then
  return 'STALE:' .. (current_epoch or '0')
end

local admitted = false
local participants = redis.call('HVALS', KEYS[4])
for i = 1, #participants do
  local worker = string.match(participants[i], '^([^|]*)')
  if worker == ARGV[4] then
    admitted = true
    break
  end
end
if not admitted then
  return 'NOTADMITTED'
end

redis.call('SADD', KEYS[2], ARGV[4])

if ARGV[5] == '0' then
  redis.call('HSET', KEYS[1], 'state', 'FAILED', 'failure_message', ARGV[6])
  return 'OK:FAILED'
end

if state == 'FAILED' or state == 'ABORTED' then
  return 'OK:' .. state
end

local reported = redis.call('SCARD', KEYS[2])
local expected = tonumber(redis.call('HGET', KEYS[3], 'expected_total'))
if expected and reported >= expected then
  redis.call('HSET', KEYS[1], 'state', 'COMPLETE')
  return 'OK:COMPLETE'
end

redis.call('HSET', KEYS[1], 'state', 'RUNNING')
return 'OK:RUNNING'
