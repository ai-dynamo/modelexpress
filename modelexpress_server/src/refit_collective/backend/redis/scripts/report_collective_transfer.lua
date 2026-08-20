-- Atomically record one participant's terminal result for a transfer.
--
-- KEYS[1]: operation hash
-- KEYS[2]: reported worker set
-- KEYS[3]: group hash
-- KEYS[4]: participants hash
-- KEYS[5..]: every lane hash, cleared if the collective fails
-- ARGV: operation_id, group_id, epoch, worker_id, succeeded ('1'/'0'), message

local function parse_participant(record)
  if not record then
    return nil
  end
  return string.match(record, '^([^|]*)|([^|]*)|([^|]*)|([^|]*)|([^|]*)$')
end

local state = redis.call('HGET', KEYS[1], 'state')
if not state then
  return 'NOTFOUND'
end

if redis.call('HGET', KEYS[1], 'group_id') ~= ARGV[2] then
  return 'WRONGGROUP'
end

local operation_epoch = redis.call('HGET', KEYS[1], 'epoch')
if not operation_epoch or tonumber(operation_epoch) ~= tonumber(ARGV[3]) then
  return 'OPSTALE:' .. (operation_epoch or '0')
end

-- Terminal state is immutable. In particular, a late failure must not regress
-- COMPLETE to FAILED, and a late success must not erase a failure.
if state == 'COMPLETE' or state == 'FAILED' or state == 'ABORTED' then
  return 'OK:' .. state
end

local current_epoch = redis.call('HGET', KEYS[3], 'epoch')
if not current_epoch or tonumber(current_epoch) ~= tonumber(ARGV[3]) then
  return 'STALE:' .. (current_epoch or '0')
end
if redis.call('HGET', KEYS[3], 'state') ~= 'READY' then
  return 'NOTREADY'
end

local admitted = false
local participants = redis.call('HVALS', KEYS[4])
for i = 1, #participants do
  local worker_id, role, index, partition, joined_epoch = parse_participant(participants[i])
  if not worker_id or tonumber(joined_epoch) ~= tonumber(ARGV[3])
      or redis.call('EXISTS', 'mx:refit:worker:' .. worker_id) ~= 1 then
    return 'NOTREADY'
  end
  if worker_id == ARGV[4] then
    admitted = true
  end
end
if not admitted then
  return 'NOTADMITTED'
end

redis.call('SADD', KEYS[2], ARGV[4])

if ARGV[5] == '0' then
  redis.call('HSET', KEYS[1], 'state', 'FAILED', 'failure_message', ARGV[6])

  -- A failed collective has an unusable communicator. Move the group epoch in
  -- the same transaction so no later operation can reuse its bootstrap IDs.
  local next_epoch = tonumber(current_epoch) + 1
  redis.call('HSET', KEYS[3],
    'epoch', next_epoch,
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '')
  for i = 5, #KEYS do
    redis.call('DEL', KEYS[i])
  end
  return 'OK:FAILED'
end

local reported = redis.call('SCARD', KEYS[2])
local expected = tonumber(redis.call('HGET', KEYS[3], 'expected_total'))
if expected and reported == expected then
  redis.call('HSET', KEYS[1], 'state', 'COMPLETE')
  return 'OK:COMPLETE'
end

redis.call('HSET', KEYS[1], 'state', 'RUNNING')
return 'OK:RUNNING'
