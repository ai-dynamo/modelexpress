-- Atomically record one participant's terminal result for a transfer.
--
-- KEYS[1]: operation hash
-- KEYS[2]: reported worker set
-- KEYS[3]: group hash
-- KEYS[4]: participants hash
-- KEYS[5]: create-request idempotency key
-- KEYS[6..]: every lane hash, cleared if the collective fails
-- ARGV: operation_id, group_id, epoch, worker_id, succeeded ('1'/'0'), message,
--       deadline_message

local function parse_participant(record)
  if not record then
    return nil
  end
  return string.match(record, '^([^|]*)|([^|]*)|([^|]*)|([^|]*)$')
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

-- The expiry check belongs in this transaction. A preflight sweep alone would
-- leave a race where the deadline elapsed after the sweep but before a late
-- report turned RUNNING into COMPLETE.
if state ~= 'COMPLETE' and state ~= 'FAILED' and state ~= 'ABORTED' then
  local deadline = tonumber(redis.call('HGET', KEYS[1], 'deadline_unix_ms'))
  local clock = redis.call('TIME')
  local now_ms = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
  if not deadline or deadline <= now_ms then
    redis.call('HSET', KEYS[1],
      'state', 'ABORTED',
      'failure_message', ARGV[7])

    local current_epoch = tonumber(redis.call('HGET', KEYS[3], 'epoch'))
    if current_epoch and current_epoch == tonumber(operation_epoch) then
      redis.call('HSET', KEYS[3],
        'epoch', current_epoch + 1,
        'state', 'FORMING',
        'bootstrap_complete_epoch', 0,
        'active_operation_id', '',
        'plan_source_worker_id', '',
        'plan_source_endpoint', '',
        'plan_source_digest', '')
      for i = 6, #KEYS do
        redis.call('DEL', KEYS[i])
      end
      local lanes = redis.call('HGET', KEYS[3], 'lanes')
      if lanes then
        for line in string.gmatch(lanes .. '\n', '([^\n]*)\n') do
          local lane_id = string.match(line, '^([^|]*)|')
          if lane_id then
            redis.call('DEL', KEYS[3] .. ':fence:' .. lane_id)
          end
        end
      end
    end
    if redis.call('GET', KEYS[5]) == ARGV[1] then
      redis.call('DEL', KEYS[5])
    end
    return 'OK:ABORTED'
  end
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
  local worker_id, role, index, joined_epoch = parse_participant(participants[i])
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
    'bootstrap_complete_epoch', 0,
    'active_operation_id', '',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '')
  for i = 6, #KEYS do
    redis.call('DEL', KEYS[i])
  end
  local lanes = redis.call('HGET', KEYS[3], 'lanes')
  if lanes then
    for line in string.gmatch(lanes .. '\n', '([^\n]*)\n') do
      local lane_id = string.match(line, '^([^|]*)|')
      if lane_id then
        redis.call('DEL', KEYS[3] .. ':fence:' .. lane_id)
      end
    end
  end
  return 'OK:FAILED'
end

local reported = redis.call('SCARD', KEYS[2])
local expected = tonumber(redis.call('HGET', KEYS[3], 'expected_total'))
if expected and reported == expected then
  redis.call('HSET', KEYS[1], 'state', 'COMPLETE')
  if redis.call('HGET', KEYS[3], 'active_operation_id') == ARGV[1] then
    redis.call('HSET', KEYS[3], 'active_operation_id', '')
  end
  return 'OK:COMPLETE'
end

redis.call('HSET', KEYS[1], 'state', 'RUNNING')
return 'OK:RUNNING'
