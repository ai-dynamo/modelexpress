-- Fence a failed post-READY bootstrap and return the group to FORMING.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash
-- KEYS[3]: caller worker registration
-- ARGV: epoch, slot_id, worker_id, failure_message
--
-- Returns OK:<new epoch>, NOTFOUND, STALE:<current epoch>, NOTREADY,
-- ACTIVE:<operation_id>, NOTADMITTED, or NOTLIVE.

local function parse_participant(record)
  if not record then
    return nil
  end
  return string.match(record, '^([^|]*)|([^|]*)|([^|]*)|([^|]*)$')
end

local epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
if not epoch then
  return 'NOTFOUND'
end
if tonumber(ARGV[1]) ~= epoch then
  return 'STALE:' .. epoch
end
if redis.call('HGET', KEYS[1], 'state') ~= 'READY' then
  return 'NOTREADY'
end
local active_operation = redis.call('HGET', KEYS[1], 'active_operation_id')
if active_operation and active_operation ~= '' then
  return 'ACTIVE:' .. active_operation
end

local record = redis.call('HGET', KEYS[2], ARGV[2])
local worker_id, role, index, joined_epoch = parse_participant(record)
if not worker_id or worker_id ~= ARGV[3] or tonumber(joined_epoch) ~= epoch then
  return 'NOTADMITTED'
end
local expected_role = role == 'TRAINER' and '1' or '2'
if redis.call('EXISTS', KEYS[3]) ~= 1
    or redis.call('HGET', KEYS[3], 'worker_id') ~= worker_id
    or redis.call('HGET', KEYS[3], 'role') ~= expected_role
    or redis.call('HGET', KEYS[3], 'model_name')
      ~= redis.call('HGET', KEYS[1], 'model_name') then
  return 'NOTLIVE'
end

local lanes = redis.call('HGET', KEYS[1], 'lanes') or ''
for line in string.gmatch(lanes .. '\n', '([^\n]*)\n') do
  local lane_id = string.match(line, '^([^|]*)|')
  if lane_id then
    redis.call('DEL', KEYS[1] .. ':lane:' .. lane_id)
    redis.call('DEL', KEYS[1] .. ':fence:' .. lane_id)
  end
end

local next_epoch = epoch + 1
redis.call('HSET', KEYS[1],
  'epoch', next_epoch,
  'state', 'FORMING',
  'bootstrap_complete_epoch', 0,
  'active_operation_id', '',
  'plan_source_worker_id', '',
  'plan_source_endpoint', '',
  'plan_source_digest', '',
  'bootstrap_failure_message', ARGV[4])
return 'OK:' .. next_epoch
