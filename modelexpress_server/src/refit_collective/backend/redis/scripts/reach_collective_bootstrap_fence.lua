-- Record one admitted worker generation at a per-lane bootstrap fence.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash
-- KEYS[3]: fixed per-group/per-lane fence hash
-- ARGV: epoch, lane_id, phase, slot_id, worker_id
--
-- Returns:
--   OK:<released>:<newline-separated sorted missing slots>
--   NOTFOUND
--   STALE:<current epoch>
--   NOTREADY
--   NOTADMITTED
--   NOTLIVE
--   NOLANE
--   PREINCOMPLETE:<lane_id>

local function parse_participant(record)
  if not record then
    return nil
  end
  return string.match(record, '^([^|]*)|([^|]*)|([^|]*)|([^|]*)$')
end

local function registration_matches(worker_id, role, model_name)
  local key = 'mx:refit:worker:' .. worker_id
  if redis.call('EXISTS', key) ~= 1 then
    return false
  end
  local expected_role = role == 'TRAINER' and '1' or '2'
  return redis.call('HGET', key, 'worker_id') == worker_id
    and redis.call('HGET', key, 'role') == expected_role
    and redis.call('HGET', key, 'model_name') == model_name
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
if ARGV[3] ~= '1' and ARGV[3] ~= '2' then
  return 'NOPHASE'
end

local declared = false
local lanes = redis.call('HGET', KEYS[1], 'lanes') or ''
for line in string.gmatch(lanes .. '\n', '([^\n]*)\n') do
  local lane_id = string.match(line, '^([^|]*)|')
  if lane_id == ARGV[2] then
    declared = true
    break
  end
end
if not declared then
  return 'NOLANE'
end

local model_name = redis.call('HGET', KEYS[1], 'model_name')
local caller = redis.call('HGET', KEYS[2], ARGV[4])
local caller_worker, caller_role, caller_index, caller_epoch =
  parse_participant(caller)
if not caller_worker or caller_worker ~= ARGV[5]
    or tonumber(caller_epoch) ~= epoch then
  return 'NOTADMITTED'
end
if not registration_matches(caller_worker, caller_role, model_name) then
  return 'NOTLIVE'
end

-- READY is revalidated atomically. A registration can expire after the group
-- was last read but before this arrival, and releasing that stale cohort would
-- let a live rank enter NCCL without its peer.
local records = redis.call('HVALS', KEYS[2])
local expected = tonumber(redis.call('HGET', KEYS[1], 'expected_total'))
if not expected or #records ~= expected then
  return 'NOTREADY'
end
for i = 1, #records do
  local worker_id, role, index, joined_epoch = parse_participant(records[i])
  if not worker_id or tonumber(joined_epoch) ~= epoch
      or not registration_matches(worker_id, role, model_name) then
    return 'NOTREADY'
  end
end

local fence_epoch = tonumber(redis.call('HGET', KEYS[3], 'epoch'))
if fence_epoch ~= epoch then
  redis.call('DEL', KEYS[3])
  redis.call('HSET', KEYS[3], 'epoch', epoch)
end

local slots = {}
for slot in string.gmatch(
    (redis.call('HGET', KEYS[1], 'expected_trainer_slots') or '') .. '\n'
      .. (redis.call('HGET', KEYS[1], 'expected_generator_slots') or '') .. '\n',
    '([^\n]*)\n') do
  if slot ~= '' then
    table.insert(slots, slot)
  end
end
table.sort(slots)

if ARGV[3] == '2' then
  for line in string.gmatch(lanes .. '\n', '([^\n]*)\n') do
    local lane_id = string.match(line, '^([^|]*)|')
    if lane_id then
      local pre_fence_key = KEYS[1] .. ':fence:' .. lane_id
      if tonumber(redis.call('HGET', pre_fence_key, 'epoch')) ~= epoch then
        return 'PREINCOMPLETE:' .. lane_id
      end
      for i = 1, #slots do
        if not redis.call(
            'HGET', pre_fence_key, 'phase:1:slot:' .. slots[i]) then
          return 'PREINCOMPLETE:' .. lane_id
        end
      end
    end
  end
end

local arrival_prefix = 'phase:' .. ARGV[3] .. ':slot:'
redis.call('HSET', KEYS[3], arrival_prefix .. ARGV[4], ARGV[5])

local missing = {}
for i = 1, #slots do
  if not redis.call('HGET', KEYS[3], arrival_prefix .. slots[i]) then
    table.insert(missing, slots[i])
  end
end
local released = #missing == 0 and '1' or '0'
if released == '1' and ARGV[3] == '2' then
  redis.call('HSET', KEYS[1], 'bootstrap_complete_epoch', epoch)
end
return 'OK:' .. released .. ':' .. table.concat(missing, '\n')
