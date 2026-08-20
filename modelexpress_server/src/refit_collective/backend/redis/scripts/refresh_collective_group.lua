-- Re-evaluate registration liveness and readiness before returning a group.
-- Expired workers are removed and fence the old communicator by moving epoch.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash
-- KEYS[3]: reported plan digests hash
-- KEYS[4..]: every lane hash

local function parse_participant(record)
  if not record then
    return nil
  end
  return string.match(record, '^([^|]*)|([^|]*)|([^|]*)|([^|]*)|([^|]*)$')
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

local model_name = redis.call('HGET', KEYS[1], 'model_name')
local changed = false
local records = redis.call('HGETALL', KEYS[2])
for i = 1, #records, 2 do
  local slot_id = records[i]
  local worker_id, role = parse_participant(records[i + 1])
  if not worker_id or not registration_matches(worker_id, role, model_name) then
    redis.call('HDEL', KEYS[2], slot_id)
    redis.call('HDEL', KEYS[3], slot_id)
    changed = true
  end
end

if changed then
  epoch = epoch + 1
  redis.call('HSET', KEYS[1],
    'epoch', epoch,
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '')
  for i = 4, #KEYS do
    redis.call('DEL', KEYS[i])
  end
end

local expected = tonumber(redis.call('HGET', KEYS[1], 'expected_total'))
local admitted = redis.call('HLEN', KEYS[2])
local ready = expected ~= nil and admitted == expected

if ready then
  local participants = redis.call('HVALS', KEYS[2])
  for i = 1, #participants do
    local worker_id, role, index, partition, joined_epoch = parse_participant(participants[i])
    if not worker_id or tonumber(joined_epoch) ~= epoch
        or not registration_matches(worker_id, role, model_name) then
      ready = false
      break
    end
  end
end

if ready then
  for i = 4, #KEYS do
    local stamped = redis.call('HGET', KEYS[i], 'bootstrap_epoch')
    if not stamped or tonumber(stamped) ~= epoch then
      ready = false
      break
    end
  end
end

if ready then
  local digest = redis.call('HGET', KEYS[1], 'plan_digest')
  local reported = redis.call('HVALS', KEYS[3])
  if #reported ~= expected then
    ready = false
  else
    for i = 1, #reported do
      if reported[i] ~= digest then
        ready = false
        break
      end
    end
  end
end

local state = ready and 'READY' or 'FORMING'
redis.call('HSET', KEYS[1], 'state', state)
return 'OK:' .. epoch .. ':' .. state
