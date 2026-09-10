-- Atomically record one lane leader's ncclUniqueId, then re-evaluate readiness.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash
-- KEYS[3]: reported plan digests hash
-- KEYS[4]: the lane hash being published
-- KEYS[5..]: every lane hash, for the readiness sweep
-- ARGV[1]: epoch the caller generated the identifier for
-- ARGV[2]: worker_id
-- ARGV[3]: nccl_unique_id, hex encoded
-- ARGV[4]: slot_id assigned rank 0 in this lane

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

if tonumber(ARGV[1]) ~= epoch then
  return 'STALE:' .. epoch
end

local model_name = redis.call('HGET', KEYS[1], 'model_name')
local leader_record = redis.call('HGET', KEYS[2], ARGV[4])
local leader_worker, leader_role, leader_index, leader_partition, joined_epoch =
  parse_participant(leader_record)
if not leader_worker or leader_worker ~= ARGV[2] or tonumber(joined_epoch) ~= epoch
    or not registration_matches(leader_worker, leader_role, model_name) then
  return 'NOTLEADER'
end

local existing_epoch = tonumber(redis.call('HGET', KEYS[4], 'bootstrap_epoch'))
if existing_epoch == epoch then
  local existing_publisher = redis.call('HGET', KEYS[4], 'published_by')
  local existing_id = redis.call('HGET', KEYS[4], 'nccl_unique_id')
  if existing_publisher ~= ARGV[2] or existing_id ~= ARGV[3] then
    return 'CONFLICT'
  end
else
  redis.call('HSET', KEYS[4],
    'nccl_unique_id', ARGV[3],
    'bootstrap_epoch', epoch,
    'published_by', ARGV[2])
end

local lane_count = #KEYS - 4
local expected = tonumber(redis.call('HGET', KEYS[1], 'expected_total'))
local admitted = redis.call('HLEN', KEYS[2])
local ready = expected ~= nil and admitted == expected

if ready then
  local records = redis.call('HVALS', KEYS[2])
  for i = 1, #records do
    local worker_id, role, index, partition, participant_epoch = parse_participant(records[i])
    if not worker_id or tonumber(participant_epoch) ~= epoch
        or not registration_matches(worker_id, role, model_name) then
      ready = false
      break
    end
  end
end

if ready then
  for i = 1, lane_count do
    local stamped = redis.call('HGET', KEYS[4 + i], 'bootstrap_epoch')
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
