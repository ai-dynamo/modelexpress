-- Atomically admit one live, registered worker to a collective group, creating
-- the group on first contact, and re-evaluate readiness.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash (slot_id -> worker_id|role|index|partition|joined_epoch)
-- KEYS[3]: reported plan digests hash (slot_id -> digest)
-- KEYS[4]: the incoming worker's TTL-bound Refit registration
-- KEYS[5..4+lane_count]: one lane hash per lane, reshard lanes then broadcast
-- ARGV[1]:  group_id
-- ARGV[2]:  model_name
-- ARGV[3]:  source_partition_count
-- ARGV[4]:  expected participant total
-- ARGV[5]:  slot_id
-- ARGV[6]:  worker_id
-- ARGV[7]:  role (TRAINER or GENERATOR)
-- ARGV[8]:  index_in_role
-- ARGV[9]:  source_partition ('' for generators)
-- ARGV[10]: plan_digest
-- ARGV[11]: plan_source_worker_id ('' when this worker does not serve it)
-- ARGV[12]: plan_source_endpoint
-- ARGV[13]: plan_source_digest
-- ARGV[14]: expected_trainer_slots, newline separated
-- ARGV[15]: expected_generator_slots, newline separated
-- ARGV[16]: created_at_unix_ms

local function contains_slot(list, target)
  for slot in string.gmatch(list .. '\n', '([^\n]*)\n') do
    if slot == target then
      return true
    end
  end
  return false
end

local function parse_participant(record)
  if not record then
    return nil
  end
  return string.match(record, '^([^|]*)|([^|]*)|([^|]*)|([^|]*)|([^|]*)$')
end

local function registration_key(worker_id)
  return 'mx:refit:worker:' .. worker_id
end

local function registration_matches(worker_id, role, model_name)
  local key = registration_key(worker_id)
  if redis.call('EXISTS', key) ~= 1 then
    return false
  end
  local expected_role = role == 'TRAINER' and '1' or '2'
  return redis.call('HGET', key, 'worker_id') == worker_id
    and redis.call('HGET', key, 'role') == expected_role
    and redis.call('HGET', key, 'model_name') == model_name
end

local lane_count = #KEYS - 4
local expected_slots = ARGV[7] == 'TRAINER' and ARGV[14] or ARGV[15]
if not contains_slot(expected_slots, ARGV[5]) then
  return 'UNEXPECTED_SLOT'
end

if not registration_matches(ARGV[6], ARGV[7], ARGV[2]) then
  return 'UNREGISTERED'
end

if ARGV[11] ~= '' then
  if ARGV[7] ~= 'TRAINER' or ARGV[8] ~= '0' or ARGV[11] ~= ARGV[6]
      or redis.call('HGET', KEYS[4], 'endpoint') ~= ARGV[12] then
    return 'INVALID_PLAN_SOURCE'
  end
end

local epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
local changed = false

if not epoch then
  epoch = 1
  -- The group hash and its derived hashes do not share a Redis lifetime. If the
  -- root hash was evicted or manually removed, never let epoch-1 participants
  -- or communicator identifiers from the prior incarnation survive recreation.
  redis.call('DEL', KEYS[2])
  redis.call('DEL', KEYS[3])
  for i = 1, lane_count do
    redis.call('DEL', KEYS[4 + i])
  end
  redis.call('HSET', KEYS[1],
    'group_id', ARGV[1],
    'model_name', ARGV[2],
    'source_partition_count', ARGV[3],
    'expected_total', ARGV[4],
    'expected_trainer_slots', ARGV[14],
    'expected_generator_slots', ARGV[15],
    'plan_digest', ARGV[10],
    'epoch', epoch,
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '',
    'created_at_unix_ms', ARGV[16])
else
  -- Expired registrations are no longer admitted. Removing them here and
  -- advancing the epoch fences any communicator they helped form.
  local records = redis.call('HGETALL', KEYS[2])
  for i = 1, #records, 2 do
    local slot_id = records[i]
    local worker_id, role = parse_participant(records[i + 1])
    if not worker_id or not registration_matches(worker_id, role, ARGV[2]) then
      redis.call('HDEL', KEYS[2], slot_id)
      redis.call('HDEL', KEYS[3], slot_id)
      changed = true
    end
  end

  local existing = redis.call('HGET', KEYS[2], ARGV[5])
  if existing then
    local worker_id, role, index, partition = parse_participant(existing)
    if not worker_id then
      return 'CORRUPT_PARTICIPANT'
    end
    -- A stable slot has one immutable rank assignment. Only its process
    -- generation may change; changing role/ordinal/partition under the same
    -- group identity would put peers in different lanes.
    if role ~= ARGV[7] or index ~= ARGV[8] or partition ~= ARGV[9] then
      return 'CONFLICTING_ASSIGNMENT'
    end
    if worker_id ~= ARGV[6] then
      changed = true
    end
  end

  local records = redis.call('HGETALL', KEYS[2])
  for i = 1, #records, 2 do
    local slot_id = records[i]
    if slot_id ~= ARGV[5] then
      local worker_id, role, index = parse_participant(records[i + 1])
      if not worker_id then
        return 'CORRUPT_PARTICIPANT'
      end
      if role == ARGV[7] and index == ARGV[8] then
        return 'DUPLICATE_RANK'
      end
      if worker_id == ARGV[6] then
        return 'DUPLICATE_WORKER'
      end
    end
  end

  if redis.call('HGET', KEYS[1], 'plan_digest') ~= ARGV[10] then
    changed = true
  end
end

if changed then
  epoch = epoch + 1
  redis.call('HSET', KEYS[1],
    'epoch', epoch,
    'plan_digest', ARGV[10],
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '')
  for i = 1, lane_count do
    redis.call('DEL', KEYS[4 + i])
  end
end

redis.call('HSET', KEYS[2], ARGV[5],
  ARGV[6] .. '|' .. ARGV[7] .. '|' .. ARGV[8] .. '|' .. ARGV[9] .. '|' .. epoch)
redis.call('HSET', KEYS[3], ARGV[5], ARGV[10])

if ARGV[11] ~= '' then
  redis.call('HSET', KEYS[1],
    'plan_source_worker_id', ARGV[11],
    'plan_source_endpoint', ARGV[12],
    'plan_source_digest', ARGV[13])
end

-- READY means every exact slot/rank is live and has acknowledged this epoch,
-- every lane is bootstrapped for it, and every digest agrees.
local admitted = redis.call('HLEN', KEYS[2])
local ready = admitted == tonumber(ARGV[4])

if ready then
  local records = redis.call('HVALS', KEYS[2])
  for i = 1, #records do
    local worker_id, role, index, partition, joined_epoch = parse_participant(records[i])
    if not worker_id or tonumber(joined_epoch) ~= epoch
        or not registration_matches(worker_id, role, ARGV[2]) then
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
  local digests = redis.call('HVALS', KEYS[3])
  local digest = redis.call('HGET', KEYS[1], 'plan_digest')
  if #digests ~= tonumber(ARGV[4]) then
    ready = false
  else
    for i = 1, #digests do
      if digests[i] ~= digest then
        ready = false
        break
      end
    end
  end
end

local state = ready and 'READY' or 'FORMING'
redis.call('HSET', KEYS[1], 'state', state)

return 'OK:' .. epoch .. ':' .. state
