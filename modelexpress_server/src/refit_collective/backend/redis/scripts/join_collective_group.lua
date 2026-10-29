-- Atomically admit one live, registered worker to a collective group, creating
-- the group on first contact, and re-evaluate readiness.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash (slot_id -> worker_id|role|index|joined_epoch)
-- KEYS[3]: reported plan digests hash (slot_id -> digest)
-- KEYS[4]: the incoming worker's TTL-bound Refit registration
-- KEYS[5]: group active-operations set
-- KEYS[6..5+lane_count]: one lane hash per declared lane, in declared order
-- ARGV[1]:  group_id
-- ARGV[2]:  model_name
-- ARGV[3]:  declared lane set, one line per lane
-- ARGV[4]:  expected participant total
-- ARGV[5]:  slot_id
-- ARGV[6]:  worker_id
-- ARGV[7]:  role (TRAINER or GENERATOR)
-- ARGV[8]:  index_in_role
-- ARGV[9]:  plan_digest
-- ARGV[10]: plan_source_worker_id ('' when this worker does not serve it)
-- ARGV[11]: plan_source_endpoint
-- ARGV[12]: plan_source_digest
-- ARGV[13]: expected_trainer_slots, newline separated
-- ARGV[14]: expected_generator_slots, newline separated
-- ARGV[15]: created_at_unix_ms
-- ARGV[16]: operation_key_prefix
--
-- The shared helpers come from common.lua, which redis.rs prepends.

local function contains_slot(list, target)
  for slot in string.gmatch(list .. '\n', '([^\n]*)\n') do
    if slot == target then
      return true
    end
  end
  return false
end

local lane_count = #KEYS - 5
local lane_keys = key_range(6, 5 + lane_count)
local expected_slots = ARGV[7] == 'TRAINER' and ARGV[13] or ARGV[14]
if not contains_slot(expected_slots, ARGV[5]) then
  return 'UNEXPECTED_SLOT'
end

if not registration_matches(ARGV[6], ARGV[7], ARGV[2], KEYS[4]) then
  return 'UNREGISTERED'
end

if ARGV[10] ~= '' then
  if ARGV[7] ~= 'TRAINER' or ARGV[8] ~= '0' or ARGV[10] ~= ARGV[6] then
    return 'INVALID_PLAN_SOURCE'
  end
end

local epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
local changed = false
local membership_change_requires_bump = false

if not epoch then
  epoch = 1
  -- The group hash and its derived hashes do not share a Redis lifetime. If the
  -- root hash was evicted or manually removed, never let epoch-1 participants,
  -- communicator identifiers or active operations from the prior incarnation
  -- survive recreation: the new cohort restarts at epoch 1 and could otherwise
  -- complete an old operation through its own reports.
  redis.call('DEL', KEYS[2])
  redis.call('DEL', KEYS[3])
  for i = 1, lane_count do
    redis.call('DEL', KEYS[5 + i])
  end
  abort_stale_ops(KEYS[5], ARGV[16])
  redis.call('HSET', KEYS[1],
    'group_id', ARGV[1],
    'model_name', ARGV[2],
    'lanes', ARGV[3],
    'expected_total', ARGV[4],
    'expected_trainer_slots', ARGV[13],
    'expected_generator_slots', ARGV[14],
    'plan_digest', ARGV[9],
    'epoch', epoch,
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '',
    'created_at_unix_ms', ARGV[15])
else
  -- The first membership change from READY invalidates the old communicator
  -- and opens a clean FORMING epoch. Further replacements in that same clean
  -- epoch only acknowledge it; otherwise a full-cohort restart would bump once
  -- per slot and no worker could share an epoch. Once any lane has published a
  -- current-epoch bootstrap, another membership change must invalidate it.
  membership_change_requires_bump = needs_epoch_bump(KEYS[1], lane_keys, epoch)

  -- Expired registrations are no longer admitted. Removing them advances the
  -- epoch whenever this membership could already have communicator state.
  local records = redis.call('HGETALL', KEYS[2])
  for i = 1, #records, 2 do
    local slot_id = records[i]
    local worker_id, role = parse_participant(records[i + 1])
    if not worker_id or not registration_matches(worker_id, role, ARGV[2]) then
      redis.call('HDEL', KEYS[2], slot_id)
      redis.call('HDEL', KEYS[3], slot_id)
      -- Same clearing the replacement path below does. Without it an expired
      -- plan-source owner leaves its endpoint and digest on the group, and a
      -- replacement that omits the optional plan_source lets the group reach
      -- READY advertising a dead one.
      if worker_id
          and redis.call('HGET', KEYS[1], 'plan_source_worker_id') == worker_id then
        redis.call('HSET', KEYS[1],
          'plan_source_worker_id', '',
          'plan_source_endpoint', '',
          'plan_source_digest', '')
      end
      if membership_change_requires_bump then
        changed = true
      end
    end
  end

  local existing = redis.call('HGET', KEYS[2], ARGV[5])
  if existing then
    local worker_id, role, index = parse_participant(existing)
    if not worker_id then
      return 'CORRUPT_PARTICIPANT'
    end
    -- A stable slot has one immutable rank assignment. Only its process
    -- generation may change; changing role or ordinal under the same group
    -- identity would put peers in different lanes.
    if role ~= ARGV[7] or index ~= ARGV[8] then
      return 'CONFLICTING_ASSIGNMENT'
    end
    if worker_id ~= ARGV[6] then
      if redis.call('HGET', KEYS[1], 'plan_source_worker_id') == worker_id then
        redis.call('HSET', KEYS[1],
          'plan_source_worker_id', '',
          'plan_source_endpoint', '',
          'plan_source_digest', '')
      end
      if membership_change_requires_bump then
        changed = true
      end
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

  if redis.call('HGET', KEYS[1], 'plan_digest') ~= ARGV[9] then
    changed = true
  end
end

if changed then
  epoch = epoch + 1
  redis.call('HSET', KEYS[1],
    'epoch', epoch,
    'plan_digest', ARGV[9],
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '')
  for i = 1, lane_count do
    redis.call('DEL', KEYS[5 + i])
  end
  abort_stale_ops(KEYS[5], ARGV[16])
end

redis.call('HSET', KEYS[2], ARGV[5],
  ARGV[6] .. '|' .. ARGV[7] .. '|' .. ARGV[8] .. '|' .. epoch)
redis.call('HSET', KEYS[3], ARGV[5], ARGV[9])

if ARGV[10] ~= '' then
  redis.call('HSET', KEYS[1],
    'plan_source_worker_id', ARGV[10],
    'plan_source_endpoint', ARGV[11],
    'plan_source_digest', ARGV[12])
end

local ready = evaluate_readiness(
  KEYS[1], KEYS[2], KEYS[3], lane_keys, tonumber(ARGV[4]), epoch, ARGV[2])

local state = ready and 'READY' or 'FORMING'
redis.call('HSET', KEYS[1], 'state', state)

return 'OK:' .. epoch .. ':' .. state
