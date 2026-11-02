-- Re-evaluate registration liveness and readiness before returning a group.
-- Expired workers are removed and fence the old communicator by moving epoch.
--
-- KEYS[1]: group hash
-- KEYS[2]: participants hash
-- KEYS[3]: reported plan digests hash
-- KEYS[4]: group active-operations set
-- KEYS[5..]: every lane hash
-- ARGV[1]: operation_key_prefix
--
-- The helpers come from common.lua, which redis.rs prepends.

local lane_keys = key_range(5, #KEYS)

local epoch = tonumber(redis.call('HGET', KEYS[1], 'epoch'))
if not epoch then
  return 'NOTFOUND'
end

local model_name = redis.call('HGET', KEYS[1], 'model_name')
local changed = false

-- Same guard join_collective_group.lua applies to its identical sweep. Only a
-- membership that could already have communicator state needs fencing: a READY
-- group, or one where some lane has published a current-epoch bootstrap. In a
-- clean FORMING epoch a lapse just removes the participant, so waiters keep
-- their epoch and time out naming the missing slot instead of being told to
-- rebuild. This script runs on every GetCollectiveGroup poll.
local membership_change_requires_bump = needs_epoch_bump(KEYS[1], lane_keys, epoch)

local records = redis.call('HGETALL', KEYS[2])
for i = 1, #records, 2 do
  local slot_id = records[i]
  local worker_id, role = parse_participant(records[i + 1])
  if not worker_id or not registration_matches(worker_id, role, model_name) then
    redis.call('HDEL', KEYS[2], slot_id)
    redis.call('HDEL', KEYS[3], slot_id)
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

if changed then
  epoch = epoch + 1
  redis.call('HSET', KEYS[1],
    'epoch', epoch,
    'state', 'FORMING',
    'plan_source_worker_id', '',
    'plan_source_endpoint', '',
    'plan_source_digest', '')
  for i = 5, #KEYS do
    redis.call('DEL', KEYS[i])
  end
  abort_stale_ops(KEYS[4], ARGV[1])
end

-- The sweep above already dropped every participant whose registration lapsed,
-- and nothing runs between it and here, so only the epoch is checked again:
-- no model_name, so evaluate_readiness skips the registration lookups.
local expected = tonumber(redis.call('HGET', KEYS[1], 'expected_total'))
local ready = evaluate_readiness(
  KEYS[1], KEYS[2], KEYS[3], lane_keys, expected, epoch, nil)

local state = ready and 'READY' or 'FORMING'
redis.call('HSET', KEYS[1], 'state', state)
return 'OK:' .. epoch .. ':' .. state
