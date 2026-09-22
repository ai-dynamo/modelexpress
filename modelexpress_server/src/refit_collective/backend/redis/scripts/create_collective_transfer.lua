-- Atomically reserve an idempotency key and open one collective transfer.
--
-- KEYS[1]: operation hash
-- KEYS[2]: create-request idempotency key
-- KEYS[3]: group hash
-- ARGV: operation_id, group_id, version_id, model_name, idempotency_key,
--       state, transfer_timeout_ms, operation_key_prefix, deadline_message
--
-- Returns:
--   CREATED
--   EXISTING:<operation_id>   another invocation already owns the key
--   COLLISION                 the generated operation ID already exists
--   NOGROUP                   no group exists for this membership
--   NOTREADY                  group is not READY
--   NOTBOOTSTRAPPED           final all-rank bootstrap completion is absent
--   ACTIVE:<operation_id>     another transfer owns this group epoch
--
-- The idempotency reservation is what makes an orchestrator retry safe: a
-- create that timed out client-side but committed server-side returns the
-- original operation instead of opening a second one against the same group.
--
-- Redis time, rather than an MX server clock, is the authority for both the
-- stored deadline and deciding whether a retry may reclaim an expired key.

local function now_ms()
  local clock = redis.call('TIME')
  return tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
end

local function clear_group_bootstraps(group_id)
  local group = 'mx:refitc:group:' .. group_id
  local lanes = redis.call('HGET', group, 'lanes')
  if not lanes then
    return
  end
  for line in string.gmatch(lanes .. '\n', '([^\n]*)\n') do
    local lane_id = string.match(line, '^([^|]*)|')
    if lane_id then
      redis.call('DEL', group .. ':lane:' .. lane_id)
      redis.call('DEL', group .. ':fence:' .. lane_id)
    end
  end
end

local function reservation_key_for(operation_key)
  local model_name = redis.call('HGET', operation_key, 'model_name')
  local idempotency_key = redis.call('HGET', operation_key, 'idempotency_key')
  if not model_name or not idempotency_key then
    return nil
  end
  return 'mx:refitc:op-request:' .. string.len(model_name)
    .. ':' .. model_name .. idempotency_key
end

local function abort_if_expired(operation_id, operation_key, reservation_key)
  local state = redis.call('HGET', operation_key, 'state')
  if not state or state == 'COMPLETE' or state == 'FAILED' or state == 'ABORTED' then
    return false
  end

  local deadline = tonumber(redis.call('HGET', operation_key, 'deadline_unix_ms'))
  if deadline and deadline > now_ms() then
    return false
  end

  redis.call('HSET', operation_key,
    'state', 'ABORTED',
    'failure_message', ARGV[9])

  local group_id = redis.call('HGET', operation_key, 'group_id')
  local operation_epoch = tonumber(redis.call('HGET', operation_key, 'epoch'))
  local group = group_id and ('mx:refitc:group:' .. group_id) or nil
  local group_epoch = group and tonumber(redis.call('HGET', group, 'epoch')) or nil
  if group_epoch and operation_epoch and group_epoch == operation_epoch then
    redis.call('HSET', group,
      'epoch', group_epoch + 1,
      'state', 'FORMING',
      'bootstrap_complete_epoch', 0,
      'active_operation_id', '',
      'plan_source_worker_id', '',
      'plan_source_endpoint', '',
      'plan_source_digest', '')
    clear_group_bootstraps(group_id)
  end

  if reservation_key and redis.call('GET', reservation_key) == operation_id then
    redis.call('DEL', reservation_key)
  end
  return true
end

local function reclaim_stale_reservation(operation_id, operation_key, reservation_key)
  local state = redis.call('HGET', operation_key, 'state')
  if state and state ~= 'COMPLETE' and state ~= 'FAILED' and state ~= 'ABORTED' then
    redis.call('HSET', operation_key,
      'state', 'ABORTED',
      'failure_message', 'collective transfer was superseded by a newer group epoch')
  end
  if redis.call('GET', reservation_key) == operation_id then
    redis.call('DEL', reservation_key)
  end
end

local epoch = redis.call('HGET', KEYS[3], 'epoch')
if not epoch then
  return 'NOGROUP'
end

local existing = redis.call('GET', KEYS[2])
if existing then
  local existing_key = ARGV[8] .. existing
  if redis.call('EXISTS', existing_key) == 1 then
    local existing_group_id = redis.call('HGET', existing_key, 'group_id')
    local existing_epoch = tonumber(redis.call('HGET', existing_key, 'epoch'))
    local existing_version_id = redis.call('HGET', existing_key, 'version_id')
    if existing_group_id == ARGV[2]
        and existing_version_id == ARGV[3]
        and existing_epoch ~= tonumber(epoch) then
      reclaim_stale_reservation(existing, existing_key, KEYS[2])
    elseif not abort_if_expired(existing, existing_key, KEYS[2]) then
      return 'EXISTING:' .. existing
    end
  end
  if redis.call('GET', KEYS[2]) == existing then
    -- Recover an orphaned reservation left by partial/manual metadata cleanup.
    -- Normal deletion and deadline expiry remove the reservation atomically.
    redis.call('DEL', KEYS[2])
  end
end

if redis.call('EXISTS', KEYS[1]) == 1 then
  return 'COLLISION'
end

local active_operation = redis.call('HGET', KEYS[3], 'active_operation_id')
if active_operation and active_operation ~= '' then
  local active_key = ARGV[8] .. active_operation
  local active_state = redis.call('HGET', active_key, 'state')
  if not active_state or active_state == 'COMPLETE'
      or active_state == 'FAILED' or active_state == 'ABORTED' then
    redis.call('HSET', KEYS[3], 'active_operation_id', '')
  else
    local active_reservation = reservation_key_for(active_key)
    if not abort_if_expired(active_operation, active_key, active_reservation) then
      return 'ACTIVE:' .. active_operation
    end
  end
end
if redis.call('HGET', KEYS[3], 'state') ~= 'READY' then
  return 'NOTREADY'
end
if tonumber(redis.call('HGET', KEYS[3], 'bootstrap_complete_epoch')) ~= tonumber(epoch) then
  return 'NOTBOOTSTRAPPED'
end

local created_at = now_ms()
local deadline = created_at + tonumber(ARGV[7])
redis.call('HSET', KEYS[1],
  'operation_id', ARGV[1],
  'group_id', ARGV[2],
  'epoch', epoch,
  'version_id', ARGV[3],
  'model_name', ARGV[4],
  'idempotency_key', ARGV[5],
  'state', ARGV[6],
  'failure_message', '',
  'created_at_unix_ms', created_at,
  'deadline_unix_ms', deadline)
redis.call('SET', KEYS[2], ARGV[1])
redis.call('HSET', KEYS[3], 'active_operation_id', ARGV[1])

return 'CREATED'
