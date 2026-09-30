-- Atomically reserve an idempotency key and open one collective transfer.
--
-- KEYS[1]: operation hash
-- KEYS[2]: create-request idempotency key
-- KEYS[3]: group hash
-- ARGV: operation_id, group_id, version_id, model_name, idempotency_key,
--       state, created_at_unix_ms, operation_key_prefix,
--       requires_bootstrap_fence ('1' or '0')
--
-- Returns:
--   CREATED
--   EXISTING:<operation_id>   another invocation already owns the key
--   COLLISION                 the generated operation ID already exists
--   NOGROUP                   no group exists for this membership
--   FENCEMISMATCH             declared fence requirement differs from the group's
--   NOTBOOTSTRAPPED           group requires the fence and has not completed it
--
-- The idempotency reservation is what makes an orchestrator retry safe: a
-- create that timed out client-side but committed server-side returns the
-- original operation instead of opening a second one against the same group.
-- A replay whose operation belongs to a superseded group epoch instead
-- reclaims the reservation and falls through to a fresh create, which
-- answers to the current epoch's bootstrap gate.

-- The group epoch leads: a replay is judged against the live group, and a
-- replay whose group is gone gets NOGROUP rather than reviving a stale
-- operation.
local epoch = redis.call('HGET', KEYS[3], 'epoch')
if not epoch then
  return 'NOGROUP'
end

-- The gate is opt-in per group: only a cohort that declared the fence at
-- formation is held to it, so pre-fence clients keep working against an
-- upgraded server. A missing field (a group formed by a pre-fence server
-- during a rolling upgrade) reads as '0'.
local fence_required = redis.call('HGET', KEYS[3], 'requires_bootstrap_fence') or '0'

local existing = redis.call('GET', KEYS[2])
if existing then
  local existing_key = ARGV[8] .. existing
  if redis.call('EXISTS', existing_key) == 1 then
    -- A replay still answers to the group's fence requirement: the gate is
    -- only as strong as its retries.
    if fence_required ~= ARGV[9] then
      return 'FENCEMISMATCH'
    end
    local same_identity = redis.call('HGET', existing_key, 'group_id') == ARGV[2]
        and redis.call('HGET', existing_key, 'version_id') == ARGV[3]
    if same_identity
        and tonumber(redis.call('HGET', existing_key, 'epoch')) ~= tonumber(epoch) then
      -- A replay from a superseded epoch reclaims the reservation and falls
      -- through to a fresh create, which enforces the current epoch's gate;
      -- the superseded row is tombstoned when it never reached a terminal
      -- state.
      local state = redis.call('HGET', existing_key, 'state')
      if state and state ~= 'COMPLETE' and state ~= 'FAILED' and state ~= 'ABORTED' then
        redis.call('HSET', existing_key,
          'state', 'ABORTED',
          'failure_message', 'collective transfer was superseded by a newer group epoch')
      end
    else
      return 'EXISTING:' .. existing
    end
  end
  -- Reclaimed reservation, or an orphan left by partial/manual cleanup;
  -- normal deletion removes the operation and reservation atomically.
  redis.call('DEL', KEYS[2])
end

if redis.call('EXISTS', KEYS[1]) == 1 then
  return 'COLLISION'
end

if fence_required ~= ARGV[9] then
  return 'FENCEMISMATCH'
end
if fence_required == '1'
    and tonumber(redis.call('HGET', KEYS[3], 'bootstrap_complete_epoch')) ~= tonumber(epoch) then
  return 'NOTBOOTSTRAPPED'
end

redis.call('HSET', KEYS[1],
  'operation_id', ARGV[1],
  'group_id', ARGV[2],
  'epoch', epoch,
  'version_id', ARGV[3],
  'model_name', ARGV[4],
  'idempotency_key', ARGV[5],
  'state', ARGV[6],
  'failure_message', '',
  'created_at_unix_ms', ARGV[7])
redis.call('SET', KEYS[2], ARGV[1])

return 'CREATED'
