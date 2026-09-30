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

local existing = redis.call('GET', KEYS[2])
if existing then
  if redis.call('EXISTS', ARGV[8] .. existing) == 1 then
    -- A replay still answers to the group's fence requirement; the gate is
    -- only as strong as its retries. A missing group row reads as '0' here,
    -- so a fenced replay against a deleted group fails loud instead of
    -- reviving a stale operation.
    local fence_required = redis.call('HGET', KEYS[3], 'requires_bootstrap_fence') or '0'
    if fence_required ~= ARGV[9] then
      return 'FENCEMISMATCH'
    end
    return 'EXISTING:' .. existing
  end
  -- Recover an orphaned reservation left by partial/manual metadata cleanup.
  -- Normal deletion removes the operation and reservation atomically.
  redis.call('DEL', KEYS[2])
end

if redis.call('EXISTS', KEYS[1]) == 1 then
  return 'COLLISION'
end

local epoch = redis.call('HGET', KEYS[3], 'epoch')
if not epoch then
  return 'NOGROUP'
end

-- The gate is opt-in per group: only a cohort that declared the fence at
-- formation is held to it, so pre-fence clients keep working against an
-- upgraded server.
local fence_required = redis.call('HGET', KEYS[3], 'requires_bootstrap_fence') or '0'
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
