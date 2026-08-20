-- Atomically reserve an idempotency key and open one collective transfer.
--
-- KEYS[1]: operation hash
-- KEYS[2]: create-request idempotency key
-- KEYS[3]: group hash
-- ARGV: operation_id, group_id, version_id, model_name, idempotency_key,
--       state, created_at_unix_ms
--
-- Returns:
--   CREATED
--   EXISTING:<operation_id>   another invocation already owns the key
--   COLLISION                 the generated operation ID already exists
--   NOGROUP                   the group has not formed yet
--
-- The idempotency reservation is what makes an orchestrator retry safe: a
-- create that timed out client-side but committed server-side returns the
-- original operation instead of opening a second one against the same group.

local existing = redis.call('GET', KEYS[2])
if existing then
  if redis.call('EXISTS', 'mx:refitc:op:' .. existing) == 1 then
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
