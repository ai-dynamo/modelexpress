-- Atomically reserve an idempotency key and open one collective transfer.
--
-- KEYS[1]: operation hash
-- KEYS[2]: create-request idempotency key
-- KEYS[3]: group hash
-- KEYS[4]: group active-operations set
-- ARGV: operation_id, group_id, version_id, model_name, idempotency_key,
--       state, created_at_unix_ms, operation_key_prefix, idempotency_ttl_secs,
--       expected_epoch
--
-- Returns:
--   CREATED
--   EXISTING:<operation_id>   another invocation already owns the key
--   COLLISION                 the generated operation ID already exists
--   NOGROUP                   no group exists for this membership
--   NOTREADY:<epoch>:<state>  the group is not READY at the expected epoch
--   BUSY:<operation_id>       another operation is still in flight at this epoch
--
-- The idempotency reservation is what makes an orchestrator retry safe: a
-- create that timed out client-side but committed server-side returns the
-- original operation instead of opening a second one against the same group.
--
-- The reservation key also carries a TTL: a worker that dies between this
-- call and the next epoch bump would otherwise leak the key forever, since
-- nothing but a terminal delete ever removes it. The active-operations set
-- is the fast path that lets an epoch bump find and abort this operation
-- before that TTL would matter.

local existing = redis.call('GET', KEYS[2])
if existing then
  if redis.call('EXISTS', ARGV[8] .. existing) == 1 then
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

-- READY admission and epoch binding are one decision. The caller names the
-- READY epoch it checked; if the group has since moved on (or never got
-- there), nothing is written: no operation that would block the next create,
-- and no reservation that would replay this refusal's key.
local group_state = redis.call('HGET', KEYS[3], 'state') or ''
if group_state ~= 'READY' or tonumber(epoch) ~= tonumber(ARGV[10]) then
  return 'NOTREADY:' .. epoch .. ':' .. group_state
end

-- At most one nonterminal operation per group epoch. Two operations on one
-- epoch would ride the same lane communicators, and a split cohort (some
-- workers bound to each) would enter collectives its peers are not in. A
-- same-key replay was already answered above, so anything still in flight
-- here belongs to a different request.
local active = redis.call('SMEMBERS', KEYS[4])
for i = 1, #active do
  local fields = redis.call('HMGET', ARGV[8] .. active[i], 'state', 'epoch')
  local op_state, op_epoch = fields[1], fields[2]
  if (op_state == 'PENDING' or op_state == 'RUNNING')
      and tonumber(op_epoch) == tonumber(epoch) then
    return 'BUSY:' .. active[i]
  end
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
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[9])
redis.call('SADD', KEYS[4], ARGV[1])

return 'CREATED'
